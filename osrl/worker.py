"""Intent worker: the ONLY path from a pending intent to a real mutation.

Autonomy gates (Phase 2 ladder rung 0 -> 1):

- The collector daemon NEVER imports or starts this module. No autonomous
  execution loop exists.
- Dry mode (default): peek pending intents, print guard verdicts, claim
  nothing, mutate nothing.
- Execute mode requires the operator's explicit ``--execute`` flag PER
  INVOCATION: that token IS the human approval gate for this phase. (The
  dashboard's per-intent approval UI arrives in Phase 5 and will write
  approval decisions into intent payloads; the guard then consumes them
  the same way.)

Per intent, atomically:
  claim (compare-and-set) -> guard.check -> on deny: ledger 'rejected'
  + intent 'rejected' -> on allow: pidfd-pinned executor apply ->
  ledger 'applied' + intent 'applied', or typed failure -> ledger
  'failed' + intent 'failed'. Every transition writes events.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path

from osrl.config import load_config
from osrl.executor import (
    ExecutorError,
    apply_renice_back,
    apply_restore,
)
from osrl.guard import ActionRequest, Guard, GuardDeps
from osrl.pidfd import TargetGone
from osrl.store import Store

_ADMITTED_KINDS = ("renice_back", "restore")


def build_request_from_intent(intent_row) -> ActionRequest | None:
    """Validate the intent payload into a request, or None (malformed)."""
    it = dict(intent_row)
    payload = it.get("payload")
    try:
        data = json.loads(payload) if isinstance(payload, str) else dict(payload)
        if not isinstance(data, dict):
            return None
    except (TypeError, ValueError):
        return None
    try:
        pid = int(data["pid"])
        start_time = float(data["start_time"])
        kind = str(it["kind"])
        if isinstance(data["pid"], bool) or float(data["pid"]) != pid:
            return None
        if not math.isfinite(start_time) or start_time <= 0:
            return None
        target_nice = data.get("target_nice")
        if target_nice is not None:
            target_nice = int(target_nice)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    parent = data.get("parent_action_id")
    return ActionRequest(
        kind=kind,
        pid=pid,
        start_time=start_time,
        parent_action_id=str(parent) if parent else None,
        target_nice=int(target_nice) if target_nice is not None else None,
        proposed_by=str(it.get("claimed_by") or "intent"),
    )


def process_one_intent(
    store: Store,
    guard: Guard,
    guard_cfg,
    intent_row,
    now: float,
    worker_id: str,
    clock=time.time,
) -> dict:
    """Claim-consume ONE claimed-by-us intent through guard + executor."""
    request = build_request_from_intent(intent_row)
    if request is None:
        store.finish_intent(
            intent_row["id"], "rejected",
            "malformed intent payload: expected keys"
            " {kind: pid, start_time[, parent_action_id, target_nice]}",
            now,
        )
        store.record_event(now, "action_rejected",
                           {"intent_id": intent_row["id"],
                            "reason": "malformed_payload"})
        return {"intent_id": intent_row["id"], "outcome": "rejected",
                "reason": "malformed_payload"}

    verdict = guard.check(request, now, intent=intent_row, worker_id=worker_id)
    row_id = str(uuid.uuid4())
    # Pre-apply nice read: recorded as the ledger's `before_nice` audit
    # column (evidence pipeline pairs need the delta). A second read
    # happens at apply time; if it fails, the typed executor error path
    # below still records the failure honestly.
    pre_nice = guard.d.read_nice(request.pid) if request.kind in _ADMITTED_KINDS else None

    def base_row(status: str, error: str | None) -> dict:
        return {
            "id": row_id,
            "created_ts": now,
            "proposed_by": request.proposed_by,
            "kind": request.kind,
            "target_pid": request.pid,
            "target_start_time": request.start_time,
            "target_nice": verdict.target_nice
            if request.kind == "renice_back"
            else request.target_nice,
            "before_nice": pre_nice,
            "guard_checks": json.dumps(verdict.as_dict(), sort_keys=True),
            "intent_id": intent_row["id"],
            "status": status,
            "error": error,
        }

    if not verdict.ok:
        store.insert_action(**base_row("rejected", verdict.reason))
        store.finish_intent(intent_row["id"], "rejected", verdict.reason, now)
        store.record_event(now, "action_rejected",
                           {"intent_id": intent_row["id"],
                            "action_id": row_id,
                            "kind": request.kind,
                            "reason": verdict.reason})
        return {"intent_id": intent_row["id"], "outcome": "rejected",
                "reason": verdict.reason}

    # Approved path: ledger 'approved' row BEFORE any mutation, so a crash
    # mid-apply leaves an auditable intent chain; the state machine then
    # advances via the CAS updates.
    store.insert_action(**base_row("approved", None))
    try:
        current = guard.d.read_nice(request.pid)
        if current is None:
            raise ExecutorError("nice unreadable at apply time")
        if request.kind == "renice_back":
            applied = apply_renice_back(
                request.pid,
                int(verdict.target_nice),
                request.start_time,
                current_nice=current,
            )
        else:
            applied = apply_restore(
                request.pid,
                int(request.target_nice),
                request.start_time,
                current_nice=current,
                capable=guard_cfg.restore_capable,
            )
    except (ExecutorError, TargetGone) as exc:
        store.mark_action_failed(row_id, clock(), str(exc))
        store.finish_intent(intent_row["id"], "failed", str(exc), now)
        store.record_event(now, "action_failed",
                           {"intent_id": intent_row["id"],
                            "action_id": row_id,
                            "kind": request.kind,
                            "error": str(exc)})
        return {"intent_id": intent_row["id"], "outcome": "failed",
                "reason": str(exc)}

    store.mark_action_applied(row_id, clock(), applied)
    store.finish_intent(
        intent_row["id"], "applied",
        {"action_id": row_id, "applied_nice": applied}, now,
    )
    store.record_event(now, "action_applied",
                       {"intent_id": intent_row["id"],
                        "action_id": row_id,
                        "kind": request.kind,
                        "pid": request.pid,
                        "applied_nice": applied})
    return {"intent_id": intent_row["id"], "outcome": "applied",
            "applied_nice": applied}


def process_pending(
    store: Store,
    guard: Guard,
    guard_cfg,
    now: float,
    worker_id: str = "guard-worker",
    execute: bool = False,
    limit: int = 20,
) -> list[dict]:
    """Consume up to `limit` pending intents (execute mode) or peek them."""
    if not execute:
        rows = store._conn.execute(
            "SELECT id, kind, payload, status FROM intents"
            " WHERE status='pending' ORDER BY created_ts, id LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for row in rows:
            request = build_request_from_intent(row)
            if request is None:
                out.append({"intent_id": row["id"], "outcome": "peek",
                            "reason": "malformed_payload"})
                continue
            verdict = guard.check(request, now)
            out.append({
                "intent_id": row["id"], "outcome": "peek",
                "verdict": "allow" if verdict.ok else "deny",
                "reason": verdict.reason or "all checks pass",
            })
        return out

    out = []
    for kind in _ADMITTED_KINDS:
        while True:
            claimed = store.claim_next_pending(kind, worker_id, now)
            if claimed is None:
                break
            if len(out) >= limit:
                # release the claim so a later operator run can take it
                store.release_claim(claimed["id"], now, "worker limit reached")
                out.append({"intent_id": claimed["id"],
                            "outcome": "released", "reason": "limit"})
                break
            out.append(process_one_intent(
                store, guard, guard_cfg, claimed, now, worker_id))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Guarded intent worker (explicit approval required)")
    parser.add_argument("--db", help="path to SQLite db (overrides OSRL_DB)")
    parser.add_argument("--execute", action="store_true",
                        help="EXPLICIT operator approval: apply allowed"
                             " intents for real this invocation")
    parser.add_argument("--worker", default="guard-worker",
                        help="worker id recorded on claims")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    cfg = load_config()
    db_path = Path(args.db).expanduser() if args.db else cfg.db_path
    store = Store(db_path)
    guard = Guard(
        cfg.guard,
        GuardDeps(),
        sample_staleness=store.sample_staleness,
        last_action_ts=store.last_action_ts,
        read_action=store.action_row,
    )
    try:
        summaries = process_pending(
            store, guard, cfg.guard,
            now=time.time(), worker_id=args.worker,
            execute=args.execute, limit=args.limit,
        )
    finally:
        store.close()
    for s in summaries:
        print(json.dumps(s, sort_keys=True))
    if not summaries:
        print("no pending intents" if not args.execute
              else "nothing admitted for execution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
