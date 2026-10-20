"""Deterministic action guard: every live action must pass ALL checks here.

Structural guarantees (enforced by tests, not convention):

- This module NEVER imports (directly or transitively) any RL/model module:
  torch, stable_baselines, sb3-contrib, gymnasium, osrl.rl. The safety path
  cannot depend on RL inference, period (acceptance #1).
- Verdicts are total functions of (request, machine state, config): no
  randomness, no inference, no network.

Check order — first failure wins and becomes THE recorded reason, so the
ledger always shows exactly one cause:

    1 kind_allowed      kind in {renice_back, restore}
    2 pid_sane          integer pid; never self, never 0/1
    3 identity          (pid, start_time) revalidation holds right now
    4 ownership         target runs as OUR effective uid
    5 telemetry_fresh   store staleness <= staleness_max_s (None = deny:
                        missing telemetry abstains, acceptance #2)
    6/7 capability+state
                        renice_back: current nice readable, raise target =
                        min(current+step, max_nice), never below its floor
                        restore: requires restore_capable=True (empirical
                        2026-10-20: unprivileged lower-nice is EPERM even
                        for self), plus an APPLIED renice_back parent row
                        targeting the SAME (pid, start_time), nice state
                        un-drifted, and a recorded target_nice to lower to
    8 rate_limit        one action of a kind per target per min_spacing_s
    9 intent_lease      live execution consumes a CLAIMED intent we own

A denied action is recorded with its exact reason; the guard never retries,
never escalates to anything the operator did not approve, and never mutates
anything itself — application belongs to osrl.executor, reached only via
osrl.worker with the intent lease in hand.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

from osrl.config import GuardConfig
from osrl.procid import get_psutil_starttime

ACTION_KINDS = ("renice_back", "restore")


@dataclass(frozen=True)
class ActionRequest:
    kind: str
    pid: int
    start_time: float
    request_id: str = ""  # ledger row id; the worker generates it
    proposed_by: str = "policy"
    parent_action_id: str | None = None  # restore chain
    target_nice: int | None = None  # restore: the recorded original nice


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class GuardVerdict:
    ok: bool
    reason: str = ""
    checks: list = field(default_factory=list)
    target_nice: int | None = None  # renice_back: computed target

    def as_dict(self) -> list[dict]:
        return [
            {"name": c.name, "passed": c.passed, "detail": c.detail}
            for c in self.checks
        ]


def _default_target_uid(pid: int) -> int | None:
    """Effective uid of the target via /proc/<pid>/status."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("Uid:"):
                    return int(line.split()[2])  # effective uid column
    except OSError:
        return None
    return None


def _default_read_nice(pid: int) -> int | None:
    try:
        import psutil

        return int(psutil.Process(pid).nice())
    except Exception:  # noqa: BLE001 - unreadable = unknown
        return None


@dataclass(frozen=True)
class GuardDeps:
    """Everything the guard observes; injectable for synthetic tests."""

    self_pid: Callable[[], int] = lambda: os.getpid()
    self_euid: Callable[[], int] = lambda: os.geteuid()
    target_uid: Callable[[int], int | None] = _default_target_uid
    start_time_of: Callable[[int], float | None] = get_psutil_starttime
    read_nice: Callable[[int], int | None] = _default_read_nice


class Guard:
    """Total-function checker; the only component that may say yes."""

    def __init__(
        self,
        cfg: GuardConfig,
        deps: GuardDeps | None = None,
        *,
        sample_staleness: Callable[[float], float | None],
        last_action_ts: Callable[[int, str], float | None],
        read_action: Callable[[str], object | None],
    ) -> None:
        self.cfg = cfg
        self.d = deps or GuardDeps()
        self._staleness = sample_staleness
        self._last_action_ts = last_action_ts
        self._read_action = read_action

    # ------------------------------------------------------------------ main

    def check(
        self,
        req: ActionRequest,
        now: float,
        intent=None,
        worker_id: str | None = None,
    ) -> GuardVerdict:
        checks: list[CheckResult] = []

        def fail(name: str, detail: str) -> GuardVerdict:
            checks.append(CheckResult(name, False, detail))
            return GuardVerdict(ok=False, reason=detail, checks=checks)

        def ok(name: str, detail: str = "") -> None:
            checks.append(CheckResult(name, True, detail))

        # 1 kind ---------------------------------------------------------------
        if req.kind not in ACTION_KINDS:
            return fail("kind_allowed", f"unknown action kind {req.kind!r}")
        ok("kind_allowed", req.kind)

        # 2 pid sanity -----------------------------------------------------------
        if not isinstance(req.pid, int) or req.pid <= 1:
            return fail("pid_sane", f"refusing pid {req.pid!r} (init/invalid)")
        if req.pid == self.d.self_pid():
            return fail("pid_sane", "refusing to target our own process")
        ok("pid_sane", f"pid={req.pid}")

        # 3 identity ---------------------------------------------------------
        st = self.d.start_time_of(req.pid)
        if st is None:
            return fail("identity", f"pid {req.pid} not found")
        if abs(st - req.start_time) > 1e-3:
            return fail(
                "identity",
                f"start_time drifted: live={st:.3f} recorded={req.start_time:.3f}",
            )
        ok("identity")

        # 4 ownership --------------------------------------------------------
        uid = self.d.target_uid(req.pid)
        if uid is None:
            return fail("ownership", "target uid unreadable")
        if uid != self.d.self_euid():
            return fail("ownership", f"target euid {uid} != ours {self.d.self_euid()}")
        ok("ownership")

        # 5 telemetry ----------------------------------------------------------
        staleness = self._staleness(now)
        if staleness is None:
            return fail("telemetry_fresh", "no samples in store")
        if staleness > self.cfg.staleness_max_s:
            return fail(
                "telemetry_fresh",
                f"stale telemetry: {staleness:.1f}s > "
                f"{self.cfg.staleness_max_s:.1f}s",
            )
        ok("telemetry_fresh", f"{staleness:.2f}s")

        # 6/7 capability + state ------------------------------------------------
        current = self.d.read_nice(req.pid)
        if req.kind == "renice_back":
            if current is None:
                return fail("capability", "current nice unreadable")
            if current < 0:
                return fail(
                    "capability",
                    f"target already prioritized (nice={current}); refusing",
                )
            target = min(current + self.cfg.renice_step, self.cfg.renice_max_nice)
            if target <= current:
                return fail("state", "renice_back exhausted (already at nice-max)")
            ok("capability", f"nice {current} -> {target}")
            ok("state")
        else:  # restore -------------------------------------------------------
            if not self.cfg.restore_capable:
                return fail(
                    "capability",
                    "restore_requires_elevation (empirical EPERM 2026-10-20;"
                    " capability disabled by config)",
                )
            parent_raw = self._read_action(req.parent_action_id or "")
            if parent_raw is None:
                return fail("state", "restore parent action row missing")
            p = parent_raw if isinstance(parent_raw, dict) else dict(parent_raw)
            if p.get("kind") != "renice_back" or p.get("status") != "applied":
                return fail(
                    "state",
                    "restore parent is not an applied renice_back: "
                    f"kind={p.get('kind')!r} status={p.get('status')!r}",
                )
            if p.get("target_pid") != req.pid or abs(
                float(p.get("target_start_time") or 0.0) - req.start_time
            ) > 1e-3:
                return fail("state", "restore parent targets a different process")
            if current is None:
                return fail("state", "current nice unreadable")
            if abs(current - int(p.get("applied_nice") or 0)) > 1e-9:
                return fail(
                    "state",
                    f"nice state drifted: current={current}, parent recorded "
                    f"{p.get('applied_nice')}",
                )
            if req.target_nice is None or req.target_nice >= current:
                return fail(
                    "state", "restore must lower to the recorded original nice"
                )
            ok("capability")
            ok("state", f"nice {current} -> {req.target_nice}")

        # 8 rate limit -----------------------------------------------------------
        last = self._last_action_ts(req.pid, req.kind)
        if last is not None and (now - last) < self.cfg.min_spacing_s:
            return fail(
                "rate_limit",
                f"last {req.kind} for pid {req.pid} was {now - last:.0f}s ago"
                f" (< {self.cfg.min_spacing_s:.0f}s spacing)",
            )
        ok("rate_limit")

        # 9 intent lease -----------------------------------------------------------
        if intent is not None:
            it = intent if isinstance(intent, dict) else dict(intent)
            if it.get("kind") != req.kind:
                return fail(
                    "intent_lease",
                    f"intent kind {it.get('kind')!r} != request {req.kind!r}",
                )
            if it.get("status") != "claimed":
                return fail("intent_lease", f"intent status {it.get('status')!r}")
            if not worker_id or it.get("claimed_by") != worker_id:
                return fail("intent_lease", "claimed by someone else")
        ok("intent_lease")

        target_nice = (
            min(current + self.cfg.renice_step, self.cfg.renice_max_nice)
            if req.kind == "renice_back"
            else req.target_nice
        )
        return GuardVerdict(ok=True, checks=checks, target_nice=target_nice)
