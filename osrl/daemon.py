"""Collector daemon: runs the sampler forever, independently of the dashboard.

Operational contracts (brief G, DESIGN section 8):

- Single instance enforced with flock on ``<db-dir>/collector.lock`` held for
  process lifetime; a second instance exits with code 4.
- Batched writes: up to ``batch_size`` samples or ``batch_interval_s`` alive,
  flushed in one short transaction. A crash loses at most the unflushed batch
  (documented), never corrupts the database.
- Periodic jobs: heartbeat (default 30 s), intent-lease reconciliation,
  15-min rollups over completed windows, hourly retention prune, hourly WAL
  checkpoint.
- Graceful shutdown on SIGINT/SIGTERM: flush, checkpoint(TRUNCATE), close,
  release lock, exit 0. Practical data-loss bound stays the unflushed batch.
- Self-overhead measured (os.times delta) and published in the heartbeat
  payload so the "<1% collector CPU" budget is a measured fact, not an
  assumption.

Phase 1 additions (observe-only):

- RiskTracker consumes each sample; level changes are recorded as
  ``risk_change`` events and the current level ships in every heartbeat.
- A dedicated poller thread arms PSI monitors (memory some / io full),
  records ``psi_trigger_fired``/``psi_trigger_failed``/``psi_trigger_dead``
  events through its OWN store connection (thread-local sqlite contract),
  and forwards memory firings to the risk tracker as extra warn votes.
  Poller failure is recorded as ``psi_poller_error`` and can never take
  sampling down.

The daemon NEVER consumes intents into actions -- Phase 2's policy worker
does that behind the guard. The collector only reconciles stale claims so a
crashed dashboard/policy process cannot hold a lease forever.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from osrl.adapters import default_providers
from osrl.collector import Collector, ProcEvent, Providers, Sample
from osrl.config import Config, load_config
from osrl.psi_trigger import PsiPoller, PsiRegistrationError, PsiTrigger, TriggerSpec
from osrl.risk import RiskTracker
from osrl.store import Store


def acquire_lock(lock_path: Path):
    """Exclusive non-blocking lock; returns the open fd (held until exit)."""
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as err:
        os.close(fd)
        raise SystemExit(
            f"another collector instance holds {lock_path}; refusing to start"
        ) from err
    return fd


def run(
    cfg: Config,
    providers: Providers | None = None,
    once: bool = False,
    register_signals: bool = True,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    stop: Callable[[], bool] | None = None,
) -> None:
    interval = cfg.collector.interval_s
    ret = cfg.retention
    lock_dir = cfg.db_path.parent
    lock_dir.mkdir(parents=True, exist_ok=True)  # lock file lives next to the db
    lock_fd = acquire_lock(lock_dir / "collector.lock")
    try:
        _run_locked(cfg, providers, once, register_signals, clock, sleep, stop,
                    interval, ret)
    finally:
        os.close(lock_fd)


def _run_locked(
    cfg, providers, once, register_signals, clock, sleep, stop, interval, ret
) -> None:
    store = Store(cfg.db_path)
    if providers is None:
        providers = default_providers()
    collector = Collector(
        providers,
        heavy_cpu_pct=cfg.collector.heavy_cpu_pct,
        heavy_rss_mib=cfg.collector.heavy_rss_mib,
        top_k_procs=cfg.collector.top_k_procs,
    )
    risk_tracker = RiskTracker(cfg.risk)
    ncpu = os.cpu_count() or 1

    batch: list[Sample] = []
    last_sample_ts: float | None = None
    now = clock()
    last_flush = now
    last_hb = now
    last_ckpt = now
    last_ret = now
    tms_prev = os.times()

    rollup_until = store.rollup_until()
    if rollup_until is None:
        first_boundary = _completed_boundary(now, ret.rollup_seconds)
        store.set_rollup_until(first_boundary)
        rollup_until = first_boundary

    # PSI monitors run beside the sampler, not inside it.
    psi_stop = threading.Event()
    psi_thread = start_psi_poller_thread(cfg, psi_stop, risk_tracker)

    running = True

    def _sig_handler(signum, frame):  # noqa: ARG001 - signal API
        nonlocal running
        running = False

    if register_signals:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _sig_handler)
            except ValueError:
                pass  # not in the main thread (tests); stop() callable handles it

    try:
        while running and not (stop and stop()):
            if last_sample_ts is not None:
                delay = last_sample_ts + interval - clock()
                if delay > 0:
                    sleep(delay)
            ts = clock()
            sample, events = collector.tick(ts)
            last_sample_ts = ts
            batch.append(sample)
            for ev in events:
                _record_proc_event(store, ev)

            assessment = risk_tracker.feed(sample, ts)
            if assessment.changed:
                store.record_event(ts, "risk_change", {
                    "level": assessment.level,
                    "strong": assessment.strong,
                    "warn": assessment.warn,
                    "psi_present": assessment.psi_present,
                })

            if (
                len(batch) >= ret.batch_size
                or (ts - last_flush) >= ret.batch_interval_s
            ):
                store.insert_samples(batch)
                batch.clear()
                last_flush = ts

            if (ts - last_hb) >= ret.heartbeat_s:
                tms = os.times()
                dt = ts - last_hb or interval
                own_cpu_pct = (
                    ((tms.user + tms.system)
                     - (tms_prev.user + tms_prev.system))
                    / max(dt, 1e-9) / ncpu * 100.0
                )
                tms_prev = tms
                store.heartbeat(
                    ts,
                    os.getpid(),
                    {
                        "interval_s": interval,
                        "collect_ms": sample.collect_ms,
                        "own_cpu_pct": own_cpu_pct,
                        "samples_in_db": store.count_samples(),
                        "risk_level": assessment.level,
                        "pending_intents": store.pending_intent_count(),
                    },
                )
                for intent_id, action in store.reconcile_intents(ts):
                    store.record_event(
                        ts, "intent_reconciled",
                        {"intent_id": intent_id, "action": action},
                    )
                last_hb = ts

            if (ts - last_ckpt) >= ret.wal_checkpoint_s:
                store.wal_checkpoint()
                last_ckpt = ts

            if (ts - last_ret) >= 3600.0:
                store.prune(ts, cfg.retention.raw_days, cfg.retention.events_days)
                last_ret = ts

            if batch and ts >= rollup_until + ret.rollup_seconds:
                store.insert_samples(batch)
                batch.clear()
                last_flush = ts
            flush_grace = max(
                15.0, ret.batch_interval_s + cfg.collector.interval_s
            )
            for outcome in store.finalize_action_outcomes(
                ts, risk_config=cfg.risk, flush_grace_s=flush_grace
            ):
                store.record_event(ts, "action_outcome_ready", outcome)
            rollup_until = _rollup_complete_windows(store, ts, rollup_until,
                                                    ret.rollup_seconds)

            if once:
                break
    finally:
        psi_stop.set()
        if psi_thread is not None:
            psi_thread.join(timeout=3.0)  # poller waits <= probe_s per poll()
        if batch:
            try:
                store.insert_samples(batch)
            except Exception:  # noqa: BLE001 - keep the lock dance intact
                pass
        store.close()


def _psi_path(resource: str) -> Path:
    return Path(f"/proc/pressure/{resource}")


def start_psi_poller_thread(
    cfg: Config,
    stop_event: threading.Event,
    risk_tracker: RiskTracker,
) -> threading.Thread | None:
    """Arm PSI monitors and record firings as events.

    Runs its OWN store connection (sqlite connections are not thread-safe
    across users); any failure is recorded and the sampling main thread
    remains unaffected. Returns the started thread, or None when
    psi_trigger.enabled is False.
    """
    if not cfg.psi_trigger.enabled:
        return None

    def _body() -> None:
        store: Store | None = None
        poller: PsiPoller | None = None
        try:
            store = Store(cfg.db_path)  # dedicated thread-local connection

            def on_event(trig: PsiTrigger, ts: float) -> None:
                try:
                    store.record_event(
                        ts, "psi_trigger_fired", trig.spec.describe()
                    )
                finally:
                    risk_tracker.note_trigger(trig.spec.resource, ts)

            def on_dead(trig: PsiTrigger, ts: float) -> None:
                store.record_event(ts, "psi_trigger_dead", trig.spec.describe())

            armed: list[PsiTrigger] = []
            psi = cfg.psi_trigger
            specs = (
                TriggerSpec(
                    "memory", "some", psi.memory_some_us, psi.memory_window_us
                ),
                TriggerSpec("io", "full", psi.io_full_us, psi.io_window_us),
            )
            for spec in specs:
                try:
                    armed.append(PsiTrigger(spec, _psi_path(spec.resource)))
                    store.record_event(
                        time.time(), "psi_trigger_registered",
                        spec.describe(),
                    )
                except PsiRegistrationError as err:
                    store.record_event(
                        time.time(), "psi_trigger_failed",
                        {**spec.describe(), "error": str(err)},
                    )

            poller = PsiPoller(
                armed,
                on_event=on_event,
                on_dead=on_dead,
                probe_s=psi.probe_interval_s,
            )
            poller.run(stop_event=stop_event)
        except Exception as exc:  # noqa: BLE001 - fail loudly into the ledger
            try:
                if store is not None:
                    store.record_event(
                        time.time(), "psi_poller_error", {"error": repr(exc)}
                    )
            except Exception:  # noqa: BLE001 - diagnostics must not crash
                pass
        finally:
            if poller is not None:
                poller.close()
            if store is not None:
                store.close()

    thread = threading.Thread(target=_body, name="psi-poller", daemon=True)
    thread.start()
    return thread


def _record_proc_event(store: Store, ev: ProcEvent) -> None:
    payload = {
        "pid": ev.pid,
        "start_time": ev.start_time,
        "name": ev.name,
        **ev.detail,
    }
    store.record_event(ev.ts, f"proc_{ev.kind}", payload)


def _completed_boundary(now: float, rollup_seconds: int) -> float:
    """Start of the current (still incomplete) rollup window on the
    epoch-anchored grid; the daemon's marker for 'nothing done yet'."""
    return math.floor(now / rollup_seconds) * rollup_seconds


def _rollup_complete_windows(
    store: Store, now: float, rollup_until: float, rollup_seconds: int
) -> float:
    """Aggregate every window fully completed since `rollup_until`.

    Bounded to 96 windows/loop to survive pathological backlog without a
    run-free loop. Windows whose sample data is missing (daemon downtime)
    are inserted as sample_count=0 rows so the rollup history stays dense.
    """
    processed = 0
    while processed < 96:
        win_end = rollup_until + rollup_seconds
        if win_end > now:
            break
        agg = store.compute_rollup(rollup_until, win_end)
        if agg is None:
            agg = {
                "window_start": rollup_until,
                "window_end": win_end,
                "sample_count": 0,
                "cpu_total_pct_avg": None, "cpu_total_pct_max": None,
                "mem_used_pct_avg": None, "mem_available_bytes_avg": None,
                "swap_used_pct_avg": None, "swap_used_pct_max": None,
                "swp_out_ps_avg": None, "swp_out_ps_max": None,
                "psi_mem_stall_s_sum": None, "psi_io_stall_s_sum": None,
            }
        store.insert_rollup(agg)
        rollup_until = win_end
        processed += 1
    store.set_rollup_until(rollup_until)
    return rollup_until


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OS-RL-Agent collector daemon")
    parser.add_argument("--db", help="path to SQLite db (overrides OSRL_DB)")
    parser.add_argument("--interval", type=float, help="sample period seconds")
    parser.add_argument("--once", action="store_true",
                        help="sample one tick, flush, exit (smoke mode)")
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.interval is not None and (not math.isfinite(args.interval) or args.interval <= 0):
        parser.error("--interval must be a finite positive number")
    cfg = replace(
        cfg,
        db_path=Path(args.db).expanduser() if args.db else cfg.db_path,
        collector=replace(cfg.collector, interval_s=(
            args.interval if args.interval is not None else cfg.collector.interval_s
        )),
    )
    try:
        run(cfg, once=args.once)
    except KeyboardInterrupt:
        pass  # SIGINT handler already flushed; treat as clean stop
    return 0


if __name__ == "__main__":
    sys.exit(main())
