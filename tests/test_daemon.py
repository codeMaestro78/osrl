"""Daemon lifecycle: single instance, batching, heartbeat, graceful stop."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from osrl.config import (
    CollectorConfig,
    Config,
    RetentionConfig,
    RiskConfig,
)
from osrl.daemon import acquire_lock, run
from osrl.store import Store
from tests.conftest import MachineFixture, make_sample


def make_cfg(tmp_path: Path, interval: float = 0.05) -> Config:
    return Config(
        db_path=tmp_path / "data" / "telemetry.db",
        collector=CollectorConfig(interval_s=interval),
        retention=RetentionConfig(
            heartbeat_s=0.02,
            batch_size=2,
            batch_interval_s=0.2,
            wal_checkpoint_s=3600.0,
        ),
    )


def test_run_once_persists_sample(tmp_path, machine):
    machine.set_proc(1, 100.0, "proc-a", cpu_time_s=0.0, rss_mib=5.0)
    cfg = make_cfg(tmp_path)
    run(cfg, providers=machine.providers(), once=True, register_signals=False)

    store = Store(cfg.db_path)
    try:
        assert store.count_samples() == 1  # finally-flush covered the batch
        row = store._conn.execute("SELECT * FROM samples").fetchone()
        assert row["ts"] is not None
    finally:
        store.close()


def test_run_background_ticks_stop_flush_and_heartbeat(tmp_path, machine):
    """Three paced ticks: stop-check counting ends the loop; interval pacing
    and advanceable clock make heartbeat timing deterministic."""
    machine.set_proc(1, 100.0, "proc-a", cpu_time_s=0.0, rss_mib=5.0)
    cfg = make_cfg(tmp_path)

    class FakeClock:
        def __init__(self) -> None:
            self.t = 1000.0

        def __call__(self) -> float:
            return self.t

        def advance(self, seconds: float) -> None:
            self.t += seconds

    clk = FakeClock()

    def sleep(seconds: float) -> None:
        clk.advance(max(seconds, 0.0))

    checks = {"n": 0}

    def stop():
        checks["n"] += 1
        return checks["n"] >= 4  # 4th evaluation = after 3 completed ticks

    run(
        cfg,
        providers=machine.providers(),
        once=False,
        register_signals=False,
        clock=clk,
        sleep=sleep,
        stop=stop,
    )

    store = Store(cfg.db_path)
    try:
        assert store.count_samples() == 3
        # ticks at 1000.0 / 1000.05 / 1000.10 -> heartbeat (0.02 s period) fired
        row = store._conn.execute(
            "SELECT * FROM heartbeat WHERE id=1"
        ).fetchone()
        assert row is not None
        assert row["ts"] == pytest.approx(1000.1)
        payload = json.loads(row["payload"])
        assert payload["interval_s"] == 0.05
        assert "collect_ms" in payload and "own_cpu_pct" in payload
        marks = [
            float(r["ts"]) for r in store._conn.execute(
                "SELECT ts FROM samples ORDER BY ts"
            ).fetchall()
        ]
        assert marks == pytest.approx([1000.0, 1000.05, 1000.1])
    finally:
        store.close()


def test_second_instance_rejected_while_lock_held(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cfg.db_path.parent / "collector.lock"
    fd = acquire_lock(lock_path)
    try:
        with pytest.raises(SystemExit):
            acquire_lock(lock_path)
        with pytest.raises(SystemExit):
            run(cfg, providers=MachineFixture(tmp_path).providers(),
                once=True, register_signals=False)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_proc_events_recorded_to_store(tmp_path, machine):
    machine.set_proc(9, 100.0, "heavy-worker", cpu_time_s=0.0, rss_mib=600.0)
    cfg = make_cfg(tmp_path)
    run(cfg, providers=machine.providers(), once=True, register_signals=False)
    store = Store(cfg.db_path)
    try:
        kinds = [r["kind"] for r in
                 store._conn.execute("SELECT kind FROM events").fetchall()]
        assert "proc_birth" in kinds
    finally:
        store.close()


def test_rollup_backlog_dense_zero_sample_windows(tmp_path):
    """Multi-day outage: rollups must fill completed windows with
    sample_count=0 rows (dense history) and advance the marker, bounded."""
    import osrl.daemon as d

    store = Store(tmp_path / "t.db")
    try:
        store.set_rollup_until(0.0)
        store.insert_samples(
            [make_sample(ts=10.0, swp_out_ps=5.0)]
        )  # only data in the first window
        now = 3 * 900 + 10.0
        d._rollup_complete_windows(store, now, 0.0, 900)
        rows = store._conn.execute(
            "SELECT window_start, sample_count FROM rollups ORDER BY window_start"
        ).fetchall()
        assert [r["sample_count"] for r in rows] == [1, 0, 0]
        assert store.rollup_until() == pytest.approx(2700.0)
    finally:
        store.close()


def test_risk_change_event_recorded(tmp_path, machine):
    """Observability check: an escalating synthetic machine writes
    risk_change events; staleness stays out of the way (fresh now==ts)."""
    checks = {"n": 0}

    def stop():
        checks["n"] += 1
        return checks["n"] >= 12  # 11 ticks > enter_n + margin

    import osrl.daemon as d

    real_tick = d.Collector.tick
    samples_seen = {"n": 0}

    def counting_tick(self, ts=None):
        samples_seen["n"] += 1
        # escalate PSI: each tick reports a full-interval memoria stall
        machine.write_psi(
            "memory", 100_000 + samples_seen["n"] * 1_000_000
        )
        return real_tick(self, ts)

    d.Collector.tick = counting_tick
    try:
        cfg = Config(
            db_path=tmp_path / "risk.db",
            collector=CollectorConfig(interval_s=0.05),
            retention=RetentionConfig(
                heartbeat_s=0.02, batch_size=1, batch_interval_s=0.05,
                wal_checkpoint_s=30.0,
            ),
            risk=RiskConfig(staleness_s=0.2, enter_n=3, exit_n=6),
        )
        run(cfg, providers=machine.providers(), register_signals=False,
            stop=stop)
    finally:
        d.Collector.tick = real_tick

    assert samples_seen["n"] >= 5, "risk needs consecutive samples"
    store = Store(cfg.db_path)
    try:
        kinds = [r["kind"] for r in
                 store._conn.execute("SELECT kind FROM events").fetchall()]
        assert "risk_change" in kinds
        row = store._conn.execute(
            "SELECT payload FROM events WHERE kind='risk_change'"
        ).fetchone()
        assert row is not None
        payload = json.loads(row["payload"])
        assert payload["level"] in ("elevated", "critical")
        assert payload["psi_present"] in (True, False)
    finally:
        store.close()


@pytest.mark.skipif(
    not Path("/proc/pressure/cpu").exists(),
    reason="real-system smoke requires /proc/pressure",
)
def test_subprocess_smoke_on_real_machine(tmp_path):
    """End-to-end on this machine: CLI entry, real collector, real store."""
    out_db = tmp_path / "smoke.db"
    proc = subprocess.run(
        [sys.executable, "-m", "osrl.daemon", "--db", str(out_db),
         "--interval", "0.2", "--once"],
        capture_output=True, text=True, timeout=60,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, proc.stderr
    store = Store(out_db)
    try:
        assert store.count_samples() == 1
        row = store._conn.execute("SELECT * FROM samples").fetchone()
        assert row["ncpu"] is not None and row["ncpu"] >= 1
        assert row["mem_total_bytes"] and row["mem_total_bytes"] > 0
    finally:
        store.close()


@pytest.mark.skipif(
    not Path("/proc/pressure/cpu").exists(),
    reason="lock roundtrip requires a real filesystem for flock",
)
def test_subprocess_second_instance_respects_lock(tmp_path):
    """A running daemon holds the collector lock; a rival must exit code 4."""
    holder_db = tmp_path / "lock-test.db"
    # the daemon locks next to ITS OWN db path
    lock = holder_db.parent / "collector.lock"
    holder = subprocess.Popen(
        [sys.executable, "-m", "osrl.daemon", "--db", str(holder_db),
         "--interval", "30"],  # long-running: holds the lock until killed
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    try:
        # Wait until we can independently confirm the lock is held.
        held = False
        for _ in range(100):
            assert holder.poll() is None, "holder died unexpectedly"
            try:
                probe_fd = acquire_lock(lock)
                fcntl.flock(probe_fd, fcntl.LOCK_UN)
                os.close(probe_fd)
                time.sleep(0.05)
            except SystemExit:
                held = True
                break
        assert held, "holder never acquired the lock"

        rival = subprocess.run(
            [sys.executable, "-m", "osrl.daemon", "--db", str(holder_db),
             "--interval", "30"],
            capture_output=True, text=True, timeout=20,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        assert rival.returncode != 0
        assert "refusing" in rival.stderr
    finally:
        holder.kill()
        holder.wait(timeout=15)
