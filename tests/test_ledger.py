"""Actions ledger: strict columns, state machine, rate-limit query, migration
from the legacy v1 shape, pruning, and intent-count observability."""

import json
import time
import uuid

import pytest

from osrl.store import Store


def test_ledger_roundtrip_and_state_chain(tmp_path):
    store = Store(tmp_path / "t.db")
    try:
        rid = str(uuid.uuid4())
        now = time.time()
        checks = json.dumps([{"name": "identity", "passed": True, "detail": ""}])
        # worker flow: the guard-approved row lands as 'approved', then the
        # executor's CAS advance moves it to 'applied' with the read-back.
        ok = store.insert_action(
            id=rid, created_ts=now, proposed_by="test", kind="renice_back",
            target_pid=42, target_start_time=1.0, target_nice=15,
            guard_checks=checks, status="approved", intent_id="i-1",
        )
        assert ok
        row = store.action_row(rid)
        assert row["status"] == "approved"
        assert row["applied_nice"] is None

        ts = time.time()
        assert store.mark_action_applied(rid, ts, 15)
        row = store.action_row(rid)
        assert row["status"] == "applied"
        assert row["applied_nice"] == 15
        assert store.last_action_ts(42, "renice_back") == pytest.approx(now)

        # unknown columns are rejected loudly
        with pytest.raises(ValueError, match="unknown ledger columns"):
            store.insert_action(id="x", created_ts=now, proposed_by="t",
                                kind="renice_back", hacker_col=1)

        # failed attempts count for the rate limiter too
        rid2 = str(uuid.uuid4())
        store.insert_action(id=rid2, created_ts=now + 5, proposed_by="t",
                            kind="renice_back", target_pid=42,
                            target_start_time=1.0, status="failed",
                            error="boom")
        assert store.last_action_ts(42, "renice_back") == pytest.approx(now + 5)

        assert [a["id"] for a in store.actions_for_intent("i-1")] == [rid]
        assert store.actions_for_intent("gone") == []
    finally:
        store.close()


def test_mark_action_applied_requires_pre_state(tmp_path):
    store = Store(tmp_path / "t.db")
    try:
        rid = str(uuid.uuid4())
        store.insert_action(id=rid, created_ts=time.time(), proposed_by="t",
                            kind="renice_back", target_pid=1,
                            target_start_time=0.0, guard_checks="[]",
                            status="applied")
        # already applied (CAS against proposed/approved only)
        assert not store.mark_action_applied(rid, time.time(), 10)
    finally:
        store.close()


def test_release_claim_returns_to_pending(tmp_path):
    store = Store(tmp_path / "t.db")
    try:
        iid = str(uuid.uuid4())
        store.insert_intent(iid, "renice_back", {"pid": 5}, time.time())
        claimed = store.claim_next_pending("renice_back", "w", time.time())
        assert claimed["status"] == "claimed"
        assert store.release_claim(iid, time.time(), "worker limit reached")
        row = store._conn.execute(
            "SELECT status, claimed_by, result FROM intents WHERE id=?",
            (iid,),
        ).fetchone()
        assert row["status"] == "pending" and row["claimed_by"] is None
        assert row["result"] == "released: worker limit reached"
    finally:
        store.close()


def test_pending_intent_count(tmp_path):
    store = Store(tmp_path / "t.db")
    try:
        store.insert_intent("a", "renice_back", {}, time.time())
        store.insert_intent("b", "renice_back", {}, time.time())
        assert store.pending_intent_count() == 2
        store.claim_next_pending("renice_back", "w", time.time())
        assert store.pending_intent_count() == 1
    finally:
        store.close()


def test_prune_removes_old_ledger_rows(tmp_path):
    store = Store(tmp_path / "t.db")
    try:
        now = time.time()
        store.insert_action(id="old", created_ts=now - 40 * 86400,
                            proposed_by="t", kind="renice_back")
        store.insert_action(id="new", created_ts=now,
                            proposed_by="t", kind="renice_back")
        out = store.prune(now, raw_days=7, events_days=30)
        assert out["actions_deleted"] == 1
        assert store.action_row("new")["id"] == "new"
        assert store.action_row("old") is None
    finally:
        store.close()


def _create_legacy_v1_db(path):
    """Rebuild the EXACT v1 layout (dead-schema ledger) to prove migration."""
    import sqlite3

    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE samples (
        id INTEGER PRIMARY KEY, ts REAL NOT NULL, interval_s REAL, ncpu INTEGER,
        cpu_total_pct REAL, cpu_percpu_json TEXT, load1 REAL, load5 REAL,
        load15 REAL, mem_total_bytes INTEGER, mem_available_bytes INTEGER,
        mem_used_pct REAL, swap_total_bytes INTEGER, swap_used_bytes INTEGER,
        swap_used_pct REAL, swp_in_ps REAL, swp_out_ps REAL, majflt_ps REAL,
        reclaim_scan_ps REAL, reclaim_steal_ps REAL, allocstall_ps REAL,
        psi_cpu_some_total_us INTEGER, psi_cpu_some_stall_us INTEGER,
        psi_cpu_full_total_us INTEGER, psi_cpu_full_stall_us INTEGER,
        psi_mem_some_total_us INTEGER, psi_mem_some_stall_us INTEGER,
        psi_mem_full_total_us INTEGER, psi_mem_full_stall_us INTEGER,
        psi_io_some_total_us INTEGER, psi_io_some_stall_us INTEGER,
        psi_io_full_total_us INTEGER, psi_io_full_stall_us INTEGER,
        gpu_util_pct REAL, gpu_mem_used_mib REAL, gpu_mem_total_mib REAL,
        gpu_temp_c REAL, gpu_power_w REAL, top_procs_json TEXT, collect_ms REAL
    );
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE actions_ledger (
        id TEXT PRIMARY KEY, created_ts REAL NOT NULL, proposed_by TEXT
            NOT NULL, kind TEXT NOT NULL, target_pid INTEGER,
        target_start_time REAL, payload TEXT,
        status TEXT NOT NULL DEFAULT 'proposed'
    );
    """)
    conn.execute(
        "INSERT INTO actions_ledger (id, created_ts, proposed_by, kind,"
        " target_pid, target_start_time, payload, status)"
        " VALUES ('legacy-1', 0.0, 'ancient', 'renice_back', 7, 1.0,"
        " '{}', 'applied')"
    )
    conn.execute("INSERT INTO meta (key,value) VALUES ('rollup_until','0.0')")
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    conn.close()


def test_v1_to_v2_migration_preserves_ledger_rows(tmp_path):
    db = tmp_path / "legacy.db"
    _create_legacy_v1_db(db)
    store = Store(db)
    try:
        ver = store._conn.execute("PRAGMA user_version").fetchone()[0]
        assert ver == 2
        lead = store.action_row("legacy-1")
        assert lead["status"] == "applied"
        assert lead["target_pid"] == 7
        assert lead["applied_nice"] is None  # new column backfills NULL

        # new float columns write fine post-migration
        store.insert_action(id="fresh", created_ts=time.time(),
                            proposed_by="m", kind="restore", target_pid=7,
                            target_start_time=1.0, target_nice=9,
                            before_nice=19, status="failed", error="e")
        assert store.action_row("fresh")["before_nice"] == 19
    finally:
        store.close()


def test_future_schema_version_is_hard_error(tmp_path):
    import sqlite3

    db = tmp_path / "future.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version=99")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="newer than supported"):
        Store(db)
