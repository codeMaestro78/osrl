"""SQLite store semantics: WAL, retention, rollups, heartbeat, intent channel."""

import pytest

from osrl.collector import PsiInterval
from osrl.store import Store
from tests.conftest import make_sample


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "telemetry.db")
    yield s
    s.close()


def test_wal_mode_and_schema_version(store):
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == Store.SCHEMA_VERSION


def test_reopening_existing_db_works(tmp_path):
    db = tmp_path / "t.db"
    st = Store(db)
    st.insert_samples([make_sample(ts=1.0)])
    st.close()
    st2 = Store(db)  # reopen: schema IF NOT EXISTS path, version stays 2
    assert st2.count_samples() == 1
    assert st2._conn.execute("PRAGMA user_version").fetchone()[0] == Store.SCHEMA_VERSION
    st2.close()


def test_newer_schema_is_rejected(tmp_path):
    db = tmp_path / "t.db"
    st = Store(db)
    st._conn.execute("PRAGMA user_version=99")
    st.close()
    with pytest.raises(RuntimeError):
        Store(db)


def test_insert_samples_and_staleness(store):
    store.insert_samples([make_sample(ts=100.0), make_sample(ts=102.0,
                                                              cpu_total_pct=5.0)])
    store.insert_samples([])
    assert store.count_samples() == 2
    assert store.latest_sample_ts() == 102.0
    assert store.sample_staleness(now=104.0) == 2.0
    assert store.sample_staleness() is not None


def test_first_derived_rates_stay_none_in_rows(store):
    store.insert_samples([make_sample(ts=100.0)])  # unpopulated rates are None
    row = store._conn.execute("SELECT * FROM samples").fetchone()
    assert row["swp_out_ps"] is None
    assert row["psi_mem_some_stall_us"] is None


def test_prune_respects_retention_windows(store):
    now = 100.0 * 86400
    dead_sample = now - 8 * 86400      # older than raw_days=7
    kept_sample = now - 3 * 86400
    dead_event = now - 31 * 86400      # older than events_days=30
    kept_event = now - 3 * 86400
    store.insert_samples(
        [make_sample(ts=dead_sample), make_sample(ts=kept_sample)]
    )
    store.record_event(dead_event, "proc_birth", {"pid": 1})
    store.record_event(kept_event, "proc_birth", {"pid": 2})
    deleted = store.prune(now, raw_days=7, events_days=30)
    assert deleted["samples_deleted"] == 1
    assert deleted["events_deleted"] == 1
    assert store.count_samples() == 1
    rows = store._conn.execute("SELECT ts FROM events").fetchall()
    assert len(rows) == 1 and rows[0]["ts"] == kept_event


def test_rollup_math_and_unit_conversion(store):
    n = 3
    samples = []
    for i in range(n):
        samples.append(
            make_sample(
                ts=0.0 + i,
                cpu_total_pct=10.0 + i * 10,
                swap_used_pct=20.0,
                swp_out_ps=100.0,
                psi_mem=PsiInterval(some_total_us=0, some_stall_us=1_000_000,
                                    full_total_us=0, full_stall_us=0),
                psi_io=PsiInterval(some_total_us=0, some_stall_us=500_000),
            )
        )
    store.insert_samples(samples)
    agg = store.compute_rollup(0.0, 900.0)
    assert agg["sample_count"] == 3
    assert agg["cpu_total_pct_avg"] == 20.0
    assert agg["cpu_total_pct_max"] == 30.0
    assert agg["swap_used_pct_avg"] == 20.0
    assert agg["swp_out_ps_max"] == 100.0
    # microseconds -> seconds, so stall math stays auditable
    assert agg["psi_mem_stall_s_sum"] == 3.0
    assert agg["psi_io_stall_s_sum"] == 1.5


def test_rollup_empty_window_is_none(store):
    assert store.compute_rollup(0.0, 900.0) is None


def test_rollup_insert_is_idempotent(store):
    agg = {"window_start": 0.0, "window_end": 900.0, "sample_count": 1,
           "cpu_total_pct_avg": 5.0, "cpu_total_pct_max": 5.0,
           "mem_used_pct_avg": 50.0, "mem_available_bytes_avg": 1.0,
           "swap_used_pct_avg": 1.0, "swap_used_pct_max": 1.0,
           "swp_out_ps_avg": 0.0, "swp_out_ps_max": 0.0,
           "psi_mem_stall_s_sum": 0.0, "psi_io_stall_s_sum": 0.0}
    store.insert_rollup(agg)
    store.insert_rollup(agg)   # re-processed window must not duplicate
    n = store._conn.execute("SELECT COUNT(*) AS n FROM rollups").fetchone()["n"]
    assert n == 1


def test_rollup_marker_roundtrip(store):
    assert store.rollup_until() is None
    store.set_rollup_until(123.5)
    assert store.rollup_until() == 123.5


def test_heartbeat_upsert_and_age(store):
    assert store.heartbeat_age(now=1000.0) == float("inf")
    store.heartbeat(1000.0, pid=42, payload={"collect_ms": 3.0})
    store.heartbeat(1100.0, pid=42, payload={"collect_ms": 4.0})
    row = store._conn.execute("SELECT ts, payload FROM heartbeat").fetchone()
    assert row["ts"] == 1100.0
    assert '"collect_ms": 4.0' in row["payload"]
    assert store.heartbeat_age(now=1120.0) == 20.0


# ------------------------------------------------------------------ intents

def test_intent_insert_is_idempotent(store):
    assert store.insert_intent("i1", "mode_switch", {"mode": "advise"}, 1.0)
    assert not store.insert_intent("i1", "mode_switch", {"mode": "advise"}, 2.0)


def test_intent_claim_is_atomic_single_owner(store):
    store.insert_intent("i1", "manual_action", {}, 1.0)
    a = store.claim_next_pending(None, "worker-a", now=5.0)
    assert a is not None and a["status"] == "claimed" and a["tries"] == 1
    # a competing worker must get nothing, never a second owner
    assert store.claim_next_pending(None, "worker-b", now=6.0) is None
    # kind filter: no pending intents of that kind -> None
    assert store.claim_next_pending("mode_switch", "worker-c", now=6.0) is None


def test_intent_kind_filter_claims_only_matching(store):
    store.insert_intent("a1", "mode_switch", {}, 1.0)
    store.insert_intent("b1", "manual_action", {}, 2.0)
    row = store.claim_next_pending("manual_action", "w", 3.0)
    assert row["id"] == "b1"


def test_intent_finish_validates_transitions(store):
    store.insert_intent("i1", "manual_action", {}, 1.0)
    assert not store.finish_intent("i1", "applied", "no claim yet?", 2.0)
    store.claim_next_pending(None, "w", 2.0)
    assert store.finish_intent("i1", "applied", {"ok": True}, 3.0)
    # second finish must fail: applied is terminal (no double-apply)
    assert not store.finish_intent("i1", "applied", {"ok": True}, 4.0)
    with pytest.raises(ValueError):
        store.finish_intent("i1", "pending", "invalid final", 5.0)


def test_intent_lease_reconcile_and_expiry(store):
    store.insert_intent("i1", "manual_action", {}, 1.0)
    backdate = "UPDATE intents SET claimed_ts=? WHERE id='i1'"

    # claim #1 (tries=1), claimant dies -> back to pending
    row = store.claim_next_pending(None, "ghost-worker", now=100.0)
    assert row["tries"] == 1
    store._conn.execute(backdate, (100.0 - 600.0,))
    assert store.reconcile_intents(now=101.0, lease_s=120.0, max_tries=3) == [
        ("i1", "back_to_pending")
    ]

    # claim #2 (tries=2), claimant dies again -> back to pending
    row = store.claim_next_pending(None, "ghost-worker", now=102.0)
    assert row["tries"] == 2
    store._conn.execute(backdate, (103.0 - 600.0,))
    assert store.reconcile_intents(now=103.0, lease_s=120.0, max_tries=3) == [
        ("i1", "back_to_pending")
    ]

    # claim #3 (tries=3): beyond max_tries -> poison message expires, no loop
    row = store.claim_next_pending(None, "ghost-worker", now=104.0)
    assert row["tries"] == 3
    store._conn.execute(backdate, (105.0 - 600.0,))
    assert store.reconcile_intents(now=105.0, lease_s=120.0, max_tries=3) == [
        ("i1", "expired")
    ]
    status = store._conn.execute(
        "SELECT status FROM intents WHERE id='i1'"
    ).fetchone()["status"]
    assert status == "expired"


def test_healthy_claimed_intent_is_not_reconciled(store):
    store.insert_intent("i1", "manual_action", {}, 1.0)
    store.claim_next_pending(None, "alive-worker", now=100.0)
    assert store.reconcile_intents(now=101.0, lease_s=120.0) == []
