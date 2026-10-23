"""Worker end-to-end on REAL processes we own: dry-run vs --execute gate,
apply, rate-limit rejection, restore elevation rejection, malformed payload,
limit release. State asserted through store + psutil."""

import json
import subprocess
import time
import uuid

import psutil
import pytest

import osrl.worker as worker
from osrl.config import GuardConfig
from osrl.executor import ExecutorError
from osrl.guard import Guard, GuardDeps
from osrl.store import Store
from osrl.worker import process_pending
from tests.conftest import make_sample


def make_guard(store: Store, cfg: GuardConfig | None = None) -> Guard:
    return Guard(
        cfg or GuardConfig(min_spacing_s=300.0),
        GuardDeps(),
        sample_staleness=store.sample_staleness,
        last_action_ts=store.last_action_ts,
        read_action=store.action_row,
    )


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "w.db")
    now = time.time()
    store.insert_samples([make_sample(ts=now, interval_s=2.0)])

    child = subprocess.Popen(["sleep", "25"])
    st = psutil.Process(child.pid).create_time()
    yield store, child, now, st
    child.kill()
    child.wait()
    store.close()

def put_intent(store: Store, kind: str, payload: dict) -> str:
    iid = str(uuid.uuid4())
    assert store.insert_intent(iid, kind, payload, time.time())
    return iid


def test_dry_run_never_claims_or_mutates(env):
    store, child, now, st = env
    before = psutil.Process(child.pid).nice()
    iid = put_intent(store, "renice_back", {"pid": child.pid,
                                            "start_time": st})
    out = process_pending(store, make_guard(store), GuardConfig(),
                          now=time.time(), execute=False)
    assert out[0]["verdict"] == "allow"
    row = store._conn.execute(
        "SELECT status FROM intents WHERE id=?", (iid,)
    ).fetchone()
    assert row["status"] == "pending"     # dry run claims nothing
    assert psutil.Process(child.pid).nice() == before
    assert store.last_action_ts(child.pid, "renice_back") is None


def test_execute_applies_renice_back_and_records_everything(env):
    store, child, now, st = env
    iid = put_intent(store, "renice_back", {"pid": child.pid,
                                            "start_time": st})
    before = psutil.Process(child.pid).nice()
    out = process_pending(store, make_guard(store), GuardConfig(),
                          now=time.time(), worker_id="w-test",
                          execute=True)
    assert out[0]["outcome"] == "applied"
    assert psutil.Process(child.pid).nice() == before + 10

    intent = store._conn.execute(
        "SELECT status, result FROM intents WHERE id=?", (iid,)
    ).fetchone()
    assert intent["status"] == "applied"
    result = json.loads(intent["result"])
    actions = store.actions_for_intent(iid)
    assert len(actions) == 1
    row = actions[0]
    assert row["status"] == "applied"
    assert row["applied_nice"] == before + 10
    # before_nice is the audit column the evidence pipeline needs to
    # compute nice deltas (bug found live 2026-10-20: it was always NULL)
    assert row["before_nice"] == before
    assert result["action_id"] == row["id"]
    guard_checks = json.loads(row["guard_checks"])
    assert all(entry["passed"] for entry in guard_checks)
    [outcome] = store.recent_action_outcomes()
    assert outcome["action_id"] == row["id"]
    assert outcome["status"] == "pending"

    kinds = [r["kind"] for r in store._conn.execute(
        "SELECT kind FROM events").fetchall()]
    assert "action_applied" in kinds


def test_executor_failure_is_recorded_as_not_applied(env, monkeypatch):
    store, child, _now, st = env
    before = psutil.Process(child.pid).nice()
    iid = put_intent(store, "renice_back", {"pid": child.pid, "start_time": st})

    def fail_before_mutation(*args, **kwargs):
        raise ExecutorError("simulated permission denial")

    monkeypatch.setattr(worker, "apply_renice_back", fail_before_mutation)
    result = worker.process_pending(
        store, make_guard(store), GuardConfig(), now=time.time(),
        worker_id="w-test", execute=True,
    )

    assert result[0]["outcome"] == "failed"
    assert psutil.Process(child.pid).nice() == before
    [outcome] = store.recent_action_outcomes()
    assert outcome["intent_id"] == iid
    assert outcome["status"] == "not_applied"
    assert outcome["details"]["attempt_status"] == "failed"
    assert "permission denial" in outcome["details"]["reason"]


def test_rerate_within_spacing_is_rejected_not_applied(env):
    store, child, now, st = env
    put_intent(store, "renice_back", {"pid": child.pid, "start_time": st})
    process_pending(store, make_guard(store), GuardConfig(),
                    now=time.time(), worker_id="w-test", execute=True)
    first_nice = psutil.Process(child.pid).nice()

    put_intent(store, "renice_back", {"pid": child.pid, "start_time": st})
    out = process_pending(store, make_guard(store), GuardConfig(),
                          now=time.time(), worker_id="w-test",
                          execute=True)
    assert out[0]["outcome"] == "rejected"
    assert "spacing" in out[0]["reason"]
    assert psutil.Process(child.pid).nice() == first_nice  # untouched


def test_restore_rejected_requires_elevation(env):
    store, child, now, st = env
    original = psutil.Process(child.pid).nice()
    # build a real parent row (simulate a prior applied renice_back)
    put_intent(store, "renice_back", {"pid": child.pid, "start_time": st})
    process_pending(store, make_guard(store), GuardConfig(),
                    now=time.time(), worker_id="w-test", execute=True)
    assert psutil.Process(child.pid).nice() == original + 10

    parent = store._conn.execute(
        "SELECT id FROM actions_ledger WHERE kind='renice_back'"
        " AND status='applied' ORDER BY created_ts DESC LIMIT 1"
    ).fetchone()

    iid = put_intent(store, "restore", {
        "pid": child.pid, "start_time": st,
        "parent_action_id": parent["id"], "target_nice": original,
    })
    out = process_pending(store, make_guard(store), GuardConfig(),
                          now=time.time(), worker_id="w-test",
                          execute=True)
    assert out[0]["outcome"] == "rejected"
    assert "elevation" in out[0]["reason"]
    assert store._conn.execute(
        "SELECT status FROM intents WHERE id=?", (iid,)
    ).fetchone()["status"] == "rejected"
    assert psutil.Process(child.pid).nice() == original + 10
    assert "action_rejected" in [r["kind"] for r in
                                 store._conn.execute(
                                     "SELECT kind FROM events").fetchall()]


def test_malformed_payload_rejected_with_reason(env):
    store, child, now, st = env
    iid = put_intent(store, "renice_back", {"nonsense": True})
    out = process_pending(store, make_guard(store), GuardConfig(),
                          now=time.time(), worker_id="w", execute=True)
    assert out[0]["outcome"] == "rejected"
    assert out[0]["reason"] == "malformed_payload"
    assert store._conn.execute(
        "SELECT status FROM intents WHERE id=?", (iid,)
    ).fetchone()["status"] == "rejected"
    [outcome] = store.recent_action_outcomes()
    assert outcome["intent_id"] == iid
    assert outcome["status"] == "not_applied"
    assert outcome["details"]["attempt_status"] == "rejected"


def test_limit_releases_instead_of_consuming(env):
    store, child, now, st = env
    put_intent(store, "renice_back", {"pid": child.pid, "start_time": st})
    i2 = put_intent(store, "renice_back", {"pid": child.pid, "start_time": st})
    out = process_pending(store, make_guard(store), GuardConfig(),
                          now=time.time(), worker_id="w-test",
                          execute=True, limit=1)
    assert out[0]["outcome"] == "applied"
    assert out[1]["outcome"] == "released"
    assert store._conn.execute(
        "SELECT status FROM intents WHERE id=?", (i2,)
    ).fetchone()["status"] == "pending"  # unconsumed, still actionable


def test_identity_mismatch_never_applies(env):
    store, child, now, st = env
    put_intent(store, "renice_back",
               {"pid": child.pid, "start_time": st + 9.9})
    out = process_pending(store, make_guard(store), GuardConfig(),
                          now=time.time(), worker_id="w-test",
                          execute=True)
    assert out[0]["outcome"] == "rejected"
    assert "drifted" in out[0]["reason"]


def test_stale_telemetry_abstains(env):
    store, child, now, st = env
    # discard the fresh sample: no telemetry -> hard deny (acceptance #2)
    store._conn.execute("DELETE FROM samples")
    put_intent(store, "renice_back", {"pid": child.pid, "start_time": st})
    out = process_pending(store, make_guard(store), GuardConfig(),
                          now=time.time(), worker_id="w-test",
                          execute=True)
    assert out[0]["outcome"] == "rejected"
    assert out[0]["reason"] == "no samples in store"
    before = psutil.Process(child.pid).nice()
    assert psutil.Process(child.pid).nice() == before
