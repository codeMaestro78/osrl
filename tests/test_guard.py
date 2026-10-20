"""Guard: full first-failure-wins matrix, all synthetic (no real procs)."""

import json
import time

import pytest

from osrl.config import GuardConfig
from osrl.guard import ActionRequest, Guard, GuardDeps

EUID = 1000
START = 555.0


class Deps(GuardDeps):
    def __init__(self, *, uid=1000, st=555.0, nice=5, self_pid=999):
        super().__init__(
            self_pid=lambda: self_pid,
            self_euid=lambda: EUID,
            target_uid=lambda _pid: uid,
            start_time_of=lambda _pid: st,
            read_nice=lambda _pid: nice,
        )


def make_guard(deps: Deps, *, capable=False, stale=1.0, last=None,
               parent=None, cfg=None):
    return Guard(
        cfg or GuardConfig(restore_capable=capable),
        deps,
        sample_staleness=lambda now: stale,
        last_action_ts=lambda pid, kind: last,
        read_action=lambda pid_id: parent,
    )


def req(kind="renice_back", pid=777, **kw):
    return ActionRequest(
        kind=kind, pid=pid, start_time=START,
        parent_action_id=kw.get("parent_action_id"),
        target_nice=kw.get("target_nice"),
    )


# ------------------------------------------------------------------ allows

def test_allow_renice_back_happy_path():
    g = make_guard(Deps())
    v = g.check(req(), now=time.time())
    assert v.ok
    assert v.target_nice == 15  # 5 + step 10
    # ALL check names present when complete
    want = {"kind_allowed", "pid_sane", "identity", "ownership",
            "telemetry_fresh", "capability", "state", "rate_limit",
            "intent_lease"}
    assert {c.name for c in v.checks} == want
    # ledger-serializable
    json.dumps(v.as_dict())


def test_target_nice_clamped_at_max():
    g = make_guard(Deps(nice=17))
    assert g.check(req(), now=0.0).target_nice == 19


# ------------------------------------------------------------------ denies

@pytest.mark.parametrize("kind", ["", "kill", "sigstop", "renice_forward"])
def test_unknown_kind_denied(kind):
    g = make_guard(Deps())
    v = g.check(req(kind=kind), now=0.0)
    assert not v.ok and v.reason.startswith("unknown action kind")


def test_refuses_self_and_init():
    g = make_guard(Deps())
    assert not g.check(req(pid=999), now=0.0).ok  # self pid
    v = g.check(req(pid=1), now=0.0)
    assert not v.ok and v.reason.startswith("refusing pid")


def test_identity_missing_target():
    g = make_guard(Deps(st=None))
    v = g.check(req(), now=0.0)
    assert not v.ok and v.reason.endswith("not found")


def test_identity_drift():
    g = make_guard(Deps(st=999.0))
    v = g.check(req(), now=0.0)
    assert not v.ok and "drifted" in v.reason


def test_ownership_other_user():
    g = make_guard(Deps(uid=999))
    v = g.check(req(), now=0.0)
    assert not v.ok and v.reason.startswith("target euid")


def test_ownership_unreadable():
    g = make_guard(Deps(uid=None))
    v = g.check(req(), now=0.0)
    assert not v.ok and v.reason == "target uid unreadable"


def test_no_telemetry_denied():
    g = make_guard(Deps(), stale=None)
    v = g.check(req(), now=0.0)
    assert not v.ok and v.reason == "no samples in store"


def test_stale_telemetry_denied():
    g = make_guard(Deps(), stale=999.0)
    v = g.check(req(), now=0.0)
    assert not v.ok and "stale telemetry" in v.reason


def test_nice_unreadable_denied():
    g = make_guard(Deps(nice=None))
    v = g.check(req(), now=0.0)
    assert not v.ok and v.reason == "current nice unreadable"


def test_prioritized_target_refused():
    g = make_guard(Deps(nice=-5))
    v = g.check(req(), now=0.0)
    assert not v.ok and "prioritized" in v.reason


def test_already_at_ceiling_denied():
    g = make_guard(Deps(nice=19))
    v = g.check(req(), now=0.0)
    assert not v.ok and "exhausted" in v.reason


    g = make_guard(Deps(), last=149.0)
    assert not g.check(req(), now=150.0).ok


def test_rate_limited_denied():
    g = make_guard(Deps(), last=90.0)  # 90s ago < 300 spacing
    v = g.check(req(), now=150.0)
    assert not v.ok and "spacing" in v.reason


# ------------------------------------------------------------------ restore

def test_restore_requires_elevation_by_default():
    g = make_guard(Deps(nice=10), capable=False)
    v = g.check(req(kind="restore", target_nice=0,
                    parent_action_id="pa1"), now=0.0)
    assert not v.ok and "elevation" in v.reason


def test_restore_happy_with_parent_and_capability():
    parent = {
        "kind": "renice_back", "status": "applied",
        "target_pid": 777, "target_start_time": START,
        "applied_nice": 15,
    }
    g = make_guard(Deps(nice=15), capable=True, parent=parent)
    v = g.check(req(kind="restore", target_nice=5,
                    parent_action_id="pa1"), now=0.0)
    assert v.ok and v.target_nice == 5


@pytest.mark.parametrize("mutate", [
    {"kind": "restore"},                      # wrong parent kind
    {"status": "proposed"},                   # not applied yet
    {"target_pid": 555},                      # different target
    {"applied_nice": 14},                     # nice state drifted
])
def test_restore_parent_state_checks(mutate):
    base = {
        "kind": "renice_back", "status": "applied",
        "target_pid": 777, "target_start_time": START,
        "applied_nice": 15,
    }
    base.update(mutate)
    g = make_guard(Deps(nice=15), capable=True, parent=base)
    v = g.check(req(kind="restore", target_nice=5,
                    parent_action_id="pa1"), now=0.0)
    assert not v.ok


def test_restore_missing_parent_row():
    g = make_guard(Deps(nice=15), capable=True, parent=None)
    v = g.check(req(kind="restore", target_nice=5,
                    parent_action_id="nope"), now=0.0)
    assert not v.ok and "missing" in v.reason


def test_restore_must_lower_priority():
    parent = {"kind": "renice_back", "status": "applied",
              "target_pid": 777, "target_start_time": START,
              "applied_nice": 15}
    g = make_guard(Deps(nice=15), capable=True, parent=parent)
    v = g.check(req(kind="restore", target_nice=20,
                    parent_action_id="pa1"), now=0.0)
    assert not v.ok and "must lower" in v.reason


def test_restore_guard_short_circuits_before_elevation_check():
    g = make_guard(Deps(nice=10), capable=False)
    v = g.check(req(kind="restore", target_nice=5,
                    parent_action_id=None), now=0.0)
    # capability check comes BEFORE parent lookup: reason is elevation, not
    # missing parent (fail order is deliberate and recorded)
    assert "elevation" in v.reason


# ------------------------------------------------------------ intent lease

def test_intent_must_be_claimed_by_us():
    g = make_guard(Deps())
    intent = {"kind": "renice_back", "status": "pending", "claimed_by": None}
    v = g.check(req(), now=0.0, intent=intent, worker_id="w1")
    assert not v.ok and "intent status" in v.reason

    intent = {"kind": "renice_back", "status": "claimed",
              "claimed_by": "other-worker"}
    assert not g.check(req(), now=0.0, intent=intent,
                       worker_id="w1").ok

    intent = {"kind": "renice_back", "status": "claimed",
              "claimed_by": "w1"}
    assert g.check(req(), now=0.0, intent=intent, worker_id="w1").ok


def test_intent_kind_mismatch_denied():
    g = make_guard(Deps())
    intent = {"kind": "restore", "status": "claimed", "claimed_by": "w1"}
    v = g.check(req(), now=0.0, intent=intent, worker_id="w1")
    assert not v.ok and "intent kind" in v.reason
