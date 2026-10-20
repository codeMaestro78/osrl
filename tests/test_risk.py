"""Deterministic risk tracker: escalation, hysteresis, PSI gating, staleness."""

import pytest

from osrl.collector import PsiInterval
from osrl.config import RiskConfig
from osrl.risk import RiskTracker
from tests.conftest import make_sample

CFG = RiskConfig()  # enter_n=5, exit_n=15, staleness 6 s


def stall(ratio: float, interval: float = 2.0) -> PsiInterval:
    """Memory-PSI interval stall encoding a given ratio (stall_us/interval)."""
    return PsiInterval(some_total_us=1, some_stall_us=int(ratio * interval * 1e6))


def feed_seq(tracker, samples, t_start=1000.0, step=2.0):
    out = []
    for i, s in enumerate(samples):
        out.append(tracker.feed(s, now=t_start + i * step))
    return out


def test_healthy_samples_stay_none():
    tr = RiskTracker(CFG)
    seq = [make_sample(ts=1000.0 + i * 2, interval_s=2.0,
                       psi_mem=stall(0.0), swp_out_ps=0.0,
                       mem_available_bytes=4 * 2**30)
           for i in range(8)]
    states = feed_seq(tr, seq)
    assert all(a.level == "none" and not a.changed for a in states)


def test_escalation_requires_persistence():
    tr = RiskTracker(CFG)
    bad = dict(psi_mem=stall(0.3), swp_out_ps=800.0)  # two warn votes
    seq = [make_sample(ts=1000.0 + i * 2, interval_s=2.0, **bad)
           for i in range(6)]
    states = feed_seq(tr, seq)
    assert [a.level for a in states] == [
        "none", "none", "none", "none", "elevated", "elevated"
    ]
    assert states[4].changed  #Exactly at the enter_n-th bad sample


def test_two_warning_votes_necessary():
    """A single warn signature is never enough (multi-condition rule)."""
    for kwargs in (
        {"psi_mem": stall(0.3)},
        {"swp_out_ps": 800.0},
        {"allocstall_ps": 10.0},
        {"mem_available_bytes": 700 * 2**20},
    ):
        tr = RiskTracker(CFG)
        seq = [make_sample(ts=1000.0 + i * 2, interval_s=2.0, **kwargs)
               for i in range(20)]
        states = feed_seq(tr, seq)
        assert all(a.level == "none" for a in states), kwargs


def test_missing_metrics_cast_no_votes():
    tr = RiskTracker(CFG)
    # every pressure field None -> no evidence, no escalation, no crash
    seq = [make_sample(ts=1000.0 + i * 2) for i in range(20)]
    states = feed_seq(tr, seq)
    assert all(a.level == "none" for a in states)
    assert all(a.warn == [] for a in states)


def test_critical_requires_psi_evidence():
    tr = RiskTracker(CFG)
    secondary = dict(swp_out_ps=3000.0, allocstall_ps=50.0)  # strong evidence
    seq = [make_sample(ts=1000.0 + i * 2, interval_s=2.0, psi_mem=None, **secondary)
           for i in range(12)]
    states = feed_seq(tr, seq)
    assert all(a.level in ("none", "elevated") for a in states)
    assert states[-1].level == "elevated"  # capped, not critical: no PSI


def test_critical_with_psi_evidence():
    tr = RiskTracker(CFG)
    bad = dict(psi_mem=stall(0.6), swp_out_ps=3000.0)  # strong + psi present
    seq = [make_sample(ts=1000.0 + i * 2, interval_s=2.0, **bad)
           for i in range(12)]
    states = feed_seq(tr, seq)
    assert states[-1].level == "critical"
    assert states[9].changed  # enter_n*2 = 10th consecutive bad sample


def test_recovery_hysteresis_stepwise():
    tr = RiskTracker(CFG)
    bad = dict(psi_mem=stall(0.6), swp_out_ps=3000.0)
    calm = dict(psi_mem=stall(0.0), swp_out_ps=0.0,
                mem_available_bytes=8 * 2**30)
    states = feed_seq(tr, [make_sample(ts=1000.0 + i * 2, interval_s=2.0, **bad)
                           for i in range(10)])
    assert states[-1].level == "critical"

    # one good sample must NOT demote; need exit_n
    states += feed_seq(tr, [make_sample(ts=1200.0 + i * 2, interval_s=2.0, **calm)
                            for i in range(CFG.exit_n - 1)], t_start=1200.0)
    assert states[-1].level == "critical"

    # exit_n good samples demote critical -> elevated (step 1)
    states += feed_seq(tr, [make_sample(ts=2000.0 + i * 2, interval_s=2.0, **calm)
                            for i in range(CFG.exit_n)], t_start=2000.0)
    assert states[-1].level == "elevated"

    # another exit_n demotes elevated -> none (step 2)
    states += feed_seq(tr, [make_sample(ts=3000.0 + i * 2, interval_s=2.0, **calm)
                            for i in range(CFG.exit_n)], t_start=3000.0)
    assert states[-1].level == "none"


def test_renewed_badness_resets_recovery_progress():
    tr = RiskTracker(CFG)
    bad = dict(psi_mem=stall(0.3), swp_out_ps=800.0)
    calm = dict(psi_mem=stall(0.0), swp_out_ps=0.0)
    seq = (
        [make_sample(ts=1000.0 + i * 2, interval_s=2.0, **bad) for i in range(4)]
        + [make_sample(ts=1010.0, interval_s=2.0, **calm)]           # single good
        + [make_sample(ts=1020.0 + i * 2, interval_s=2.0, **bad)
           for i in range(4)]
    )
    states = feed_seq(tr, seq)
    assert states[-1].level != "elevated"  # streak was reset by the calm tick


def test_staleness_gate_forces_unknown():
    tr = RiskTracker(CFG)
    # fresh and pressing: streak builds
    a = tr.feed(make_sample(ts=1000.0, interval_s=2.0,
                            psi_mem=stall(0.3), swp_out_ps=800.0), now=1001.0)
    assert a.level == "none"
    # a LATE reading (> staleness_s after its ts) forces unknown
    late = tr.feed(make_sample(ts=1000.0, interval_s=2.0,
                               psi_mem=stall(0.3), swp_out_ps=800.0), now=1100.0)
    assert late.level == "unknown" and late.stale
    # unknown is not conflated with elevated; fresh data resumes cleanly
    fresh = tr.feed(make_sample(ts=1102.0, interval_s=2.0), now=1103.0)
    assert fresh.level in ("none", "unknown")
    seq = feed_seq(tr, [make_sample(ts=1200.0 + i * 2, interval_s=2.0,
                                    psi_mem=stall(0.3), swp_out_ps=800.0)
                        for i in range(5)], t_start=1200.0)
    assert seq[-1].level == "elevated"


def test_future_timestamp_treated_as_stale():
    tr = RiskTracker(CFG)
    a = tr.feed(make_sample(ts=5000.0, interval_s=2.0, psi_mem=stall(0.3)), now=0.0)
    assert a.level == "unknown" and a.stale  # clock skew must not escalate


def test_psi_trigger_fire_adds_bounded_vote():
    cfg = RiskConfig(enter_n=2)  # shorter persistence for a compact scenario
    tr = RiskTracker(cfg)
    tr.note_trigger("memory", 1020.0)  # kernel fired: recent stall window
    states = feed_seq(
        tr,
        [make_sample(ts=1020.0 + i * 2, interval_s=2.0, psi_mem=stall(0.3))
         for i in range(3)],
        t_start=1020.0,
    )
    # within the freshness budget each tick carries the trigger vote...
    assert all("psi_memory_trigger" in s.warn for s in states)
    # ...making psi-warn + trigger two warn votes -> persistent bad -> raise
    assert states[-1].level == "elevated"

    # the identical dataset without a trigger fire provides zero extra votes
    tr2 = RiskTracker(cfg)
    states2 = feed_seq(
        tr2,
        [make_sample(ts=1020.0 + i * 2, interval_s=2.0, psi_mem=stall(0.3))
         for i in range(3)],
        t_start=1020.0,
    )
    assert all(a.level == "none" for a in states2)


def test_stale_gate_released_by_fresh_clean_sample():
    tr = RiskTracker(CFG)
    far = tr.feed(make_sample(ts=1000.0, interval_s=2.0), now=1500.0)
    assert far.level == "unknown"
    fresh = tr.feed(make_sample(ts=1502.0, interval_s=2.0), now=1503.0)
    assert fresh.level == "none"  # wait-state released, never stuck unknown


def test_lock_is_thread_safe_smoke():
    tr = RiskTracker(CFG)
    import threading as _t

    def note():
        for _ in range(200):
            tr.note_trigger("memory", 1000.0)

    threads = [_t.Thread(target=note) for _ in range(4)]
    for th in threads:
        th.start()
    tr.feed(make_sample(ts=1000.0, interval_s=2.0), now=1001.0)
    for th in threads:
        th.join()


@pytest.mark.parametrize("ts_gap,expect_unknown", [(7.0, True), (5.0, False)])
def test_staleness_boundary(ts_gap, expect_unknown):
    tr = RiskTracker(CFG)
    a = tr.feed(make_sample(ts=1000.0, interval_s=2.0), now=1000.0 + ts_gap)
    assert (a.level == "unknown") is expect_unknown
