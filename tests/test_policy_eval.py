"""Policy-level evaluation tests: frozen mechanical labels, hysteresis
value, zero-positives honesty. All scenarios are fully synthetic and
deterministic; the REAL RiskTracker is replayed, not re-implemented."""

import pytest

from osrl.policy_eval import (
    LabelConfig,
    NaiveThresholdPolicy,
    RiskPolicy,
    evaluate_alarm_policies,
    event_segments,
    sample_from_row,
)


def make_rows(spec):
    """spec: list of (ts, interval_s, stall_us, swp_out_ps)."""
    return [
        {
            "ts": ts,
            "interval_s": interval,
            "psi_mem_some_stall_us": stall,
            "swp_out_ps": swp,
            "allocstall_ps": None,
            "mem_available_bytes": 8_000_000_000,
        }
        for ts, interval, stall, swp in spec
    ]


def scenario_rows():
    """2 s cadence over [0, 120]:
    - t=30: a SINGLE super-crit sample (0.55 >= 0.50) -> sustained-run rule
      keeps it a non-event; naive threshold alarms (FP), tracker must not.
    - t in [60, 100]: sustained 0.80 crit psi ratio -> one label event.
    """
    spec = []
    for i in range(61):
        t = i * 2.0
        if t == 30.0:
            stall = 1_100_000          # 0.55 ratio
        elif 60.0 <= t <= 100.0:
            stall = 1_600_000          # 0.80 ratio
        else:
            stall = 0
        spec.append((t, 2.0, stall, 0.0))
    return make_rows(spec)


def test_sample_from_row_preserves_none_and_values():
    r = make_rows([(7.0, 2.0, 1_600_000, 42.0)])[0]
    s = sample_from_row(r)
    assert s.ts == 7.0 and s.interval_s == 2.0
    assert s.psi_mem.some_stall_us == 1_600_000
    assert s.swp_out_ps == 42.0
    assert s.mem_available_bytes == 8_000_000_000
    r2 = make_rows([(7.0, None, None, None)])[0]
    s2 = sample_from_row(r2)
    assert s2.interval_s is None
    assert s2.psi_mem.some_stall_us is None


def test_event_segments_frozen_mechanical_rules():
    rows = scenario_rows()
    segs = event_segments(rows, LabelConfig())
    # t=30 spike: 1 qualifying sample < sustain_n -> NOT an event
    # t=60..100: 21 qualifying samples -> one event
    assert len(segs) == 1
    assert segs[0].start_ts == 60.0
    assert segs[0].end_ts == 100.0


def test_event_segments_do_not_bridge_outages():
    # runs separated by an outage gap never merge into one event
    spec = (
        [(8.0 + i * 2.0, 2.0, 1_600_000, 0.0) for i in range(3)]   # len 3 ok
        + [(500.0 + i * 2.0, 2.0, 1_600_000, 0.0) for i in range(3)]
    )
    segs = event_segments(make_rows(spec), LabelConfig())
    assert len(segs) == 2


def test_hysteresis_beats_naive_on_false_alarms():
    rows = scenario_rows()
    out = evaluate_alarm_policies(
        rows, [RiskPolicy(), NaiveThresholdPolicy()], LabelConfig()
    )
    risk = out["risk_tracker"]
    naive = out["naive_threshold"]

    assert risk["n_events"] == 1
    assert risk["recall"] == 1.0
    assert risk["n_false_alarm_runs"] == 0
    assert risk["precision"] == 1.0
    # enter_n=5 strong ticks at 2 s cadence: critical at t=78 -> 18 s delay
    assert risk["mean_detection_delay_s"] == pytest.approx(18.0)

    assert naive["recall"] == 1.0
    assert naive["n_false_alarm_runs"] == 1   # the t=30 single-sample spike
    assert naive["precision"] == pytest.approx(0.5)
    assert naive["mean_detection_delay_s"] == pytest.approx(0.0)


def test_zero_events_recalls_unmeasurable_not_zero():
    """Honesty rule: an empty-event bank must report recall=None (cannot
    measure), never 0/0 coerced, and never a fabricated 'perfect' score."""
    spec = [(i * 2.0, 2.0, 0, 0.0) for i in range(30)]
    rows = make_rows(spec)
    out = evaluate_alarm_policies(
        rows, [RiskPolicy(), NaiveThresholdPolicy()], LabelConfig()
    )
    for policy in ("risk_tracker", "naive_threshold"):
        assert out[policy]["n_events"] == 0
        assert out[policy]["recall"] is None
        assert out[policy]["precision"] is None
        assert out[policy]["n_false_alarm_runs"] == 0
        assert out[policy]["false_alarms_per_hour"] == 0.0


def test_prealarm_only_runs_are_possible_and_counted():
    """ELEVATED without CRITICAL requires the bad streak to end below the
    critical bar (tracker reaches critical at bad >= 10 with readable
    PSI, regardless of warn-vs-strong), then calm ticks demote. 7 bad
    ticks -> elevated at the 5th (t=12), calm resumes t=18, demote at
    the 15th good tick (t=46). Label events: zero (warn-level never
    reaches the frozen crit thresholds)."""
    spec = [
        (i * 2.0, 2.0,
         600_000 if 2 <= i <= 8 else 0,     # 0.30 ratio warn, 7 bad ticks
         600.0 if 2 <= i <= 8 else 0.0)     # swap warn -> 2 warn votes/bad
        for i in range(40)
    ]
    rows = make_rows(spec)
    out = evaluate_alarm_policies(rows, [RiskPolicy()], LabelConfig())
    risk = out["risk_tracker"]
    assert risk["n_events"] == 0
    assert risk["recall"] is None
    assert risk["n_alarm_runs"] == 0
    assert risk["n_prealarm_only_runs"] == 1
    assert risk["precision"] is None


def test_labels_reuse_risk_config_not_duplicated_thresholds():
    """The labeler is parameterized so thresholds track the deployed
    config: changing psi_crit in LabelConfig moves the event boundary."""
    spec = [(i * 2.0, 2.0, 700_000, 0.0) for i in range(10)]  # 0.35 ratio
    rows = make_rows(spec)
    strict = LabelConfig(psi_crit_ratio=0.5)
    loose = LabelConfig(psi_crit_ratio=0.3)
    assert event_segments(rows, strict) == []
    assert len(event_segments(rows, loose)) == 1


def test_prealarm_runs_cannot_make_precision_negative():
    from osrl.policy_eval import AlarmDecision, AlarmPolicy, ALARM, PREALARM, QUIET
    class Scripted(AlarmPolicy):
        name = "scripted"
        def reset(self):
            self.index = 0
        def observe(self, sample, ts):
            decisions = [PREALARM, QUIET, PREALARM, QUIET, ALARM, QUIET]
            value = decisions[self.index]
            self.index += 1
            return AlarmDecision(value)
    rows = make_rows([(i * 2, 2, 0, 0) for i in range(6)])
    result = evaluate_alarm_policies(rows, [Scripted()])["scripted"]
    assert result["precision"] == 0
    assert result["n_false_alarm_runs"] == 1
    assert result["n_prealarm_only_runs"] == 2


def test_policy_streaks_reset_across_outages():
    # Five bad ticks before and after an outage cannot form a ten-tick critical streak.
    rows = make_rows([(i * 2, 2, 1600000, 0) for i in range(5)] +
                     [(1000 + i * 2, 2, 1600000, 0) for i in range(5)])
    result = evaluate_alarm_policies(rows, [RiskPolicy()])["risk_tracker"]
    assert result["n_alarm_runs"] == 0
    assert result["n_prealarm_only_runs"] == 2
