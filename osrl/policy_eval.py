"""Policy-level evaluation: alarm decisions scored against ground-truth
events (Phase 4.0). The twin harness (osrl/twin.py) scores STATE
forecasters; this scores ALARM POLICIES — the decision layer a future RL
candidate must beat.

Integrity rules:

- Ground truth is a FROZEN mechanical labeler, not the policy being
  scored: an event = a maximal run of >= label_sustain_n consecutive
  samples (tolerating only outage-sized gaps) whose sample carries a
  critical signature — (psi_mem stall ratio >= psi_crit) OR
  (swp_out >= swp_crit), using the SAME thresholds the risk config
  declares. The label is therefore auditable data, not the tracker's
  opinion.
- Causality: a policy at ts sees only rows <= ts (streamed in ts order).
- Replay freshness: decisions are made at now == row.ts, i.e. perfectly
  fresh telemetry. That is the strongest case for every policy; the
  staleness gate therefore does not fire here (recorded honestly as
  0 abstentions rather than simulated).
- Zero-positives honesty: with no events in the data, recall is None
  (unmeasurable); with no alarms at all, precision is None. False
  alarms remain measurable: maximal consecutive non-quiet runs outside
  events, normalized per hour.
- The shipped comparison is RiskTracker (=today's deployed logic,
  unchanged: real Sample objects are reconstructed from rows) versus
  the NaiveThresholdPolicy strawman (single-sample critical threshold,
  no sustain/hysteresis) — measuring what enter_n discipline buys.
- No RL imports (structural scan); observe-only, never mutates.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from osrl.collector import PsiInterval, Sample
from osrl.config import RiskConfig
from osrl.risk import RiskTracker
from osrl.twin import median_interval

ALARM = "alarm"
PREALARM = "prealarm"
QUIET = "quiet"
ABSTAIN = "abstain"


def _field(row: sqlite3.Row | dict, key: str) -> float | None:
    try:
        v = row[key]
    except (KeyError, IndexError):
        return None
    return None if v is None else float(v)


def sample_from_row(row: sqlite3.Row | dict) -> Sample:
    """Reconstruct the collector-shaped Sample a row represents so the
    REAL risk tracker can be replayed on recorded telemetry (no parallel
    re-implementation of its semantics)."""
    stall = _field(row, "psi_mem_some_stall_us")
    return Sample(
        ts=_field(row, "ts") or 0.0,
        interval_s=_field(row, "interval_s"),
        psi_mem=PsiInterval(
            some_stall_us=None if stall is None else int(stall),
        ),
        swp_out_ps=_field(row, "swp_out_ps"),
        allocstall_ps=_field(row, "allocstall_ps"),
        mem_available_bytes=_field(row, "mem_available_bytes"),
    )


# ------------------------------------------------------------- ground truth

@dataclass(frozen=True)
class LabelConfig:
    label_sustain_n: int = 3          # consecutive qualifying samples
    psi_crit_ratio: float = 0.50      # mirrors RiskConfig defaults
    swp_out_crit_ps: float = 2048.0


@dataclass
class EventSegment:
    start_ts: float
    end_ts: float


def event_segments(
    rows: list[sqlite3.Row], label_cfg: LabelConfig, max_gap_s: float = 60.0
) -> list[EventSegment]:
    """Frozen mechanical events: maximal runs of sample-level critical
    signatures with length >= label_sustain_n; the ratio is computed
    exactly as the risk layer does (stall_us / (interval * 1e6)). A gap
    larger than max_gap_s (an outage) can never bridge samples of one
    run."""
    segs: list[EventSegment] = []

    def ratio(row) -> float | None:
        interval = _field(row, "interval_s")
        stall = _field(row, "psi_mem_some_stall_us")
        if interval is None or interval <= 0 or stall is None:
            return None
        return stall / (interval * 1e6)

    def qualifies(row) -> bool:
        r = ratio(row)
        if r is not None and r >= label_cfg.psi_crit_ratio:
            return True
        swp = _field(row, "swp_out_ps")
        return swp is not None and swp >= label_cfg.swp_out_crit_ps

    run: list[float] = []

    def flush() -> None:
        if len(run) >= label_cfg.label_sustain_n:
            segs.append(EventSegment(start_ts=run[0], end_ts=run[-1]))
        run.clear()

    prev_ts: float | None = None
    for row in rows:
        ts = _field(row, "ts")
        if ts is None:  # unlabelable row
            flush()
            continue
        if prev_ts is not None and ts - prev_ts > max_gap_s:
            flush()
        if qualifies(row):
            run.append(ts)
        else:
            flush()
        prev_ts = ts
    flush()
    return segs


# ----------------------------------------------------------------- policies

@dataclass
class AlarmDecision:
    decision: str  # ALARM | PREALARM | QUIET | ABSTAIN
    reason: str = ""


class AlarmPolicy:
    """Sees one reconstructed sample per tick (fed causally in ts order)."""

    name = "policy"

    def observe(self, sample: Sample, ts: float) -> AlarmDecision:
        raise NotImplementedError

    def reset(self) -> None:
        """Fresh state per session (no cross-outage memory)."""
        raise NotImplementedError


class RiskPolicy(AlarmPolicy):
    """The deployed risk tracker, replayed chronologically. Alarm when the
    tracker says critical; prealarm at elevated; abstain at unknown (only
    reachable if a caller ever feeds stale pairs — replay at now == ts
    cannot)."""

    name = "risk_tracker"

    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or RiskConfig()
        self._tracker = RiskTracker(self.cfg)

    def observe(self, sample: Sample, ts: float) -> AlarmDecision:
        a = self._tracker.feed(sample, now=ts)
        if a.level == "critical":
            return AlarmDecision(ALARM, "risk tracker critical")
        if a.level == "elevated":
            return AlarmDecision(PREALARM, "risk tracker elevated")
        if a.level == "unknown":
            return AlarmDecision(ABSTAIN, "stale telemetry")
        return AlarmDecision(QUIET, f"level={a.level}")

    def reset(self) -> None:
        self._tracker = RiskTracker(self.cfg)


class NaiveThresholdPolicy(AlarmPolicy):
    """Single-sample critical threshold with no sustain/hysteresis: the
    strawman whose false alarms the tracker's enter_n discipline must
    beat on real data."""

    name = "naive_threshold"

    def __init__(self, cfg: LabelConfig | None = None) -> None:
        self.cfg = cfg or LabelConfig()

    def observe(self, sample: Sample, ts: float) -> AlarmDecision:  # noqa: ARG002
        if (
            sample.psi_mem is not None
            and sample.psi_mem.some_stall_us is not None
            and sample.interval_s
            and sample.interval_s > 0
            and sample.psi_mem.some_stall_us / (sample.interval_s * 1e6)
            >= self.cfg.psi_crit_ratio
        ):
            return AlarmDecision(ALARM, "instant crit ratio")
        if sample.swp_out_ps is not None and (
            sample.swp_out_ps >= self.cfg.swp_out_crit_ps
        ):
            return AlarmDecision(ALARM, "instant swap rate")
        return AlarmDecision(QUIET)

    def reset(self) -> None:
        return


# ---------------------------------------------------------------- evaluation

@dataclass
class AlarmRun:
    """Maximal CONSECUTIVE non-quiet stretch (any gap breaks the run)."""

    start_ts: float
    end_ts: float
    decisions: list[str] = field(default_factory=list)

    @property
    def is_prealarm_only(self) -> bool:
        return ALARM not in self.decisions


def _alarm_runs(times: list[float], decisions: list[str], max_gap_s: float = 60.0) -> list[AlarmRun]:
    out: list[AlarmRun] = []
    cur: AlarmRun | None = None
    previous = None
    for t, d in zip(times, decisions, strict=True):
        if previous is not None and t - previous > max_gap_s:
            cur = None
        previous = t
        if d in (ALARM, PREALARM):
            if cur is None:
                cur = AlarmRun(t, t, [d])
                out.append(cur)
            else:
                cur.end_ts = t
                cur.decisions.append(d)
        else:
            cur = None
    return out


def evaluate_alarm_policies(
    rows: list[sqlite3.Row],
    policies: list[AlarmPolicy],
    label_cfg: LabelConfig | None = None,
) -> dict[str, dict]:
    """Score each policy on chronologically streamed rows. Per policy:
    events/detected/recall, precision, false-alarm runs (outside events)
    with per-hour normalization, mean detection delay from event start
    (first ALARM inside the event), prealarm-only runs, abstentions.
    None = unmeasurable on this data (zero events or zero alarm runs)."""
    label_cfg = label_cfg or LabelConfig()
    rows = sorted(rows, key=lambda r: float(r["ts"]))
    max_gap = max(3 * median_interval(rows), 10.0)
    segs = event_segments(rows, label_cfg, max_gap)
    times = [float(r["ts"]) for r in rows]

    def in_event(t: float) -> bool:
        return any(s.start_ts <= t <= s.end_ts for s in segs)

    out: dict[str, dict] = {}
    for policy in policies:
        policy.reset()
        decisions: list[str] = []
        previous = None
        for row in rows:
            if previous is not None and float(row["ts"]) - previous > max_gap:
                policy.reset()
            previous = float(row["ts"])
            s = sample_from_row(row)
            decisions.append(policy.observe(s, float(row["ts"])).decision)

        runs = _alarm_runs(times, decisions, max_gap)
        alarm_runs = [r for r in runs if not r.is_prealarm_only]
        n_events = len(segs)
        detected = 0
        delays: list[float] = []
        for seg in segs:
            inside = [
                t for t, d in zip(times, decisions, strict=True)
                if d == ALARM and seg.start_ts <= t <= seg.end_ts
            ]
            if inside:
                detected += 1
                delays.append(inside[0] - seg.start_ts)
        false_runs = [
            r for r in alarm_runs
            if not any(r.start_ts <= s.end_ts and r.end_ts >= s.start_ts for s in segs)
        ]
        span_h = sum(b - a for a, b in zip(times, times[1:])
                     if 0 < b - a <= max_gap) / 3600.0
        n_true_alarm_runs = len(alarm_runs) - len(false_runs)
        precision = (
            n_true_alarm_runs / len(alarm_runs) if alarm_runs else None
        )
        out[policy.name] = {
            "n_events": n_events,
            "n_detected": detected,
            "recall": (detected / n_events) if n_events else None,
            "precision": precision,
            "n_alarm_runs": len(alarm_runs),
            "n_false_alarm_runs": len(false_runs),
            "false_alarms_per_hour": (
                len(false_runs) / span_h if span_h > 0 else None
            ),
            "mean_detection_delay_s": (
                sum(delays) / len(delays) if delays else None
            ),
            "n_prealarm_only_runs": sum(
                1 for r in runs if r.is_prealarm_only
            ),
            "n_abstained_samples": sum(1 for d in decisions if d == ABSTAIN),
        }
    return out
