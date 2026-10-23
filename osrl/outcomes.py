"""Measure observed telemetry around guarded intervention attempts.

These are before/after observations, not causal estimates: workload changes,
other processes, and unrelated events can change the same metrics.
"""

from __future__ import annotations

import json
import math

from osrl.config import RiskConfig
from osrl.policy_eval import sample_from_row
from osrl.risk import RiskTracker

DEFAULT_PRE_WINDOW_S = 30.0
DEFAULT_POST_WINDOW_S = 120.0
DEFAULT_WARMUP_S = 30.0
MIN_WINDOW_SAMPLES = 5


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metric(rows: list[dict], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value is not None and math.isfinite(float(value)):
            values.append(float(value))
    return _mean(values)


def _psi_mean(rows: list[dict]) -> float | None:
    values = []
    for row in rows:
        interval = row.get("interval_s")
        stall = row.get("psi_mem_some_stall_us")
        if interval is None or stall is None:
            continue
        interval, stall = float(interval), float(stall)
        if interval > 0 and math.isfinite(interval) and math.isfinite(stall) and stall >= 0:
            values.append(min(stall / (interval * 1_000_000), 1.0))
    return _mean(values)


def _target_cpu_mean(rows: list[dict], pid: int | None) -> float | None:
    if pid is None:
        return None
    values = []
    for row in rows:
        try:
            processes = json.loads(row.get("top_procs_json") or "[]")
        except (TypeError, ValueError):
            continue
        for process in processes:
            if process.get("pid") == pid:
                cpu = process.get("cpu_pct")
                if cpu is not None and math.isfinite(float(cpu)):
                    values.append(float(cpu))
                break
    return _mean(values)


def _covers_window(rows: list[dict], start: float, end: float, max_gap_s: float) -> float:
    if not rows or end <= start:
        return 0.0
    times = [float(row["ts"]) for row in rows]
    if times[0] > start + max_gap_s or times[-1] < end - max_gap_s:
        return 0.0
    if any(b - a > max_gap_s for a, b in zip(times, times[1:], strict=False)):
        return 0.0
    return min(1.0, max(0.0, (times[-1] - times[0]) / (end - start)))


def measure_action_outcome(
    action: dict,
    samples: list[dict],
    *,
    pre_window_s: float = DEFAULT_PRE_WINDOW_S,
    post_window_s: float = DEFAULT_POST_WINDOW_S,
    warmup_s: float = DEFAULT_WARMUP_S,
    min_window_samples: int = MIN_WINDOW_SAMPLES,
    risk_config: RiskConfig | None = None,
) -> dict:
    """Summarize telemetry around one applied action without causal claims."""
    applied_ts = action.get("ts_applied")
    if applied_ts is None or not math.isfinite(float(applied_ts)):
        return {"status": "insufficient_telemetry", "reason": "missing_apply_timestamp"}
    applied_ts = float(applied_ts)
    cfg = risk_config or RiskConfig()
    ordered = sorted(samples, key=lambda row: float(row["ts"]))
    pre_start = applied_ts - pre_window_s
    post_end = applied_ts + post_window_s
    pre = [r for r in ordered if pre_start <= float(r["ts"]) < applied_ts]
    post = [r for r in ordered if applied_ts < float(r["ts"]) <= post_end]
    pre_coverage = _covers_window(pre, pre_start, applied_ts, cfg.staleness_s)
    post_coverage = _covers_window(post, applied_ts, post_end, cfg.staleness_s)
    if (len(pre) < min_window_samples or len(post) < min_window_samples
            or pre_coverage < 0.8 or post_coverage < 0.8):
        return {
            "status": "insufficient_telemetry",
            "reason": "incomplete_pre_or_post_window",
            "pre_samples": len(pre),
            "post_samples": len(post),
            "pre_coverage": pre_coverage,
            "post_coverage": post_coverage,
            "pressure_before": "unknown",
            "pressure_after": "unknown",
            "pressure_recovered": None,
            "recovered_seconds": None,
            "pressure_returned": None,
            "psi_mem_before_avg": None,
            "psi_mem_after_avg": None,
            "psi_mem_change": None,
            "target_cpu_before_avg_pct": None,
            "target_cpu_after_avg_pct": None,
            "target_cpu_pct_delta": None,
            "foreground_impact": None,
            "foreground_impact_note": "Active foreground application identity is not collected.",
            "causal_claim": False,
        }

    tracker = RiskTracker(cfg)
    timeline = [r for r in ordered if applied_ts - pre_window_s - warmup_s <= float(r["ts"])
                <= applied_ts + post_window_s]
    assessments = []
    previous_ts = None
    for row in timeline:
        ts = float(row["ts"])
        if previous_ts is not None and ts - previous_ts > cfg.staleness_s:
            tracker.feed(None, ts)
        assessment = tracker.feed(sample_from_row(row), ts)
        assessments.append((ts, assessment.level))
        previous_ts = ts

    pre_levels = [level for ts, level in assessments if ts < applied_ts]
    post_levels = [level for ts, level in assessments if ts > applied_ts]
    pressure_levels = {"elevated", "critical"}
    was_pressured = bool(pre_levels and pre_levels[-1] in pressure_levels)
    pressure_after = next(
        (level for level in ("critical", "elevated") if level in post_levels), "none"
    )
    recovery_ts = None
    if was_pressured:
        recovery_ts = next(
            (ts for ts, level in assessments if ts > applied_ts and level == "none"), None
        )
    returned = (
        any(ts > recovery_ts and level in pressure_levels for ts, level in assessments)
        if recovery_ts is not None
        else None
    )

    before_psi, after_psi = _psi_mean(pre), _psi_mean(post)
    before_cpu = _target_cpu_mean(pre, action.get("target_pid"))
    after_cpu = _target_cpu_mean(post, action.get("target_pid"))
    target_cpu_delta = (
        after_cpu - before_cpu
        if before_cpu is not None and after_cpu is not None
        else None
    )
    return {
        "status": "measured",
        "pre_samples": len(pre),
        "post_samples": len(post),
        "pre_coverage": pre_coverage,
        "post_coverage": post_coverage,
        "pressure_before": pre_levels[-1] if pre_levels else "unknown",
        "pressure_after": pressure_after,
        "pressure_recovered": recovery_ts is not None if was_pressured else None,
        "recovered_seconds": (
            recovery_ts - applied_ts if recovery_ts is not None else None
        ),
        "pressure_returned": returned,
        "psi_mem_before_avg": before_psi,
        "psi_mem_after_avg": after_psi,
        "psi_mem_change": (
            after_psi - before_psi
            if before_psi is not None and after_psi is not None
            else None
        ),
        "cpu_before_avg_pct": _metric(pre, "cpu_total_pct"),
        "cpu_after_avg_pct": _metric(post, "cpu_total_pct"),
        "mem_available_before_avg_bytes": _metric(pre, "mem_available_bytes"),
        "mem_available_after_avg_bytes": _metric(post, "mem_available_bytes"),
        "target_cpu_before_avg_pct": before_cpu,
        "target_cpu_after_avg_pct": after_cpu,
        "target_cpu_pct_delta": target_cpu_delta,
        "foreground_impact": None,
        "foreground_impact_note": "Active foreground application identity is not collected.",
        "causal_claim": False,
    }
