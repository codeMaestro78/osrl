"""Digital-twin substrate + baseline registry (Phase 3, observe-only).

This is the layer any future RL policy must BEAT (brief F / DESIGN §11.8):
no-op, hold-state, and linear-trend baselines over held-out sessions with
horizon-wise error accounting. PPO arrives (Phase 4) only if it
demonstrably beats all of these on the SAME harness.

Data-integrity rules carried from §11:

- Sessions = contiguous sample runs; a gap > max(3x median observed
  interval, 10 s) starts a new session. Predictions never span daemon
  outages; every baseline restarts fresh per session.
- Missing metrics stay missing: features preserve None; error metrics
  report their own coverage (n_pairs / n_abstained) so a "great MAE over
  2 pairs" can never masquerade as solid evidence.
- Horizon targets must EXIST as recorded rows within their own session,
  matched nearest-first under a small timestamp-jitter tolerance (1% of
  the median interval — jitter, never a full sampling interval) — the
  target reading is contemporaneous, never interpolated.
- The renice forward model ABSTAINS until real ledger evidence reaches
  min pairs; on this deployment real pairs are ~zero, so its honest state
  is "abstains everywhere" until actions accumulate. Nothing invents data.
- Causal harness: at target ts, a baseline has observed ONLY rows with
  ts <= target_ts inside its session.
- No RL imports (structural scan in tests/test_safety_invariants.py).
"""

from __future__ import annotations

import bisect
import sqlite3
from dataclasses import dataclass, field
from typing import Any

FEATURE_KEYS = (
    "psi_mem_ratio", "swp_out_ps", "allocstall_ps",
    "mem_available_mib", "swap_used_pct", "cpu_total_pct",
)

EVAL_HORIZONS_S = (30.0, 60.0, 120.0)


# ------------------------------------------------------------------ sessions

@dataclass
class Session:
    """A contiguous stretch of stored samples (no outage-sized gaps)."""

    start_ts: float
    end_ts: float
    rows: list[sqlite3.Row] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts


def sample_features(row: sqlite3.Row | dict) -> dict:
    """Feature derivation with None-preservation (missing stays missing)."""
    def num(key: str) -> float | None:
        try:
            v = row[key]
        except (KeyError, IndexError):
            return None
        return None if v is None else float(v)

    interval = num("interval_s")
    stall = num("psi_mem_some_stall_us")
    ratio = None
    if interval is not None and interval > 0 and stall is not None:
        ratio = stall / (interval * 1e6)
    mem_avail = num("mem_available_bytes")
    return {
        "psi_mem_ratio": ratio,
        "swp_out_ps": num("swp_out_ps"),
        "allocstall_ps": num("allocstall_ps"),
        "mem_available_mib": (
            None if mem_avail is None else mem_avail / (1024.0 * 1024.0)
        ),
        "swap_used_pct": num("swap_used_pct"),
        "cpu_total_pct": num("cpu_total_pct"),
    }


def median_interval(rows: list[sqlite3.Row]) -> float:
    intervals = [
        float(r["interval_s"])
        for r in rows
        if r["interval_s"] is not None and float(r["interval_s"]) > 0
    ]
    return _median_of(intervals)


def sessions_from_rows(
    rows: list[sqlite3.Row],
    *,
    median_interval_s: float | None = None,
    max_gap_factor: float = 3.0,
    min_gap_s: float = 10.0,
) -> list[Session]:
    """Slice ascending rows on outage gaps."""
    if median_interval_s is None:
        median_interval_s = median_interval(rows)
    max_gap = max(max_gap_factor * median_interval_s, min_gap_s)

    out: list[Session] = []
    cur: Session | None = None
    prev_ts: float | None = None
    for row in rows:
        ts = float(row["ts"])
        if cur is None or (prev_ts is not None and ts - prev_ts > max_gap):
            cur = Session(start_ts=ts, end_ts=ts)
            out.append(cur)
        cur.rows.append(row)
        cur.end_ts = ts
        prev_ts = ts
    return [s for s in out if len(s.rows) >= 2]


def _median_of(values: list[float]) -> float:
    if not values:
        return 30.0  # documented fallback (hold-test cadence)
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def sessions_from_store(
    conn: sqlite3.Connection | Any, *, limit: int = 200_000, **kwargs: Any
) -> list[Session]:
    rows = conn.execute(
        "SELECT * FROM samples ORDER BY ts LIMIT ?", (int(limit),)
    ).fetchall()
    return sessions_from_rows(rows, **kwargs)


# ------------------------------------------------------------------ baselines

@dataclass
class Forecast:
    value: float | None  # None wherever abstained
    abstained: bool
    reason: str = ""


class Baseline:
    """Causal streaming forecaster: observe() <= t, then predict at t+h."""

    name = "baseline"

    def observe(self, features: dict, ts: float) -> None:
        raise NotImplementedError

    def predict_features(
        self, feature_keys: list[str], ts: float, horizon_s: float
    ) -> dict[str, Forecast]:
        raise NotImplementedError

    def clone(self) -> Baseline:
        raise NotImplementedError


class NoOpBaseline(Baseline):
    """Always abstains: the floor every policy must beat by DOING BETTER,
    and the mirror of 'unknown is not actionable' from the risk layer."""

    name = "noop"

    def observe(self, features: dict, ts: float) -> None:  # noqa: ARG002
        return

    def predict_features(
        self, feature_keys: list[str], ts: float, horizon_s: float
    ) -> dict[str, Forecast]:  # noqa: ARG002
        return {k: Forecast(None, True, "noop-always") for k in feature_keys}

    def clone(self) -> NoOpBaseline:
        return NoOpBaseline()


class HoldLastBaseline(Baseline):
    """Persistence: predicted value = most recent observed value."""

    name = "hold"

    def __init__(self) -> None:
        self._last: dict[str, tuple[float, float]] = {}  # key -> (ts, value)

    def observe(self, features: dict, ts: float) -> None:
        for k, v in features.items():
            if v is not None:
                self._last[k] = (ts, float(v))

    def predict_features(
        self, feature_keys: list[str], ts: float, horizon_s: float
    ) -> dict[str, Forecast]:  # noqa: ARG002
        out = {}
        for k in feature_keys:
            if k in self._last:
                out[k] = Forecast(self._last[k][1], False)
            else:
                out[k] = Forecast(None, True, "no-history")
        return out

    def clone(self) -> HoldLastBaseline:
        return HoldLastBaseline()


class LinearTrendBaseline(Baseline):
    """OLS slope per feature over a trailing window; linear extrapolation.

    Per feature abstains until >= min_points valid samples AND >=
    min_window_s of temporal span within the window (short sessions
    honestly cannot support trends).
    """

    name = "linear"

    def __init__(
        self,
        window_s: float = 300.0,
        min_points: int = 3,
        min_window_s: float = 60.0,
    ) -> None:
        self.window_s = window_s
        self.min_points = min_points
        self.min_window_s = min_window_s
        self._series: dict[str, list[tuple[float, float]]] = {}

    def observe(self, features: dict, ts: float) -> None:
        for k, v in features.items():
            if v is None:
                continue
            series = self._series.setdefault(k, [])
            series.append((ts, float(v)))
            cutoff = ts - self.window_s
            while series and series[0][0] < cutoff:
                series.pop(0)

    def predict_features(
        self, feature_keys: list[str], ts: float, horizon_s: float
    ) -> dict[str, Forecast]:
        out = {}
        for k in feature_keys:
            series = self._series.get(k) or []
            if len(series) < self.min_points:
                out[k] = Forecast(None, True,
                                  f"insufficient-history({len(series)})")
                continue
            ts0 = series[0][0]
            t_last, v_last = series[-1]
            span = t_last - ts0
            if span < self.min_window_s:
                out[k] = Forecast(None, True, f"insufficient-span({span:.0f}s)")
                continue
            slope = _lin_slope(series)
            if slope is None:
                out[k] = Forecast(None, True, "degenerate-series")
                continue
            future = (ts + horizon_s) - t_last
            out[k] = Forecast(v_last + slope * future, False)
        return out

    def clone(self) -> LinearTrendBaseline:
        return LinearTrendBaseline(self.window_s, self.min_points,
                                   self.min_window_s)


def _lin_slope(series: list[tuple[float, float]]) -> float | None:
    n = len(series)
    if n < 2:
        return None
    mean_t = sum(t for t, _ in series) / n
    mean_v = sum(v for _, v in series) / n
    s_tt = sum((t - mean_t) ** 2 for t, _ in series)
    if s_tt <= 1e-12:
        return None
    s_tv = sum((t - mean_t) * (v - mean_v) for t, v in series)
    return s_tv / s_tt


# ---------------------------------------------------------------- evaluation

def evaluate_baselines(
    rows: list[sqlite3.Row],
    baselines: list[Baseline],
    horizons: tuple[float, ...] = EVAL_HORIZONS_S,
    *,
    feature_keys: list[str] | tuple[str, ...] = FEATURE_KEYS,
    median_interval_s: float | None = None,
    max_gap_factor: float = 3.0,
    min_gap_s: float = 10.0,
) -> dict[str, dict[float, dict[str, dict]]]:
    """Run the causal harness; see module docstring for integrity rules.

    Returns out[baseline_name][horizon][feature_key] = {
        n_pairs, mae, bias, n_abstained} — per-feature coverage always
    reported; mae None when zero pairs.
    """
    if median_interval_s is None:
        median_interval_s = median_interval(rows)
    sessions = sessions_from_rows(
        rows,
        median_interval_s=median_interval_s,
        max_gap_factor=max_gap_factor,
        min_gap_s=min_gap_s,
    )
    tol = 0.01 * max(median_interval_s, 1e-6)

    out: dict[str, dict[float, dict[str, dict]]] = {
        b.name: {h: {k: {"n_pairs": 0, "mae": None, "bias": None,
                         "n_abstained": 0}
                     for k in feature_keys} for h in horizons}
        for b in baselines
    }

    # contiguous exact-timestamp targets: a target at (ts + h) must exist
    # as a recorded row within tol, else that (anchor, horizon) is skipped.
    errors: dict[str, dict[float, dict[str, list[float]]]] = {
        b.name: {h: {k: [] for k in feature_keys} for h in horizons}
        for b in baselines
    }
    signs: dict[str, dict[float, dict[str, list[float]]]] = {
        b.name: {h: {k: [] for k in feature_keys} for h in horizons}
        for b in baselines
    }
    abstains: dict[str, dict[float, dict[str, int]]] = {
        b.name: {h: {k: 0 for k in feature_keys} for h in horizons}
        for b in baselines
    }

    for session in sessions:
        # Targets must live in the SAME session as the anchor: a horizon
        # landing in (or across) an outage gap yields no pair, ever.
        sess_rows = {round(float(r["ts"]), 6): r for r in session.rows}
        ts_list = sorted(sess_rows)
        for baseline in baselines:
            fresh = baseline.clone()
            rows_cur = session.rows
            for j in range(len(rows_cur)):
                anchor_row = rows_cur[j]
                anchor_ts = float(anchor_row["ts"])
                fresh.observe(sample_features(anchor_row), anchor_ts)
                for horizon in horizons:
                    want = anchor_ts + horizon
                    actual_row = None
                    idx = bisect.bisect_left(ts_list, want - tol)
                    if idx < len(ts_list):
                        cand = ts_list[idx]
                        if abs(cand - want) <= tol:
                            actual_row = sess_rows[cand]
                    if actual_row is None:
                        continue
                    preds = fresh.predict_features(
                        list(feature_keys), anchor_ts, horizon
                    )
                    feats_actual = sample_features(actual_row)
                    for key in feature_keys:
                        fc = preds[key]
                        actual = feats_actual.get(key)
                        if fc.abstained or actual is None:
                            if actual is not None:
                                abstains[baseline.name][horizon][key] += 1
                            continue
                        errors[baseline.name][horizon][key].append(
                            abs(fc.value - actual)
                        )
                        signs[baseline.name][horizon][key].append(
                            fc.value - actual
                        )

    for baseline in baselines:
        for horizon in horizons:
            for key in feature_keys:
                errs = errors[baseline.name][horizon][key]
                sgn = signs[baseline.name][horizon][key]
                out[baseline.name][horizon][key] = {
                    "n_pairs": len(errs),
                    "mae": (sum(errs) / len(errs)) if errs else None,
                    "bias": (sum(sgn) / len(sgn)) if sgn else None,
                    "n_abstained": abstains[baseline.name][horizon][key],
                }
    return out


# ------------------------------------------------- renice forward model

SPEC_RENICE_TWIN = {
    "mechanism": (
        "renice_back lowers the target's CPU priority (nice +step). RAM is "
        "NOT directly freed; the memory effect runs through reduced CPU "
        "contention (allocation threads progress, reclaim pressure falls). "
        "Estimand: delta psi_mem_stall_ratio and delta swp_out_ps between "
        "matched before/after windows, per unit of nice delta, on "
        "program-holding same-uid targets."
    ),
    "estimation_plan": (
        "For each applied renice_back ledger row: matching pre-window "
        "[ts_applied-H, ts_applied) and post window [ts_applied, "
        "ts_applied+H) from the SAME session; H in {30,60,120}s; effect = "
        "median over pairs of (post - pre) difference-in-differences vs a "
        "no-action matched segment; uncertainty from bootstrap; abstains "
        "below min pairs."
    ),
    "min_pairs": 8,
    "status_on_this_deployment": (
        "0 real action pairs recorded so far -> the model ABSTAINS on every "
        "prediction. This is the honest state, NOT a failure: it encodes "
        "'no evidence yet' per the same abstention rule the risk layer "
        "uses for stale telemetry."
    ),
}


class ReniceForwardModel:
    """Action-conditioned twin: predicts renice_back effects from ledger
    evidence. ABSTAINS everywhere until real evidence accumulates."""

    name = "renice_forward"

    def __init__(self, conn: sqlite3.Connection, min_pairs: int = 8) -> None:
        self._conn = conn
        self.min_pairs = min_pairs

    def evidence_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM actions_ledger"
            " WHERE kind='renice_back' AND status='applied'"
        ).fetchone()
        return int(row["n"])

    def predict(
        self, applied_nice: int, before_nice: int, horizon_s: float
    ) -> Forecast:
        n = self.evidence_count()
        if n < self.min_pairs:
            return Forecast(
                None, True,
                f"insufficient-real-evidence({n}<{self.min_pairs} ledger"
                f" pairs; {SPEC_RENICE_TWIN['status_on_this_deployment']})",
            )
        # Real-pairs path (future Phase 4): difference-in-differences per
        # SPEC_RENICE_TWIN. Unreachable until actions accumulate; refusal
        # to fabricate is the spec-correct behavior until then.
        raise NotImplementedError(
            "estimation activates only with real ledger evidence"
        )


# ------------------------------------------------------------------- report CLI

def _jsonify(report: dict[str, dict[float, dict[str, dict]]]) -> dict:
    """JSON-safe report: float horizon keys become strings."""
    return {
        baseline: {str(h): per_feature for h, per_feature in horizons.items()}
        for baseline, horizons in report.items()
    }


def main(argv: list[str] | None = None) -> int:
    """osrl-twin --db PATH [--horizons 60,120]: evaluate the baseline
    registry on a recorded telemetry DB and print a coverage-honest JSON
    report. Read-only over the store; observe-only by construction."""
    import argparse
    import json

    p = argparse.ArgumentParser(
        prog="osrl-twin",
        description=(
            "OS-RL-Agent digital twin: baseline forecast evaluation over"
            " recorded telemetry (no-op / hold / linear-trend)."
        ),
    )
    p.add_argument("--db", required=True, help="telemetry DB path")
    p.add_argument(
        "--horizons", default="30,60,120",
        help="comma-separated horizon seconds (default 30,60,120)",
    )
    args = p.parse_args(argv)
    horizons = tuple(float(x) for x in args.horizons.split(",") if x.strip())
    if not horizons or any(h <= 0 for h in horizons):
        p.error("horizons must be positive numbers")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM samples ORDER BY ts").fetchall()
        report = evaluate_baselines(
            rows,
            [NoOpBaseline(), HoldLastBaseline(), LinearTrendBaseline()],
            horizons=horizons,
        )
    finally:
        conn.close()
    print(json.dumps(_jsonify(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
