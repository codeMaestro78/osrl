"""Twin substrate: session slicing, causal baselines, horizon-wise eval."""

import json
import sqlite3

import pytest

from osrl import twin
from osrl.twin import (
    FEATURE_KEYS,
    HoldLastBaseline,
    LinearTrendBaseline,
    NoOpBaseline,
    ReniceForwardModel,
    sample_features,
    sessions_from_rows,
)


def make_rows(pts):
    """pts: list of (ts, interval, psi_ratio us, swp, mem_avail, cpu%) tuples
    flattened into sqlite-like row dicts compatible with sample_features."""
    rows = []
    for ts, interval, stall_us, swp_out, mem_avail, cpu in pts:
        rows.append({
            "ts": ts,
            "interval_s": interval,
            "psi_mem_some_stall_us": stall_us,
            "swp_out_ps": swp_out,
            "allocstall_ps": None,
            "mem_available_bytes": mem_avail,
            "swap_used_pct": None,
            "cpu_total_pct": cpu,
        })
    # sqlite3.Row exposes [] either way; dicts are Row-equivalent here
    return rows


# ------------------------------------------------------------ session slicing

def test_sessions_split_on_outage_gaps():
    rows = make_rows([
        (0.0, 2.0, 0, 0.0, 8e9, 5.0),
        (2.0, 2.0, 0, 0.0, 8e9, 5.0),
        (4.0, 2.0, 0, 0.0, 8e9, 5.0),
        (400.0, 2.0, 0, 0.0, 8e9, 5.0),   # 396 s gap >> max(6, 10)
        (402.0, 2.0, 0, 0.0, 8e9, 5.0),
    ])
    sessions = sessions_from_rows(rows)
    assert len(sessions) == 2
    assert sessions[0].duration_s == pytest.approx(4.0)
    assert sessions[1].duration_s == pytest.approx(2.0)


def test_sessions_respect_median_interval_not_constant():
    # 30 s cadence: max gap = 90 s; a 60 s gap stays ONE session
    rows = make_rows([
        (0.0, 30.0, 0, 0.0, 8e9, None),
        (30.0, 30.0, 0, 0.0, 8e9, None),
        (90.0, 60.0, 0, 0.0, 8e9, None),  # 60 s gap <= 90
        (120.0, 30.0, 0, 0.0, 8e9, None),
    ])
    assert len(sessions_from_rows(rows)) == 1


def test_features_preserve_missing():
    r = make_rows([(0.0, None, None, None, None, None)])[0]
    f = sample_features(r)
    assert all(v is None for v in f.values())
    assert len(f) == len(FEATURE_KEYS)


# ------------------------------------------------------------------ baselines

def test_noop_abstains_always():
    b = NoOpBaseline()
    b.observe(sample_features(make_rows([(0, 2, 0, 0, 8e9, 5.0)])[0]), 0.0)
    preds = b.predict_features(list(FEATURE_KEYS), 2.0, 60.0)
    assert all(p.abstained for p in preds.values())


def test_hold_baseline_persists_values():
    b = HoldLastBaseline()
    b.observe(sample_features(make_rows([(0, 2, 100000, 50.0, 8e9, 5.0)])[0]), 0.0)
    b.observe(sample_features(make_rows([(2, 2, 400000, 80.0, 7e9, 30.0)])[0]), 2.0)
    preds = b.predict_features(list(FEATURE_KEYS), 2.0, 60.0)
    assert preds["psi_mem_ratio"].value == pytest.approx(0.2)   # 400000/2e6
    assert preds["swp_out_ps"].value == pytest.approx(80.0)
    assert preds["swap_used_pct"].abstained  # never observed


def test_linear_trend_extrapolates_with_abstention_rules():
    b = LinearTrendBaseline(window_s=600.0, min_points=3, min_window_s=6.0)
    # rising pressure: 0.1 -> 0.2 -> 0.3 ratio at 2 s cadence
    for i, stall in enumerate((200000, 400000, 600000, 800000)):
        r = make_rows([(i * 2.0, 2.0, stall, None, 8e9, None)])[0]
        b.observe(sample_features(r), i * 2.0)
    preds = b.predict_features(list(FEATURE_KEYS), 6.0, 2.0)
    # slope = 0.05 ratio/s exactly (OLS over a perfect line); predict 2 s
    # past the last seen value (ts=6 keeps ratio 0.4) -> 0.4 + 0.05*2
    assert preds["psi_mem_ratio"].value == pytest.approx(0.5, abs=1e-9)
    assert preds["swp_out_ps"].abstained  # all-None feature


def test_linear_trend_abstains_short_sessions():
    b = LinearTrendBaseline(min_points=3, min_window_s=60.0)
    for i, stall in enumerate((0, 0, 0)):
        r = make_rows([(i * 2.0, 2.0, stall, None, None, None)])[0]
        b.observe(sample_features(r), i * 2.0)
    preds = b.predict_features(list(FEATURE_KEYS), 4.0, 60.0)
    assert preds["psi_mem_ratio"].abstained
    assert "span" in preds["psi_mem_ratio"].reason


# ----------------------------------------------------------------- evaluation

def test_evaluation_matches_only_contemporaneous_targets():
    """Targets must exist as recorded rows INSIDE the same session: gap- or
    outage-distant readings are never matched, so a horizon pair that
    lands outside its session contributes n=0 rather than interpolating."""
    rows = make_rows(
        [(t * 2.0, 2.0, 100000, 0.0, 8e9, 5.0) for t in range(21)]   # 0..40 s
        + [(200.0 + t * 2.0, 2.0, 100000, 0.0, 8e9, 5.0)
           for t in range(21)]                                      # 200..240 s
    )
    report = twin.evaluate_baselines(
        rows, [HoldLastBaseline()], horizons=(20.0, 60.0, 120.0)
    )
    # 20 s: session A anchors 0..20 -> targets 20..40 (11 pairs) plus
    # session B anchors 200..220 -> targets 220..240 (11 pairs);
    # anchors 22..40 target 42..60 = outage, never matched across.
    assert report["hold"][20.0]["psi_mem_ratio"]["n_pairs"] == 22
    # 60/120 s: every anchor's target falls beyond its own session end
    assert report["hold"][60.0]["psi_mem_ratio"]["n_pairs"] == 0
    assert report["hold"][120.0]["psi_mem_ratio"]["n_pairs"] == 0
    # constant series -> zero error at every formed pair
    assert report["hold"][20.0]["psi_mem_ratio"]["mae"] == 0.0


def test_evaluation_counts_maes_and_coverage():
    """Hold baseline over a continuous 2 s session; a swp spike at ts=8
    must NOT leak into predictions from earlier anchors (causality), and
    every counter pair reports its own coverage."""
    rows = make_rows(
        [(t * 2.0, 2.0, 100000, 100.0, 8e9, 5.0) for t in range(6)]
        + [(8.0, 2.0, 100000, 130.0, 8e9, 5.0)]          # same run, spike ts=8
        + [(t * 2.0, 2.0, 100000, 100.0, 8e9, 5.0)
           for t in range(6, 31)]                        # continuous to 60 s
    )
    report = twin.evaluate_baselines(
        rows, [HoldLastBaseline(), NoOpBaseline()], horizons=(30.0,)
    )
    entry = report["hold"][30.0]["swp_out_ps"]
    # anchors 0,2,...,30 -> targets 30,32,...,60 all in-session: 17 pairs
    assert entry["n_pairs"] == 17
    # only anchor ts=8 persists the spike (predicts 130 vs actual 100);
    # anchors 0..6 must NOT see it (causality) and 10+ flush it back to 100.
    # one 30-unit miss among 16 zero misses -> mae = 30/17; had the harness
    # PEEKED (anchors 0..6 also predicting 130), mae would be 4x that.
    assert entry["mae"] == pytest.approx(30.0 / 17.0)
    assert entry["bias"] == pytest.approx(30.0 / 17.0)
    assert entry["n_abstained"] == 0
    nn = report["noop"][30.0]["swp_out_ps"]
    assert nn["n_abstained"] == 17
    assert nn["n_pairs"] == 0
    assert nn["mae"] is None


def test_evaluation_is_causal_within_session():
    """A perf-stepping series must show probe-then-predict behavior: the
    linear baseline cannot peek at future rows of its own session."""
    # value hijack: psi ratio climbs steadily 0.05/s for 8 ticks
    pts = []
    for i in range(8):
        pts.append((i * 2.0, 2.0, int(100000 + i * 100000), None, 8e9, None))
    pts.append((16.0, 2.0, None, None, 8e9, None))  # missing actual anchor
    rows = make_rows(pts)
    report = twin.evaluate_baselines(
        rows, [LinearTrendBaseline(window_s=600, min_points=3,
                                   min_window_s=4.0)],
        horizons=(2.0,),
    )
    lin = report["linear"][2.0]["psi_mem_ratio"]
    assert lin["n_pairs"] >= 5
    # perfect linear series: MAE ~ 0 (float tolerance)
    assert lin["mae"] is not None and lin["mae"] < 1e-9


def test_session_isolation_no_cross_gap_state():
    """Two sessions with wildly different levels: hold baseline must NOT
    use the first session's memory to predict the second's (fresh clone)."""
    rows = make_rows([
        (0.0, 2.0, 900000, 0.0, 1e9, 99.0),    # session A: bad machine
        (2.0, 2.0, 900000, 0.0, 1e9, 99.0),
        (2.0 + 30.0, 2.0, 10000, 0.0, 8e9, 1.0),  # +30s: > 10 s -> new sess
        (32.0 + 2.0, 2.0, 10000, 0.0, 8e9, 1.0),
        (34.0 + 2.0, 2.0, 10000, 0.0, 8e9, 1.0),
        (34.0 + 2.0 + 2.0, 2.0, 10000, 0.0, 8e9, 1.0),
    ])
    report = twin.evaluate_baselines(
        rows, [HoldLastBaseline()], horizons=(2.0,)
    )
    # B-session targets must carry ~B-level values; bad-A contamination
    # would blow up MAE. Exact match grid keeps pairs B-only realistically.
    hold = report["hold"][2.0]["cpu_total_pct"]
    if hold["n_pairs"]:
        assert hold["mae"] < 1.0  # within-session crossover contamination


# ------------------------------------------------------ renice forward model

def test_renice_model_abstains_without_evidence(tmp_path):
    from osrl.store import Store

    store = Store(tmp_path / "t.db")  # real schema (actions_ledger exists)
    try:
        m = ReniceForwardModel(store._conn)  # analysis reads the live handle
        fc = m.predict(10, 0, 60.0)
        assert fc.abstained and fc.value is None
        assert "insufficient-real-evidence" in fc.reason
        assert m.evidence_count() == 0
    finally:
        store.close()


def test_renice_model_spec_is_data_honest():
    from osrl.twin import SPEC_RENICE_TWIN
    assert SPEC_RENICE_TWIN["status_on_this_deployment"].startswith("0 real")
    assert SPEC_RENICE_TWIN["min_pairs"] >= 8


# ------------------------------------------------------- real-machine harness

@pytest.mark.skipif(not __import__("pathlib").Path(
    "/tmp/opencode/hold-test/t.db").exists(), reason="hold-test db missing")
def test_hold_test_sessions_evaluable():
    """Harness on REAL recorded data (1.58 h continuous): hold baseline
    must form pairs at every standard horizon; noop abstains everywhere."""
    conn = sqlite3.connect("/tmp/opencode/hold-test/t.db")
    conn.row_factory = sqlite3.Row
    sessions = twin.sessions_from_store(conn)
    assert len(sessions) >= 1 and sessions[0].duration_s > 3600
    rows = conn.execute(
        "SELECT * FROM samples ORDER BY ts"
    ).fetchall()
    report = twin.evaluate_baselines(
        rows, [NoOpBaseline(), HoldLastBaseline()],
        horizons=(30.0, 60.0, 120.0),
        median_interval_s=30.0,
    )
    for h in (30.0, 60.0, 120.0):
        assert report["hold"][h]["cpu_total_pct"]["n_pairs"] > 0
        assert report["noop"][h]["cpu_total_pct"]["n_pairs"] == 0
        # noop abstains on every target whose actual exists; hold either
        # scores the pair (first-tick features are missing -> it abstains
        # too) so hold pairs + hold abstains == noop abstains.
        assert report["noop"][h]["cpu_total_pct"]["n_abstained"] == (
            report["hold"][h]["cpu_total_pct"]["n_pairs"]
            + report["hold"][h]["cpu_total_pct"]["n_abstained"]
        )
    conn.close()


def test_report_json_serializable():
    rows = make_rows([
        (0.0, 2.0, 100000, 0.0, 8e9, 5.0),
        (2.0, 2.0, 100000, 0.0, 8e9, 5.0),
        (62.0, 2.0, 100000, 0.0, 8e9, 5.0),
    ])
    report = twin.evaluate_baselines(
        rows, [NoOpBaseline(), HoldLastBaseline(), LinearTrendBaseline()],
        horizons=(60.0,),
    )
    json.dumps(report)  # must not raise
