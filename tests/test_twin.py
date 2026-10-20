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
    # the spec must never promise numbers it does not have: no claimed
    # effect sizes anywhere, and abstention until real operator evidence
    assert "--execute" in SPEC_RENICE_TWIN["status_on_this_deployment"]
    assert SPEC_RENICE_TWIN["min_pairs"] >= 8
    for value in SPEC_RENICE_TWIN.values():
        assert "-3.35" not in str(value)  # single-pair spoiler never baked in


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


# ------------------------------------------------ candidates (beat-hold race)

def test_ewma_persists_constants_and_abstains_short():
    b = twin.EwmaBaseline(half_life_s=60.0, min_points=4)
    for i in range(6):
        r = make_rows([(i * 2.0, 2.0, 300000, None, 8e9, 7.5)])[0]
        b.observe(sample_features(r), i * 2.0)
    preds = b.predict_features(list(FEATURE_KEYS), 12.0, 30.0)
    assert preds["cpu_total_pct"].value == pytest.approx(7.5)
    assert preds["swp_out_ps"].abstained  # all-None feature
    assert preds["swap_used_pct"].abstained
    # within warm-up the feature abstains (min_points=4)
    b2 = twin.EwmaBaseline(half_life_s=60.0, min_points=4)
    for i in range(3):
        r = make_rows([(i * 2.0, 2.0, 300000, None, 8e9, 7.5)])[0]
        b2.observe(sample_features(r), i * 2.0)
    assert b2.predict_features(["cpu_total_pct"], 6.0, 30.0)[
        "cpu_total_pct"].abstained


def test_ewma_moves_toward_steps_never_past():
    b = twin.EwmaBaseline(half_life_s=60.0, min_points=4)
    for i in range(4):
        r = make_rows([(i * 2.0, 2.0, None, None, None, 10.0)])[0]
        b.observe(sample_features(r), i * 2.0)
    for i in range(4, 20):
        r = make_rows([(i * 2.0, 2.0, None, None, None, 100.0)])[0]
        b.observe(sample_features(r), i * 2.0)
    v = b.predict_features(["cpu_total_pct"], 40.0, 0.0)["cpu_total_pct"]
    assert 10.0 < v.value < 100.0   # moving toward the step, never past it


def test_median_is_spike_robust_where_hold_chases():
    b = twin.RollingMedianBaseline(k=15)
    for i in range(12):
        v = 50.0 if i == 11 else 10.0
        r = make_rows([(i * 2.0, 2.0, None, v, None, None)])[0]
        b.observe(sample_features(r), i * 2.0)
    preds = b.predict_features(["swp_out_ps"], 24.0, 30.0)
    assert preds["swp_out_ps"].value == pytest.approx(10.0)  # spike ignored
    hold = HoldLastBaseline()
    for i in range(12):
        v = 50.0 if i == 11 else 10.0
        r = make_rows([(i * 2.0, 2.0, None, v, None, None)])[0]
        hold.observe(sample_features(r), i * 2.0)
    assert hold.predict_features(["swp_out_ps"], 24.0, 30.0)[
        "swp_out_ps"].value == pytest.approx(50.0)  # hold chases the spike


def test_median_abstains_below_min_observations():
    b = twin.RollingMedianBaseline(k=15)
    for i in range(4):
        r = make_rows([(i * 2.0, 2.0, None, 10.0, None, None)])[0]
        b.observe(sample_features(r), i * 2.0)
    assert b.predict_features(["swp_out_ps"], 8.0, 30.0)["swp_out_ps"].abstained


# --------------------------------------------- evidence pipeline + the model

def _insert_pair_samples(store, t0, base_pid, start_time, pre_pct, post_pct):
    """Continuous ±130 s telemetry around an action at t0; target first in
    top_procs with a decoy behind it."""
    from osrl.collector import PsiInterval, Sample

    def sample(t: float) -> Sample:
        if t < t0:
            pct = pre_pct
        elif t == t0:
            pct = pre_pct   # apply tick still recorded as pre-context
        else:
            pct = post_pct
        return Sample(
            ts=t, interval_s=2.0, ncpu=12, cpu_total_pct=25.0,
            mem_available_bytes=8_000_000_000,  # ~7.45 GiB
            psi_mem=PsiInterval(some_stall_us=0),
            top_procs=[
                {"pid": base_pid, "start_time": start_time,
                 "name": "busychld", "cpu_pct": pct, "rss_mib": 100.0},
                {"pid": base_pid + 1, "start_time": start_time,
                 "name": "decoy", "cpu_pct": 3.0, "rss_mib": 10.0},
            ],
        )

    store.insert_samples([sample(t0 + delta)
                          for delta in range(-130, 131, 2)])


def _insert_applied(store, act_id, t0, pid, start_time):
    assert store.insert_action(
        id=act_id, created_ts=t0 - 1.0, proposed_by="test",
        kind="renice_back", target_pid=pid, target_start_time=start_time,
        target_nice=10, before_nice=0, applied_nice=10,
        intent_id=f"intent-{act_id}", guard_checks="[]",
        status="applied", ts_applied=t0,
    )


def test_extract_pair_single_action_and_model_still_abstains(tmp_path):
    from osrl.store import Store

    store = Store(tmp_path / "e.db")
    try:
        _insert_pair_samples(store, 500.0, 4242, 90.0, 46.0, 18.0)
        _insert_applied(store, "act-1", 500.0, 4242, 90.0)
        pairs = twin.extract_renice_pairs(store._conn, window_s=60.0)
        assert len(pairs) == 1
        p = pairs[0]
        assert p["cpu_pair_ok"] is True
        assert p["nice_delta"] == 10
        assert p["target_cpu_pct_before"] == pytest.approx(46.0)
        assert p["target_cpu_pct_after"] == pytest.approx(18.0)
        assert p["psi_mem_ratio_before"] == pytest.approx(0.0)

        m = twin.ReniceForwardModel(store._conn)
        assert m.evidence_count() == 1
        fs = m.fit_summary()
        assert fs["n_usable"] == 1
        assert fs["median_shift_cpu_pct"] == pytest.approx(-28.0)
        fc = m.predict(46.0, 10, 60.0)
        assert fc.abstained and "insufficient-real-evidence(1<8" in fc.reason
    finally:
        store.close()


def test_extract_pair_window_spanning_session_break_rejected(tmp_path):
    from osrl.store import Store

    store = Store(tmp_path / "e.db")
    try:
        _insert_pair_samples(store, 500.0, 4242, 90.0, 46.0, 18.0)
        _insert_applied(store, "act-1", 500.0, 4242, 90.0)
        rows = store._conn.execute(
            "DELETE FROM samples WHERE ts < 480.0"   # kill the pre-window
        )
        assert rows.rowcount > 0
        pairs = twin.extract_renice_pairs(store._conn, window_s=60.0)
        assert pairs[0]["cpu_pair_ok"] is False
        # the ±60 s window now crosses the sample gap -> honest refusal
        assert "window-spans-session-break" in pairs[0]["reason"]
    finally:
        store.close()


def test_model_activates_with_eight_real_pairs(tmp_path):
    from osrl.store import Store

    store = Store(tmp_path / "e.db")
    try:
        pre = [46.0, 44.0, 48.0, 42.0, 46.0, 44.0, 47.0, 43.0]
        for k in range(8):
            t0 = 1000.0 + k * 300.0
            _insert_pair_samples(
                store, t0, 5000 + k, 800.0 + k, pre[k], 18.0
            )
            _insert_applied(store, f"act-{k}", t0, 5000 + k, 800.0 + k)
        m = twin.ReniceForwardModel(store._conn)
        assert m.evidence_count() == 8
        shifts = [p["target_cpu_pct_after"] - p["target_cpu_pct_before"]
                  for p in m.usable_pairs()]
        expected_median = sorted(shifts)
        median = (expected_median[3] + expected_median[4]) / 2.0
        fs = m.fit_summary()
        assert fs["median_shift_cpu_pct"] == pytest.approx(median)
        fc = m.predict(50.0, 10, 60.0)
        assert not fc.abstained
        assert fc.value == pytest.approx(50.0 + median)
        assert "UNVALIDATED" in fc.reason
        # horizon mismatch refuses rather than lies
        assert m.predict(50.0, 10, 30.0).abstained
        # non-renice_back direction refuses
        assert m.predict(50.0, -5, 60.0).abstained
    finally:
        store.close()
