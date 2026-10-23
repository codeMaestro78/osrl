"""Evidence and coverage for risk decisions and telemetry forecasts."""

from __future__ import annotations

import json

import streamlit as st

from osrl.dashboard.shared import get_conn
from osrl.policy_eval import NaiveThresholdPolicy, RiskPolicy, evaluate_alarm_policies
from osrl.rl.dataset import quality_summary, split_rows
from osrl.twin import (
    EwmaBaseline,
    HoldLastBaseline,
    LinearTrendBaseline,
    NoOpBaseline,
    RollingMedianBaseline,
    evaluate_baselines,
)


def render() -> None:
    st.title("Evaluation & Data Quality")
    st.caption(
        "Measure recorded evidence before trusting a model. These checks do not execute OS actions."
    )
    conn = get_conn()
    # Keep an interactive report bounded; the training CLI snapshots the full dataset.
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM (SELECT * FROM samples ORDER BY ts DESC LIMIT 20000) ORDER BY ts"
        )
    ]
    quality = quality_summary(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Recent samples checked", f"{quality['samples']:,}")
    c2.metric("Memory PSI coverage", f"{quality['psi_coverage']:.0%}")
    c3.metric("Pressure samples (≥20%)", f"{quality['pressure_samples']:,}")
    if not quality["pressure_samples"]:
        st.info(
            "No memory-pressure samples in this range. Freeze detection accuracy needs "
            "pressure events to be measurable."
        )
    try:
        split = split_rows(rows)
    except ValueError as exc:
        st.warning(f"Training is waiting for usable data: {exc}")
        return
    st.subheader("Training readiness")
    a, b, c = st.columns(3)
    a.metric("Training samples", f"{split.summary['train_samples']:,}")
    b.metric("Held-out samples", f"{split.summary['eval_samples']:,}")
    c.metric("Boundary embargo", f"{split.summary['embargo_samples']} samples")
    st.caption(
        "Oldest 80% for learning; later telemetry for evaluation, with a full decision "
        "window excluded at the boundary. Sessions never bridge outages."
    )
    st.subheader("Compare policies and forecasts")
    horizon = st.selectbox("Forecast horizon", [30, 60, 120], format_func=lambda v: f"{v} seconds")
    st.caption(
        "The evaluation uses the most recent held-out data, capped at 2,000 samples for "
        "interactive use. Labels are sustained pressure signatures, not independently "
        "confirmed desktop freezes."
    )
    if st.button("Evaluate recorded telemetry", type="primary"):
        heldout = [r for session in split.evaluation for r in session][-2000:]
        with st.spinner("Evaluating causal forecasts and alarm decisions…"):
            report = {
                "dataset": split.summary,
                "quality": quality,
                "policies": evaluate_alarm_policies(
                    heldout, [RiskPolicy(), NaiveThresholdPolicy()]
                ),
                "forecasts": evaluate_baselines(
                    heldout,
                    [
                        NoOpBaseline(),
                        HoldLastBaseline(),
                        LinearTrendBaseline(),
                        EwmaBaseline(),
                        RollingMedianBaseline(),
                    ],
                    horizons=(float(horizon),),
                ),
            }
        st.session_state["evaluation_report"] = report
    report = st.session_state.get("evaluation_report")
    if report:
        st.subheader("Alarm policy results")
        st.dataframe(
            [{"Policy": name, **values} for name, values in report["policies"].items()],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Empty recall means no labelled events; empty precision means no alarm runs. "
            "Outage time is excluded from false alarms per hour."
        )
        st.subheader("Forecast error and coverage")
        table = []
        for name, horizons in report["forecasts"].items():
            for seconds, features in horizons.items():
                for feature, scores in features.items():
                    table.append(
                        {"Model": name, "Horizon s": seconds, "Feature": feature, **scores}
                    )
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            "Download evaluation report",
            json.dumps(report, indent=2),
            "osrl-evaluation.json",
            "application/json",
        )
        st.info(
            "Replay forecasts and alarm scores do not validate the benefit of renicing. "
            "Live policy promotion requires action-conditioned evidence."
        )
