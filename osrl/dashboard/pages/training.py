"""Training control plane: candidate evidence, runs, checkpoints and logs."""

from __future__ import annotations

import datetime as dt
import json

import plotly.graph_objects as go
import streamlit as st

from osrl.dashboard.shared import (
    checkpoint_dir,
    find_checkpoints,
    get_conn,
    recent_samples,
    training_available,
)
from osrl.rl.artifacts import read_manifest


def render() -> None:
    st.title("RL Training")
    st.caption(
        "Train reproducible candidates, inspect held-out results, and compare them with "
        "doing nothing."
    )
    conn = get_conn()
    directory = checkpoint_dir()
    refresh = st.session_state.get("refresh_s", 10)
    st.info(
        "Recorded replay cannot measure the benefit of an intervention. Candidates "
        "remain UNVALIDATED until action-conditioned outcomes can be evaluated."
    )

    @st.fragment(run_every=refresh)
    def metrics():
        rows = conn.execute(
            "SELECT ts, kind, payload FROM events WHERE kind LIKE 'rl_train%' "
            "ORDER BY id DESC LIMIT 1000"
        ).fetchall()
        events = []
        for row in reversed(rows):
            try:
                payload = json.loads(row["payload"])
            except (ValueError, TypeError):
                continue
            if isinstance(payload, dict):
                events.append({"ts": row["ts"], "kind": row["kind"], **payload})
        starts = [e for e in events if e["kind"] == "rl_train_start"]
        latest = starts[-1] if starts else None
        steps = [
            e
            for e in events
            if e["kind"] == "rl_train_step"
            and (latest is None or e["ts"] >= latest["ts"])
            and e.get("eval_mean_reward") is not None
        ]
        st.subheader("Latest run")
        if not latest and not steps:
            st.info(
                "No runs recorded yet. Collect telemetry, check Evaluation for readiness, then "
                "launch a candidate."
            )
        elif steps:
            last = steps[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Evaluated steps", f"{last.get('steps_done', 0):,}")
            c2.metric("Policy reward", f"{last['eval_mean_reward']:.3f}")
            baseline = last.get("baseline_reward")
            c3.metric("NOOP reward", f"{baseline:.3f}" if baseline is not None else "Unavailable")
            figure = go.Figure()
            for key, label, color in [
                ("eval_mean_reward", "Policy", "#7dd3fc"),
                ("baseline_reward", "NOOP", "#cbd5e1"),
            ]:
                figure.add_scatter(
                    x=[e.get("steps_done") for e in steps],
                    y=[e.get(key) for e in steps],
                    name=label,
                    mode="lines+markers",
                    line=dict(color=color),
                )
            figure.update_layout(
                height=300,
                margin=dict(t=24, b=36, l=40, r=20),
                xaxis_title="Actual training steps",
                yaxis_title="Raw episode reward",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(figure, key="reward_curve", width="stretch")
            st.caption(
                last.get(
                    "validation_reason",
                    "Legacy run: held-out evaluation provenance is unavailable.",
                )
            )
            if last.get("dataset"):
                with st.expander("Dataset split and evaluation evidence"):
                    st.json(
                        {
                            k: last[k]
                            for k in (
                                "run_id",
                                "dataset",
                                "policy_episode_rewards",
                                "noop_episode_rewards",
                                "reward_delta",
                                "reward_gate_passed",
                            )
                            if k in last
                        }
                    )
        else:
            st.info("Run started; waiting for its first checkpoint evaluation.")
        failures = [
            e
            for e in events
            if e["kind"] == "rl_train_failed" and (latest is None or e["ts"] >= latest["ts"])
        ]
        if failures:
            st.error(f"Training failed: {failures[-1].get('error', 'Check the training log.')}")

    metrics()
    st.divider()

    @st.fragment(run_every=refresh)
    def breakdown():
        st.subheader("Current telemetry reward breakdown")
        st.caption(
            "Computed from the latest ten samples; this is separate from held-out episode "
            "evaluation."
        )
        # Sample reconstruction imports Gymnasium, so use the dependency-free
        # collector types here to keep Cloud dashboards lightweight.
        from osrl.collector import PsiInterval, Sample
        from osrl.rl.reward import reward_breakdown

        rows = recent_samples(conn, 10)
        if not rows:
            st.info("No telemetry available.")
            return
        samples = []
        for row in rows:
            sample = Sample(
                ts=row["ts"],
                interval_s=row["interval_s"],
                cpu_total_pct=row["cpu_total_pct"],
                swap_used_pct=row["swap_used_pct"],
                swp_out_ps=row["swp_out_ps"],
                mem_available_bytes=row["mem_available_bytes"],
            )
            for resource in ("cpu", "mem", "io"):
                setattr(
                    sample,
                    f"psi_{resource}",
                    PsiInterval(some_stall_us=row[f"psi_{resource}_some_stall_us"]),
                )
            sample.top_procs = json.loads(row.get("top_procs_json") or "[]")
            samples.append(sample)
        terms = reward_breakdown(samples)
        st.bar_chart({"Contribution": {k: v for k, v in terms.items() if k != "total"}})
        st.caption(f"Total: {terms['total']:+.4f}")

    breakdown()
    st.divider()
    st.subheader("Checkpoints")
    checkpoints = find_checkpoints()
    if checkpoints:
        st.dataframe(
            [
                {
                    "Checkpoint": c["name"],
                    "Steps": c["steps"],
                    "Status": "VALIDATED" if c["validated"] else "UNVALIDATED",
                    "Saved": dt.datetime.fromtimestamp(c["mtime"]).isoformat(timespec="minutes"),
                    "Size MiB": round(c["size_mb"], 2),
                }
                for c in checkpoints
            ],
            hide_index=True,
            width="stretch",
        )
        selected = st.selectbox(
            "Evaluation report", checkpoints, format_func=lambda c: f"{c['path']}"
        )
        manifest = read_manifest(selected["path"])
        if manifest:
            with st.expander("Checkpoint evaluation report"):
                st.json(manifest)
            st.download_button(
                "Download checkpoint report",
                json.dumps(manifest, indent=2),
                "checkpoint-evaluation.json",
                "application/json",
            )
        else:
            st.caption("Legacy checkpoint: no matching evaluation report is available.")
    else:
        st.info("No checkpoints saved yet.")

    st.divider()
    st.subheader("Launch a candidate")
    available = training_available()
    if not available:
        st.info(
            "Training is unavailable in this deployment. Use a local environment with the "
            "RL extra and live telemetry."
        )
    with st.form("launch_form"):
        a, b, c = st.columns(3)
        steps = a.number_input("Steps (thousands)", 1, 500, 50, 1)
        synthetic = b.checkbox("Synthetic stress overlay", value=True)
        resume = c.checkbox("Resume latest compatible candidate", value=True)
        submitted = st.form_submit_button("Launch training", type="primary", disabled=not available)
    if submitted:
        from osrl.rl.auto_train import _launch_training

        db_path = conn.execute("PRAGMA database_list").fetchone()[2]
        try:
            process = _launch_training(db_path, steps * 1000, synthetic, directory, resume)
            st.success(
                f"Candidate launched (PID {process.pid}). "
                "A database lock prevents concurrent training."
            )
        except OSError as exc:
            st.error(f"Could not launch training: {exc}")

    @st.fragment(run_every=refresh)
    def logs():
        st.subheader("Training log")
        path = directory / "training.log"
        if path.exists():
            # Tail a bounded byte range, even after many scheduler runs.
            with path.open("rb") as source:
                source.seek(max(0, path.stat().st_size - 16000))
                tail = source.read().decode(errors="replace")
            st.code(tail, language="text")
        else:
            st.caption("No launcher log yet.")
        rows = conn.execute(
            "SELECT ts, kind, payload FROM events "
            "WHERE kind LIKE 'rl_train%' OR kind IN ('auto_train_failed','auto_train_finished') "
            "ORDER BY id DESC LIMIT 30"
        ).fetchall()
        if rows:
            with st.expander("Recent run events"):
                st.dataframe([dict(r) for r in rows], hide_index=True, width="stretch")

    logs()
