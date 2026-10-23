"""Premium Agent page — mode control, policy status, reward history, action ledger."""

from __future__ import annotations

import datetime as dt
import json

import plotly.graph_objects as go
import streamlit as st

from osrl.dashboard.shared import (
    agent_mode,
    best_checkpoint,
    find_checkpoints,
    get_conn,
    set_agent_mode,
)

_BG, _PLOT = "rgba(0,0,0,0)", "rgba(6,13,26,0.95)"
_FONT = dict(color="#94a3b8", size=11, family="Inter")
_GRID = "rgba(56,189,248,0.05)"
_CHART = dict(
    paper_bgcolor=_BG, plot_bgcolor=_PLOT, font=_FONT,
    margin=dict(t=44, b=32, l=52, r=24), height=280,
    xaxis=dict(gridcolor=_GRID, showline=False, zeroline=False, title=""),
    yaxis=dict(gridcolor=_GRID, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#0f1f3d", font_color="#e2e8f0",
                    bordercolor="rgba(56,189,248,0.4)"),
)

_MODE_META = {
    "observe":   ("#38bdf8", "👁️",  "Watches silently. No actions taken."),
    "advise":    ("#f59e0b", "💡",  "Generates recommendations for human review."),
    "autopilot": ("#818cf8", "🤖", "Queues eligible intents; a separate worker needs explicit --execute approval."),
}


def _mode_card(mode: str, current: str) -> str:
    color, icon, desc = _MODE_META.get(mode, ("#64748b", "•", ""))
    active = mode == current
    border = f"2px solid {color}" if active else "1px solid rgba(56,189,248,0.1)"
    bg = f"rgba({','.join(str(int(color[i:i+2], 16)) for i in (1,3,5))},0.12)" if active else "rgba(10,20,45,0.7)"
    glow = f"0 0 20px {color}30" if active else "none"
    badge = (f"<span style='background:{color}22;color:{color};font-size:9px;"
             f"font-weight:700;letter-spacing:1px;padding:2px 8px;border-radius:20px;"
             f"text-transform:uppercase'>ACTIVE</span>") if active else ""
    return f"""
<div style='background:{bg};border:{border};border-radius:14px;
            padding:18px;text-align:center;box-shadow:{glow};
            transition:all 0.3s ease;animation:slide-in-up 0.4s ease-out'>
    <div style='font-size:2rem;margin-bottom:6px'>{icon}</div>
    <div style='color:#f1f5f9;font-weight:700;font-size:15px;
                text-transform:capitalize;margin-bottom:4px'>{mode}</div>
    <div style='color:#94a3b8;font-size:11px;margin-bottom:10px'>{desc}</div>
    {badge}
</div>"""


def render() -> None:
    st.title("🤖 RL Agent")
    st.caption("Mode is a stored preference. This dashboard does not start inference or an execution worker.")
    conn    = get_conn()
    refresh = st.session_state.get("refresh_s", 10)

    # ── Mode selector (fragment) ──────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _mode_selector() -> None:
        current = agent_mode(conn)
        color, icon, _ = _MODE_META.get(current, ("#64748b", "•", ""))

        # Animated banner
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(10,20,45,0.95),rgba(15,30,65,0.85));
                    border:1px solid {color}33;border-radius:16px;padding:20px 24px;
                    display:flex;align-items:center;gap:16px;margin-bottom:4px;
                    box-shadow:0 0 30px {color}15;animation:slide-in-down 0.5s ease-out'>
            <div style='font-size:2.5rem;filter:drop-shadow(0 0 10px {color}80)'>{icon}</div>
            <div>
                <div style='color:#94a3b8;font-size:9px;text-transform:uppercase;
                            letter-spacing:2px;font-weight:700'>Current Mode</div>
                <div style='color:{color};font-size:26px;font-weight:800;
                            letter-spacing:-0.5px;font-family:Inter,sans-serif'>{current.upper()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mode cards
        c1, c2, c3 = st.columns(3)
        c1.markdown(_mode_card("observe",   current), unsafe_allow_html=True)
        c2.markdown(_mode_card("advise",    current), unsafe_allow_html=True)
        c3.markdown(_mode_card("autopilot", current), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Mode change
        new_mode = st.selectbox(
            "Switch Mode",
            ["observe", "advise", "autopilot"],
            index=["observe", "advise", "autopilot"].index(current),
            key="mode_select",
        )
        if st.button("Apply Mode Change", type="primary", key="apply_mode"):
            if new_mode != current:
                if "confirm_mode" not in st.session_state or st.session_state.confirm_mode != new_mode:
                    st.session_state.confirm_mode = new_mode
                    st.warning(f"⚠️ Click **Apply Mode Change** again to confirm switching to **{new_mode.upper()}**")
                    st.stop()
                else:
                    set_agent_mode(conn, new_mode)
                    st.session_state.confirm_mode = None
                    new_c, new_i, _ = _MODE_META.get(new_mode, ("#64748b", "•", ""))
                    st.success(f"{new_i} Mode switched to **{new_mode.upper()}**")
                    st.rerun()
            else:
                st.info(f"Already in {current.upper()} mode.")

    _mode_selector()
    st.divider()

    # ── Policy checkpoint status ──────────────────────────────────────────────
    @st.fragment(run_every=refresh * 2)
    def _policy_status() -> None:
        st.subheader("🏆 Policy Checkpoint")
        best = best_checkpoint()
        all_ckpts = find_checkpoints()

        if not best and not all_ckpts:
            st.markdown("""
            <div style='background:rgba(8,15,31,0.7);border:1px dashed rgba(56,189,248,0.2);
                        border-radius:12px;padding:24px;text-align:center'>
                <div style='font-size:2.5rem;margin-bottom:8px'>🎯</div>
                <div style='color:#94a3b8'>No checkpoint yet — run training first.</div>
            </div>""", unsafe_allow_html=True)
            return

        c1, c2, c3, c4 = st.columns(4)
        n_val = sum(1 for c in all_ckpts if c.get("validated"))
        n_all = len(all_ckpts)

        if best:
            c1.metric("Best Checkpoint", "✅ FOUND")
            c2.metric("Status", "VALIDATED" if best.get("validated") else "UNVALIDATED")
            c3.metric("Steps", f"{best.get('steps', 0):,}")
        c4.metric("Total Checkpoints", f"{n_val}/{n_all} validated")

        if best:
            st.markdown(f"""
            <div style='background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);
                        border-radius:10px;padding:12px 16px;margin-top:8px;
                        font-family:JetBrains Mono,monospace;font-size:12px'>
                <span style='color:#22c55e'>🏆 BEST → </span>
                <span style='color:#64748b'>{best.get("name","")}</span>
            </div>""", unsafe_allow_html=True)

    _policy_status()
    st.divider()

    # ── Recommendation uncertainty ───────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _recommendation_assessment() -> None:
        st.subheader("Recommendation Evidence")
        row = conn.execute(
            "SELECT ts, payload FROM events WHERE kind='policy_suggestion'"
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            st.caption("No inference recommendation has been recorded yet.")
            return
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError):
            st.warning("The latest recommendation record could not be parsed.")
            return
        uncertainty = payload.get("uncertainty") or {}
        c1, c2, c3 = st.columns(3)
        probability = payload.get("probability")
        c1.metric("Policy probability", "—" if probability is None else f"{probability:.1%}")
        age = uncertainty.get("telemetry_age_s")
        c2.metric("Telemetry age", "—" if age is None else f"{age:.1f} s")
        count = uncertainty.get("relevant_outcomes")
        c3.metric("Relevant measured outcomes", "—" if count is None else str(count))
        st.caption(
            f"Latest action: {payload.get('action', 'unknown')} · "
            f"mode: {payload.get('mode', 'unknown')} · "
            f"risk state: {uncertainty.get('risk_level', 'unknown')} · "
            "policy probability is not calibrated confidence."
        )
        reasons = uncertainty.get("reasons") or []
        if reasons:
            st.warning("Recommendation abstained: " + ", ".join(map(str, reasons)))
        elif payload.get("intent_gate_reason"):
            st.info("No intent queued: " + str(payload["intent_gate_reason"]))
        else:
            st.success("The recommendation passed the recorded uncertainty checks.")
        if uncertainty.get("signal_disagreement") is True:
            st.caption("The current risk state conflicts with the latest pressure signals.")

    _recommendation_assessment()
    st.divider()

    # ── Measured action outcomes ──────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _action_outcomes() -> None:
        st.subheader("Intervention Outcomes")
        rows = conn.execute(
            "SELECT intent_id, action_id, created_ts, status, details"
            " FROM action_outcomes ORDER BY created_ts DESC LIMIT 30"
        ).fetchall()
        if not rows:
            st.caption("No guarded intervention attempts have been recorded.")
            return
        table = []
        for row in rows:
            try:
                details = json.loads(row["details"] or "{}")
            except (TypeError, ValueError):
                details = {}
            measurement = details.get("measurement") or {}
            status = row["status"]
            result = {
                "pending": "Waiting for post-action window",
                "measured": "Measured before/after",
                "insufficient_telemetry": "Insufficient telemetry",
                "not_applied": details.get("attempt_status", "Not applied"),
            }.get(status, status)
            recovered = measurement.get("recovered_seconds")
            psi_change = measurement.get("psi_mem_change")
            cpu_change = measurement.get("target_cpu_pct_delta")
            system_cpu_before = measurement.get("cpu_before_avg_pct")
            system_cpu_after = measurement.get("cpu_after_avg_pct")
            mem_before = measurement.get("mem_available_before_avg_bytes")
            mem_after = measurement.get("mem_available_after_avg_bytes")
            system_cpu_change = (
                system_cpu_after - system_cpu_before
                if system_cpu_before is not None and system_cpu_after is not None
                else None
            )
            mem_change_mib = (
                (mem_after - mem_before) / (1024**2)
                if mem_before is not None and mem_after is not None
                else None
            )
            table.append({
                "Intent": str(row["intent_id"])[:8],
                "Action": str(row["action_id"] or "—")[:8],
                "Attempt": dt.datetime.fromtimestamp(float(row["created_ts"])).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Result": result,
                "Reason": details.get("reason", measurement.get("reason", "")),
                "Pressure before": measurement.get("pressure_before", "—"),
                "Pressure after": measurement.get("pressure_after", "—"),
                "Recovery s": "—" if recovered is None else round(float(recovered), 1),
                "Pressure returned": (
                    "—" if measurement.get("pressure_returned") is None
                    else str(bool(measurement["pressure_returned"]))
                ),
                "Memory PSI Δ": "—" if psi_change is None else f"{float(psi_change):+.1%}",
                "System CPU Δ": (
                    "—" if system_cpu_change is None else f"{float(system_cpu_change):+.1f} pp"
                ),
                "Available Memory Δ": (
                    "—" if mem_change_mib is None else f"{float(mem_change_mib):+.0f} MiB"
                ),
                "Target CPU Δ": "—" if cpu_change is None else f"{float(cpu_change):+.1f} pp",
                "Foreground impact": "Not collected",
            })
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption(
            "Applied actions are measured over a 30 s pre-window and 120 s post-window. "
            "Results are observational, not causal; active foreground-app identity "
            "is not collected. Rejected and failed attempts remain visible without "
            "attributing telemetry changes to them."
        )

    _action_outcomes()
    st.divider()

    # ── Reward trend chart ────────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _reward_chart() -> None:
        st.subheader("📈 Policy Reward History")
        rows = [dict(r) for r in conn.execute(
            "SELECT ts, payload FROM events WHERE kind='rl_train_step' ORDER BY ts"
        ).fetchall()]
        if not rows:
            st.caption("No training steps recorded yet.")
            return

        xs, ys, baselines, validated_x, validated_y = [], [], [], [], []
        for r in rows:
            try: p = json.loads(r["payload"])
            except Exception: continue  # noqa: BLE001
            ts_str = dt.datetime.fromtimestamp(float(r["ts"])).strftime("%m/%d %H:%M")
            xs.append(ts_str)
            ys.append(p.get("eval_mean_reward", 0))
            baselines.append(p.get("baseline_reward", 0))
            if p.get("validated"):
                validated_x.append(ts_str)
                validated_y.append(p.get("eval_mean_reward", 0))

        fig = go.Figure()

        # Baseline
        fig.add_scatter(x=xs, y=baselines, mode="lines", name="NOOP Baseline",
                       line=dict(color="rgba(100,116,139,0.5)", dash="dot", width=1.5))

        # Fill between baseline and reward
        if ys and baselines:
            fig.add_scatter(
                x=xs + xs[::-1],
                y=ys + baselines[::-1],
                fill="toself", fillcolor="rgba(56,189,248,0.04)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="Gap",
            )

        # Reward line with glow effect via marker
        fig.add_scatter(
            x=xs, y=ys, mode="lines+markers", name="Policy Reward",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(color="#38bdf8", size=6, line=dict(color="#fff", width=1)),
        )

        # Validated markers (gold stars)
        if validated_x:
            fig.add_scatter(
                x=validated_x, y=validated_y, mode="markers", name="✅ Validated",
                marker=dict(color="#fbbf24", size=14, symbol="star",
                            line=dict(color="#fff", width=1.5)),
            )

        fig.update_layout(**_CHART, title="Eval Reward Over Time",
                         xaxis_title="Timestamp", yaxis_title="Mean Reward")
        st.plotly_chart(fig, key="agent_reward", width="stretch")

    _reward_chart()
    st.divider()

    # ── Action ledger ─────────────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _ledger() -> None:
        from osrl.dashboard.shared import render_action_ledger
        st.subheader("Action Ledger")
        render_action_ledger(conn)

    _ledger()
    st.divider()

    # ── Training log expander ─────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _train_log() -> None:
        evs = [dict(r) for r in conn.execute(
            "SELECT ts, kind, payload FROM events "
            "WHERE kind IN ('rl_train_start','rl_train_step','rl_train_done') "
            "ORDER BY id DESC LIMIT 20"
        ).fetchall()]
        if not evs:
            return

        with st.expander("📊 Training Event Log", expanded=False):
            _C = {"rl_train_start": "#38bdf8", "rl_train_step": "#818cf8", "rl_train_done": "#22c55e"}
            _I = {"rl_train_start": "🚀", "rl_train_step": "📈", "rl_train_done": "🏁"}
            for ev in evs:
                ts_s = dt.datetime.fromtimestamp(float(ev["ts"])).strftime("%H:%M:%S")
                k    = ev["kind"]
                try: p = json.loads(ev.get("payload") or "{}")
                except Exception: p = {}  # noqa: BLE001
                detail = (f"steps={p.get('steps_done',0):,}  reward={p.get('eval_mean_reward',0):.4f}"
                          if k == "rl_train_step" else str(p)[:80])
                st.markdown(
                    f"<div style='display:flex;gap:8px;padding:5px 10px;margin:2px 0;"
                    f"border-radius:6px;background:rgba(8,15,31,0.7);border-left:2px solid {_C.get(k,'#475569')}'>"
                    f"<code style='color:#94a3b8;font-size:10px;font-family:JetBrains Mono,monospace'>{ts_s}</code>"
                    f"<span>{_I.get(k,'•')}</span>"
                    f"<span style='color:{_C.get(k,'#64748b')};font-size:11px;font-weight:600'>{k}</span>"
                    f"<span style='color:#94a3b8;font-size:11px'>{detail}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    _train_log()
