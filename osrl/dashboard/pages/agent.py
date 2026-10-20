"""Premium Agent page — mode control, policy status, reward history, action ledger."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from osrl.dashboard.shared import (
    agent_mode, best_checkpoint, find_checkpoints, get_conn,
    recent_events, safe_float, set_agent_mode,
)

_BG, _PLOT = "rgba(0,0,0,0)", "rgba(6,13,26,0.95)"
_FONT = dict(color="#64748b", size=11, family="Inter")
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
    "autopilot": ("#818cf8", "🤖", "Acts autonomously within guard constraints."),
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
    <div style='color:#475569;font-size:11px;margin-bottom:10px'>{desc}</div>
    {badge}
</div>"""


def render() -> None:
    st.title("🤖 RL Agent")
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
                <div style='color:#334155;font-size:9px;text-transform:uppercase;
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
                <div style='color:#475569'>No checkpoint yet — run training first.</div>
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
        st.subheader("📋 Action Ledger")
        st.caption("Recent actions taken by the RL agent")
        try:
            rows = [dict(r) for r in conn.execute(
                "SELECT ts, pid, name, action, reason FROM actions_ledger "
                "ORDER BY id DESC LIMIT 30"
            ).fetchall()]
        except Exception:  # noqa: BLE001
            rows = []

        if not rows:
            st.markdown("""
            <div style='background:rgba(8,15,31,0.6);border:1px dashed rgba(56,189,248,0.15);
                        border-radius:10px;padding:20px;text-align:center;color:#334155'>
                No agent actions yet. Switch to Autopilot mode to enable autonomous actions.
            </div>""", unsafe_allow_html=True)
            return

        _ACTION_COLOR = {
            "noop": "#334155", "alert": "#f59e0b",
            "renice_back": "#22c55e", "request_approval": "#818cf8",
            "tune_threshold": "#38bdf8",
        }
        for r in rows:
            ts_s   = dt.datetime.fromtimestamp(float(r["ts"])).strftime("%H:%M:%S")
            action = r.get("action", "noop")
            color  = _ACTION_COLOR.get(action, "#64748b")
            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:center;padding:7px 12px;"
                f"margin:2px 0;border-radius:8px;background:rgba(8,15,31,0.8);"
                f"border-left:3px solid {color};animation:slide-in-up 0.3s ease-out'>"
                f"<code style='color:#334155;font-size:10px;font-family:JetBrains Mono,monospace'>{ts_s}</code>"
                f"<span style='background:{color}22;color:{color};font-size:10px;"
                f"font-weight:700;letter-spacing:0.5px;padding:2px 8px;border-radius:4px'>{action.upper()}</span>"
                f"<span style='color:#64748b;font-size:12px'>PID {r.get('pid','?')} — {r.get('name','?')}</span>"
                f"<span style='color:#334155;font-size:11px;margin-left:auto'>{r.get('reason','')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

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
                    f"<code style='color:#334155;font-size:10px;font-family:JetBrains Mono,monospace'>{ts_s}</code>"
                    f"<span>{_I.get(k,'•')}</span>"
                    f"<span style='color:{_C.get(k,'#64748b')};font-size:11px;font-weight:600'>{k}</span>"
                    f"<span style='color:#475569;font-size:11px'>{detail}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    _train_log()
