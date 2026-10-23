"""Premium Processes page — live process table, RSS bar, action ledger, event stream."""

from __future__ import annotations

import datetime as dt
import json

import plotly.graph_objects as go
import streamlit as st

from osrl.dashboard.shared import get_conn

_BG, _PLOT = "rgba(0,0,0,0)", "rgba(6,13,26,0.95)"
_FONT = dict(color="#94a3b8", size=11, family="Inter")
_GRID = "rgba(56,189,248,0.05)"
_CHART = dict(
    paper_bgcolor=_BG, plot_bgcolor=_PLOT, font=_FONT,
    margin=dict(t=44, b=32, l=80, r=24), height=300,
    xaxis=dict(gridcolor=_GRID, showline=False, zeroline=False),
    yaxis=dict(gridcolor=_GRID, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="#0f1f3d", font_color="#e2e8f0",
                    bordercolor="rgba(56,189,248,0.4)"),
)


def _chart(**overrides) -> dict:
    """Return _CHART merged with overrides — avoids duplicate keyword argument errors."""
    return {**_CHART, **overrides}


def render() -> None:
    st.title("⚙️ Processes")
    conn    = get_conn()
    refresh = st.session_state.get("refresh_s", 10)

    # ── Top-K live table ──────────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _top_procs() -> None:
        st.subheader("🔝 Top Processes — Live")

        sample = conn.execute(
            "SELECT top_procs_json FROM samples ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if not sample or not sample[0]:
            st.info("No process data yet.")
            return

        try:
            procs = json.loads(sample[0])
        except Exception:  # noqa: BLE001
            procs = []

        if not procs:
            st.info("Process list empty.")
            return

        st.dataframe([{
            "Process": p.get("name", "?"), "PID": p.get("pid"),
            "CPU % (machine share)": p.get("cpu_pct"), "RSS MiB": p.get("rss_mib"),
            "Started": p.get("start_time"),
        } for p in procs], hide_index=True, width="stretch")
        st.caption("Recorded top CPU processes; first-tick ranking uses RSS. Missing CPU readings stay empty.")

    _top_procs()
    st.divider()

    # ── Top-10 RSS bar chart ──────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _rss_chart() -> None:
        st.subheader("🧠 Top-10 Memory Consumers")

        sample = conn.execute(
            "SELECT top_procs_json FROM samples ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not sample or not sample[0]:
            return
        try:
            procs = sorted(
                json.loads(sample[0]),
                key=lambda p: float(p.get("rss_bytes") or 0) or float(p.get("rss_mib") or 0),
                reverse=True
            )[:10]
        except Exception:  # noqa: BLE001
            return

        names = [str(p.get("name", "?"))[:18] for p in procs]
        rss   = [float(p.get("rss_bytes") or 0) / 1024**2 if p.get("rss_bytes") else float(p.get("rss_mib") or 0) for p in procs]
        cpu   = [float(p.get("cpu_pct") or 0) for p in procs]

        # Dual axis: RSS (bars) + CPU (line)
        fig = go.Figure()
        fig.add_bar(
            x=names, y=rss, name="RSS MiB",
            marker=dict(
                color=rss,
                colorscale=[[0, "rgba(56,189,248,0.3)"], [0.5, "rgba(129,140,248,0.6)"],
                            [1, "rgba(239,68,68,0.8)"]],
                line=dict(width=0),
            ),
            text=[f"{v:.0f}" for v in rss],
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        )
        fig.add_scatter(
            x=names, y=cpu, mode="lines+markers", name="CPU %",
            line=dict(color="#fbbf24", width=2, dash="dot"),
            marker=dict(color="#fbbf24", size=7),
            yaxis="y2",
        )
        fig.update_layout(**_chart(
            height=300,
            title="Memory Usage by Process (RSS MiB) + CPU %",
            yaxis_title="RSS MiB",
            yaxis2=dict(overlaying="y", side="right", title="CPU %",
                        color="#fbbf24", gridcolor="rgba(0,0,0,0)"),
            xaxis=dict(tickangle=-25, gridcolor=_GRID, tickfont=dict(size=10)),
        ))
        st.plotly_chart(fig, key="rss_chart", width="stretch")

    _rss_chart()
    st.divider()

    # ── CPU distribution across cores ────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _percpu_chart() -> None:
        st.subheader("💻 Per-Core CPU Distribution")
        sample = conn.execute(
            "SELECT cpu_percpu_json FROM samples ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if not sample or not sample[0]:
            st.caption("No per-core data.")
            return
        try:
            per_cpu = json.loads(sample[0])
        except Exception:  # noqa: BLE001
            return

        labels = [f"Core {i}" for i in range(len(per_cpu))]
        colors = [
            "#64748b" if v is None else "#ef4444" if v > 80 else "#f97316" if v > 50 else "#38bdf8" if v > 20 else "#1e3a5f"
            for v in per_cpu
        ]
        fig = go.Figure(go.Bar(
            x=labels, y=per_cpu, marker_color=colors,
            text=[f"{v:.0f}%" if v is not None else "Unknown" for v in per_cpu],
            textposition="outside", textfont=dict(size=10, color="#94a3b8"),
        ))
        known = [v for v in per_cpu if v is not None]
        avg = sum(known) / len(known) if known else 0
        fig.add_hline(y=avg, line_color="rgba(56,189,248,0.5)", line_dash="dot",
                     annotation_text=f"avg {avg:.1f}%", line_width=1.5)
        fig.update_layout(**_chart(height=240, title="Per-Core CPU Utilisation %",
                                   yaxis_range=[0, 110]))
        st.plotly_chart(fig, key="percpu", width="stretch")

    _percpu_chart()
    st.divider()

    # ── Process event stream ──────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _proc_events() -> None:
        st.subheader("🔔 Process Event Stream")
        st.caption("Process births, deaths, and risk flags from the daemon")

        evs = [dict(r) for r in conn.execute(
            "SELECT ts, kind, payload FROM events "
            "WHERE kind IN ('proc_birth','proc_death','process_flagged','psi_trigger_fired','risk_change') "
            "ORDER BY id DESC LIMIT 25"
        ).fetchall()]

        if not evs:
            st.markdown("""
            <div style='background:rgba(8,15,31,0.6);border:1px dashed rgba(56,189,248,0.15);
                        border-radius:10px;padding:20px;text-align:center;color:#94a3b8;
                        font-size:13px'>
                No process events yet — events appear as the daemon detects activity.
            </div>""", unsafe_allow_html=True)
            return

        _STYLE = {
            "proc_birth":       ("#22c55e", "🟢", "BIRTH"),
            "proc_death":       ("#ef4444", "🔴", "DEATH"),
            "process_flagged":  ("#f43f5e", "🚨", "FLAGGED"),
            "psi_trigger_fired":("#f97316", "🔥", "PSI SPIKE"),
            "risk_change":      ("#f59e0b", "⚠️", "RISK CHANGE"),
        }

        for ev in evs:
            ts_s   = dt.datetime.fromtimestamp(float(ev["ts"])).strftime("%H:%M:%S")
            kind   = ev.get("kind", "?")
            color, icon, badge = _STYLE.get(kind, ("#64748b", "•", kind[:10].upper()))
            try: p = json.loads(ev.get("payload") or "{}")
            except Exception: p = {}  # noqa: BLE001

            detail = (p.get("name") or p.get("new_level") or
                     p.get("risk_level") or str(p)[:60])
            pid_str = f"PID {p.get('pid')}" if p.get("pid") else ""

            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:center;padding:7px 12px;"
                f"margin:3px 0;border-radius:8px;"
                f"background:linear-gradient(135deg,rgba(10,20,45,0.8),rgba(15,30,60,0.5));"
                f"border:1px solid {color}18;border-left:3px solid {color};"
                f"animation:slide-in-up 0.3s ease-out'>"
                f"<code style='color:#94a3b8;font-size:10px;white-space:nowrap;"
                f"font-family:JetBrains Mono,monospace'>{ts_s}</code>"
                f"<span style='background:{color}22;color:{color};font-size:9px;"
                f"font-weight:700;letter-spacing:0.5px;padding:2px 7px;"
                f"border-radius:4px;white-space:nowrap'>{badge}</span>"
                f"<span style='font-size:13px'>{icon}</span>"
                f"<span style='color:#94a3b8;font-size:12px;font-weight:500'>{detail}</span>"
                f"<span style='color:#94a3b8;font-size:11px;margin-left:auto;"
                f"font-family:JetBrains Mono,monospace'>{pid_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    _proc_events()
    st.divider()

    # ── Action ledger ─────────────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _action_ledger() -> None:
        from osrl.dashboard.shared import render_action_ledger
        st.subheader("Action Ledger")
        render_action_ledger(conn)

    _action_ledger()
