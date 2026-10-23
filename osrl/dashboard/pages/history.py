"""Premium History page — 7-day trends, heatmaps, rollup charts, event log."""

from __future__ import annotations

import datetime as dt
import json

import plotly.graph_objects as go
import streamlit as st

from osrl.dashboard.shared import ensure_rollups, get_conn, recent_rollups

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


def _chart(**overrides) -> dict:
    """Return _CHART merged with overrides — avoids duplicate keyword argument errors."""
    return {**_CHART, **overrides}


def render() -> None:
    st.title("📅 History & Trends")
    conn    = get_conn()
    refresh = st.session_state.get("refresh_s", 10)

    # Ensure rollups are computed
    try:
        ensure_rollups(conn)
    except Exception:  # noqa: BLE001
        pass

    # ── Rollup summary banner ─────────────────────────────────────────────────
    @st.fragment(run_every=refresh * 3)
    def _summary() -> None:
        try:
            n_roll = conn.execute("SELECT COUNT(*) FROM rollups").fetchone()[0]
            n_samp = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            bounds = conn.execute("SELECT MIN(window_start), MAX(window_end) FROM rollups").fetchone()
            span_h = 0.0
            if bounds[0] and bounds[1]:
                span_h = (float(bounds[1]) - float(bounds[0])) / 3600
        except Exception:  # noqa: BLE001
            n_roll, n_samp, span_h = 0, 0, 0.0

        if n_roll == 0:
            st.markdown("""
            <div style='background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.25);
                        border-radius:12px;padding:16px 20px;animation:slide-in-down 0.4s ease-out'>
                <span style='color:#f97316;font-weight:700'>⏳ Computing Rollups…</span>
                <span style='color:#64748b;margin-left:8px;font-size:13px'>
                    15-minute windows are computed every 15 minutes. Check back soon.
                </span>
            </div>""", unsafe_allow_html=True)
            return

        st.markdown(f"""
        <div class='status-grid' style='
                    margin-bottom:4px;animation:fade-in 0.5s ease-out'>
            <div style='background:rgba(10,20,45,0.9);border:1px solid rgba(56,189,248,0.15);
                        border-radius:12px;padding:14px;text-align:center'>
                <div style='color:#94a3b8;font-size:9px;text-transform:uppercase;letter-spacing:1px'>15-min Windows</div>
                <div style='color:#38bdf8;font-size:22px;font-weight:800;
                            font-family:JetBrains Mono,monospace'>{n_roll}</div>
            </div>
            <div style='background:rgba(10,20,45,0.9);border:1px solid rgba(129,140,248,0.15);
                        border-radius:12px;padding:14px;text-align:center'>
                <div style='color:#94a3b8;font-size:9px;text-transform:uppercase;letter-spacing:1px'>Total Samples</div>
                <div style='color:#818cf8;font-size:22px;font-weight:800;
                            font-family:JetBrains Mono,monospace'>{n_samp:,}</div>
            </div>
            <div style='background:rgba(10,20,45,0.9);border:1px solid rgba(34,197,94,0.15);
                        border-radius:12px;padding:14px;text-align:center'>
                <div style='color:#94a3b8;font-size:9px;text-transform:uppercase;letter-spacing:1px'>Data Span</div>
                <div style='color:#22c55e;font-size:22px;font-weight:800;
                            font-family:JetBrains Mono,monospace'>{span_h:.1f}h</div>
            </div>
            <div style='background:rgba(10,20,45,0.9);border:1px solid rgba(251,146,60,0.15);
                        border-radius:12px;padding:14px;text-align:center'>
                <div style='color:#94a3b8;font-size:9px;text-transform:uppercase;letter-spacing:1px'>Avg/Window</div>
                <div style='color:#fb923c;font-size:22px;font-weight:800;
                            font-family:JetBrains Mono,monospace'>{n_samp//max(1,n_roll)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    _summary()
    st.divider()

    # ── 7-day trend charts ────────────────────────────────────────────────────
    @st.fragment(run_every=refresh * 3)
    def _trends() -> None:
        st.subheader("📈 7-Day Trend Charts")

        rolls = recent_rollups(conn, limit=700)
        if not rolls:
            st.info("No rollup data yet. Rollups are computed every 15 minutes.")
            return

        def _ts(r): return dt.datetime.fromtimestamp(float(r["window_start"])).strftime("%m/%d %H:%M")
        xs  = [_ts(r) for r in rolls]

        tabs = st.tabs(["CPU & Memory", "Swap Pressure", "PSI Stall", "IO & Alloc Stall"])

        # ── CPU & Memory ───────────────────────────────────────────────────────
        with tabs[0]:
            cpu_avg = [r.get("cpu_total_pct_avg") for r in rolls]
            cpu_max = [r.get("cpu_total_pct_max") for r in rolls]
            mem_avg = [r.get("mem_used_pct_avg")  for r in rolls]

            fig = go.Figure()
            fig.add_scatter(x=xs, y=cpu_max, mode="lines", name="CPU Max %",
                           line=dict(color="rgba(239,68,68,0.4)", width=1, dash="dot"))
            fig.add_scatter(x=xs, y=cpu_avg, mode="lines", name="CPU Avg %",
                           line=dict(color="#38bdf8", width=2),
                           fill="tozeroy", fillcolor="rgba(56,189,248,0.05)")
            fig.add_scatter(x=xs, y=mem_avg, mode="lines", name="RAM Used % (avg)",
                           line=dict(color="#22c55e", width=1.5, dash="dot"), yaxis="y2")
            fig.update_layout(
                **_CHART, title="CPU & RAM — 15-min Averages",
                yaxis2=dict(overlaying="y", side="right", title="RAM %",
                            color="#22c55e", gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, key="h_cpu", width="stretch")

        # ── Swap ───────────────────────────────────────────────────────────────
        with tabs[1]:
            swap_avg = [r.get("swap_used_pct_avg") for r in rolls]
            swap_max = [r.get("swap_used_pct_max") for r in rolls]
            swp_out  = [r.get("swp_out_ps_avg")    for r in rolls]

            fig = go.Figure()
            fig.add_scatter(x=xs, y=swap_max, name="Swap Max %",
                           line=dict(color="rgba(249,115,22,0.4)", dash="dot", width=1), mode="lines")
            fig.add_scatter(x=xs, y=swap_avg, name="Swap Avg %",
                           line=dict(color="#fb923c", width=2), mode="lines",
                           fill="tozeroy", fillcolor="rgba(251,146,60,0.06)")
            fig.add_scatter(x=xs, y=swp_out, name="Swap-Out avg pg/s",
                           line=dict(color="#f43f5e", width=1.5, dash="dashdot"),
                           mode="lines", yaxis="y2")
            fig.add_hline(y=80, line_color="rgba(239,68,68,0.4)", line_dash="dot", line_width=1)
            fig.update_layout(
                **_CHART, title="Swap Pressure — 15-min Averages",
                yaxis2=dict(overlaying="y", side="right", title="pg/s",
                            color="#f43f5e", gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, key="h_swap", width="stretch")

        # ── PSI ────────────────────────────────────────────────────────────────
        with tabs[2]:
            psi_mem = [r.get("psi_mem_stall_s_sum") for r in rolls]
            psi_io  = [r.get("psi_io_stall_s_sum")  for r in rolls]

            fig = go.Figure()
            fig.add_bar(x=xs, y=psi_mem, name="Mem Stall s/window",
                       marker_color="rgba(167,139,250,0.7)")
            fig.add_scatter(x=xs, y=psi_io, name="IO Stall s",
                           line=dict(color="#f43f5e", width=1.5), mode="lines", yaxis="y2")
            fig.update_layout(
                **_CHART, title="PSI Stall Seconds per 15-min Window",
                yaxis_title="Mem stall s",
                yaxis2=dict(overlaying="y", side="right", title="IO stall s",
                            color="#f43f5e", gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, key="h_psi", width="stretch")

        # ── IO / Alloc stall ───────────────────────────────────────────────────
        with tabs[3]:
            # Derive from raw samples (15-min windows)
            raw = conn.execute(
                "SELECT CAST(ts/900 AS INT)*900 AS bucket, "
                "AVG(majflt_ps) as mf, AVG(allocstall_ps) as al, AVG(reclaim_scan_ps) as rs "
                "FROM samples GROUP BY bucket ORDER BY bucket"
            ).fetchall()
            if raw:
                bts  = [dt.datetime.fromtimestamp(float(r[0])).strftime("%m/%d %H:%M") for r in raw]
                mf   = [r[1] for r in raw]
                al   = [r[2] for r in raw]
                rs   = [r[3] for r in raw]
                fig  = go.Figure()
                fig.add_scatter(x=bts, y=mf, name="Maj Faults/s", mode="lines",
                               line=dict(color="#f97316", width=2),
                               fill="tozeroy", fillcolor="rgba(249,115,22,0.05)")
                fig.add_scatter(x=bts, y=al, name="Alloc Stalls/s", mode="lines",
                               line=dict(color="#ef4444", width=1.5, dash="dot"), yaxis="y2")
                fig.add_scatter(x=bts, y=rs, name="Reclaim Scans/s", mode="lines",
                               line=dict(color="#818cf8", width=1.5, dash="dashdot"), yaxis="y2")
                fig.update_layout(
                    **_CHART, title="Memory Pressure — Faults, Alloc Stalls & Reclaim",
                    yaxis_title="Maj Faults/s",
                    yaxis2=dict(overlaying="y", side="right", title="per s",
                                color="#ef4444", gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, key="h_io", width="stretch")

    _trends()
    st.divider()

    # ── Heatmap: CPU by hour-of-day ───────────────────────────────────────────
    @st.fragment(run_every=refresh * 5)
    def _heatmaps() -> None:
        st.subheader("🔥 CPU Heatmap — Hour of Day vs Day of Week")

        raw = conn.execute(
            "SELECT ts, cpu_total_pct FROM samples WHERE cpu_total_pct IS NOT NULL ORDER BY ts"
        ).fetchall()
        if len(raw) < 20:
            st.info("Need more data for heatmap. Keep the collector running.")
            return

        # Build 7×24 matrix
        import numpy as np
        matrix = np.full((7, 24), np.nan)
        counts = np.zeros((7, 24))
        for row in raw:
            d = dt.datetime.fromtimestamp(float(row[0]))
            matrix[d.weekday(), d.hour] = (
                float(row[1]) if np.isnan(matrix[d.weekday(), d.hour])
                else (matrix[d.weekday(), d.hour] * counts[d.weekday(), d.hour] + float(row[1]))
                     / (counts[d.weekday(), d.hour] + 1)
            )
            counts[d.weekday(), d.hour] += 1

        days  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        hours = [f"{h:02d}:00" for h in range(24)]

        # Fill NaN with 0
        # Missing hour buckets remain gaps, never idle measurements.

        fig = go.Figure(go.Heatmap(
            z=matrix, x=hours, y=days,
            colorscale=[
                [0.0,  "rgba(6,13,26,1)"],
                [0.3,  "rgba(29,78,216,0.8)"],
                [0.6,  "rgba(249,115,22,0.9)"],
                [1.0,  "rgba(239,68,68,1)"],
            ],
            zmid=40,
            text=[[f"{matrix[i][j]:.0f}%" if np.isfinite(matrix[i][j]) else "" for j in range(24)] for i in range(7)],
            texttemplate="%{text}", textfont=dict(size=9, color="rgba(255,255,255,0.7)"),
            colorbar=dict(thickness=12, tickfont=dict(size=10, color="#94a3b8"),
                         title=dict(text="CPU %", font=dict(size=10, color="#94a3b8"))),
            hoverongaps=False,
        ))
        fig.update_layout(
            height=280, margin=dict(t=44, b=60, l=50, r=20),
            paper_bgcolor=_BG, font=_FONT,
            title="Average CPU % — Hour of Day × Day of Week",
            xaxis=dict(tickangle=-35, tickfont=dict(size=9, color="#94a3b8")),
            yaxis=dict(tickfont=dict(size=10, color="#94a3b8")),
        )
        st.plotly_chart(fig, key="heatmap_cpu", width="stretch")

    _heatmaps()
    st.divider()

    # ── Live event log ────────────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _event_log() -> None:
        st.subheader("📋 Full Event Log")

        # Filters
        cf1, cf2, cf3 = st.columns([2, 2, 1])
        all_kinds = [r[0] for r in conn.execute(
            "SELECT DISTINCT kind FROM events ORDER BY kind"
        ).fetchall()]
        kind_filter = cf1.multiselect("Filter by kind", all_kinds,
                                      key="hist_kind_filter",
                                      placeholder="All event types")
        limit = cf2.select_slider("Rows", [25, 50, 100, 200, 500], value=50,
                                  key="hist_limit")
        cf3.markdown("<br>", unsafe_allow_html=True)
        cf3.markdown(f"<div style='color:#94a3b8;font-size:12px;padding-top:8px'>{len(all_kinds)} kinds</div>",
                    unsafe_allow_html=True)

        if kind_filter:
            placeholders = ",".join(["?" for _ in kind_filter])
            evs = [dict(r) for r in conn.execute(
                f"SELECT ts, kind, payload FROM events WHERE kind IN ({placeholders}) "
                f"ORDER BY id DESC LIMIT ?", (*kind_filter, limit)
            ).fetchall()]
        else:
            evs = [dict(r) for r in conn.execute(
                "SELECT ts, kind, payload FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()]

        if not evs:
            st.info("No events match the filter.")
            return

        # Frequency chart
        from collections import Counter
        kind_counts = Counter(e["kind"] for e in evs)
        if len(kind_counts) > 1:
            kc_sorted = sorted(kind_counts.items(), key=lambda x: x[1], reverse=True)
            fig = go.Figure(go.Bar(
                x=[k[0] for k in kc_sorted],
                y=[k[1] for k in kc_sorted],
                marker_color="#38bdf8", opacity=0.8,
                text=[str(k[1]) for k in kc_sorted],
                textposition="outside", textfont=dict(size=10, color="#94a3b8"),
            ))
            fig.update_layout(**_chart(
                height=200,
                title=f"Event Frequency (showing {len(evs)} events)",
                margin=dict(t=40, b=50, l=40, r=16),
                xaxis=dict(tickangle=-25, gridcolor=_GRID, tickfont=dict(size=10)),
            ))
            st.plotly_chart(fig, key="ev_freq", width="stretch")

        # Event rows
        _COLORS = {
            "psi_trigger_fired":      ("#ef4444", "🔥"),
            "psi_trigger_registered": ("#f97316", "🔔"),
            "risk_change":            ("#f59e0b", "⚠️"),
            "proc_birth":             ("#22c55e", "🟢"),
            "proc_death":             ("#ef4444", "🔴"),
            "process_flagged":        ("#f43f5e", "🚨"),
            "rl_train_start":         ("#38bdf8", "🚀"),
            "rl_train_step":          ("#818cf8", "📈"),
            "rl_train_done":          ("#22c55e", "🏁"),
            "auto_train_triggered":   ("#34d399", "⚡"),
            "auto_train_status":      ("#1e3a5f", "📊"),
        }

        for ev in evs:
            ts_s  = dt.datetime.fromtimestamp(float(ev["ts"])).strftime("%Y-%m-%d %H:%M:%S")
            kind  = ev.get("kind", "?")
            color, icon = _COLORS.get(kind, ("#475569", "•"))
            try: p = json.loads(ev.get("payload") or "{}")
            except Exception: p = {}  # noqa: BLE001

            if kind == "rl_train_step":
                detail = f"step={p.get('steps_done',0):,}  reward={p.get('eval_mean_reward',0):.4f}  validated={p.get('validated')}"
            elif kind == "rl_train_done":
                detail = f"best_reward={p.get('best_reward',0):.4f}"
            elif kind == "auto_train_status":
                detail = f"new={p.get('new_samples',0)} needed={p.get('needed_samples',0)} wait={p.get('next_run_in_s',0)}s"
            else:
                detail = (p.get("name") or p.get("new_level") or p.get("risk_level") or str(p)[:60])

            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:center;padding:6px 12px;"
                f"margin:2px 0;border-radius:7px;background:rgba(8,15,31,0.7);"
                f"border-left:3px solid {color};'>"
                f"<code style='color:#94a3b8;font-size:10px;white-space:nowrap;"
                f"font-family:JetBrains Mono,monospace'>{ts_s}</code>"
                f"<span style='font-size:13px'>{icon}</span>"
                f"<span style='color:{color};font-size:11px;font-weight:600;"
                f"white-space:nowrap'>{kind}</span>"
                f"<span style='color:#94a3b8;font-size:11px;flex:1'>{detail}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    _event_log()
