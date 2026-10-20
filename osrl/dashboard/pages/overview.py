"""Overview page — animated status cards, live gauges, 6-tab charts, event ticker."""

from __future__ import annotations

import datetime as dt
import json
import math

import plotly.graph_objects as go
import streamlit as st

from osrl.dashboard.shared import (
    agent_mode, ago, get_conn, heartbeat_row,
    is_demo, latest_sample, recent_samples, safe_float,
)

# ── Shared chart theme ────────────────────────────────────────────────────────
_BG, _PLOT = "rgba(0,0,0,0)", "rgba(6,13,26,0.95)"
_FONT = dict(color="#64748b", size=11, family="Inter")
_GRID = "rgba(56,189,248,0.05)"
_CHART = dict(
    paper_bgcolor=_BG, plot_bgcolor=_PLOT, font=_FONT,
    margin=dict(t=44, b=32, l=52, r=24),
    height=280,
    xaxis=dict(gridcolor=_GRID, showline=False, zeroline=False, title=""),
    yaxis=dict(gridcolor=_GRID, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#0f1f3d", font_color="#e2e8f0", bordercolor="rgba(56,189,248,0.4)"),
)


def _sparkline(vals: list, color: str = "#38bdf8") -> go.Figure:
    """Tiny inline sparkline figure."""
    fig = go.Figure(go.Scatter(
        y=vals, mode="lines",
        line=dict(color=color, width=1.5),
        fill="tozeroy", fillcolor=color.replace(")", ",0.08)").replace("rgb", "rgba") if "rgb" in color else f"rgba(56,189,248,0.06)",
    ))
    fig.update_layout(
        height=60, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def _gauge(label: str, pct: float, warn: float = 20.0, unit: str = "%") -> go.Figure:
    """Animated semicircle gauge."""
    if pct > warn * 2:
        bar_color, glow = "#ef4444", "rgba(239,68,68,0.3)"
    elif pct > warn:
        bar_color, glow = "#f97316", "rgba(249,115,22,0.2)"
    else:
        bar_color, glow = "#22c55e", "rgba(34,197,94,0.2)"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        delta={"reference": warn, "relative": False,
               "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
        title={"text": label, "font": {"size": 12, "color": "#475569", "family": "Inter"}},
        number={"suffix": unit, "font": {"size": 24, "color": "#f1f5f9", "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155",
                     "tickfont": {"size": 9}, "nticks": 6},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, warn],       "color": "rgba(34,197,94,0.06)"},
                {"range": [warn, warn*2],  "color": "rgba(249,115,22,0.06)"},
                {"range": [warn*2, 100],   "color": "rgba(239,68,68,0.06)"},
            ],
            "threshold": {
                "line": {"color": "#f97316", "width": 2},
                "thickness": 0.75, "value": warn,
            },
        },
    ))
    fig.update_layout(
        height=210, margin=dict(t=44, b=8, l=16, r=16),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _metric_card(label: str, value: str, sub: str = "", color: str = "#38bdf8",
                 icon: str = "", alert: bool = False) -> str:
    border = "rgba(239,68,68,0.5)" if alert else f"{color}33"
    glow   = "0 0 20px rgba(239,68,68,0.15)" if alert else f"0 0 20px {color}15"
    return f"""
<div style='background:linear-gradient(135deg,rgba(10,20,45,0.95),rgba(15,30,65,0.85));
            border:1px solid {border};border-radius:14px;padding:16px 18px;
            backdrop-filter:blur(12px);transition:all 0.3s ease;
            box-shadow:{glow};animation:slide-in-up 0.4s ease-out;
            position:relative;overflow:hidden;'>
    <div style='position:absolute;top:0;right:0;width:60px;height:60px;
                background:radial-gradient(circle at center,{color}18,transparent);
                border-radius:0 14px 0 100%;'></div>
    <div style='font-size:10px;color:#334155;text-transform:uppercase;
                letter-spacing:1.5px;font-weight:700;margin-bottom:6px'>{icon} {label}</div>
    <div style='font-size:22px;font-weight:800;color:#f1f5f9;
                font-family:"JetBrains Mono",monospace;letter-spacing:-0.5px'>{value}</div>
    {f"<div style='font-size:11px;color:#475569;margin-top:3px'>{sub}</div>" if sub else ""}
</div>"""


def render() -> None:
    st.title("System Overview")
    conn    = get_conn()
    refresh = st.session_state.get("refresh_s", 10)

    # ── Animated status bar ───────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _status_bar() -> None:
        import time
        hb   = heartbeat_row(conn)
        s    = latest_sample(conn)
        mode = agent_mode(conn)

        if hb is None:
            st.markdown("""
            <div style='background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);
                        border-radius:12px;padding:14px 18px;animation:slide-in-down 0.4s ease-out'>
                <span style='color:#ef4444;font-weight:700'>🔴 Daemon Offline</span>
                <span style='color:#64748b;margin-left:8px;font-size:13px'>
                    Run: <code>python -m osrl.daemon --db ~/os-rl-agent/data/telemetry.db</code>
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            staleness = time.time() - float(hb["ts"])
            try:
                payload = json.loads(hb.get("payload") or "{}")
            except Exception:  # noqa: BLE001
                payload = {}

            if staleness > 60 and not is_demo():
                st.warning(f"⚠️ Heartbeat stale ({ago(hb['ts'])})")
            else:
                n_samples = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
                n_events  = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

                # Animated top banner
                risk  = (payload.get("risk_level") or "none").upper()
                r_col = {"NONE": "#22c55e", "ELEVATED": "#f59e0b", "CRITICAL": "#ef4444"}.get(risk, "#64748b")
                m_col = {"observe": "#38bdf8", "advise": "#f59e0b", "autopilot": "#818cf8"}.get(mode, "#64748b")
                if is_demo():
                    source_card = _metric_card("Dataset", "📼 Recorded", ago(hb["ts"]), "#38bdf8", "●")
                else:
                    source_card = _metric_card("Daemon", "🟢 Live", ago(hb["ts"]), "#22c55e", "●")

                st.markdown(f"""
                <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:10px;
                            margin-bottom:4px;animation:fade-in 0.5s ease-out'>
                    {source_card}
                    {_metric_card("Risk Level", risk, "", r_col, "⚡")}
                    {_metric_card("Agent Mode", mode.upper(), "", m_col, "🤖")}
                    {_metric_card("Samples", f"{n_samples:,}", "total collected", "#38bdf8", "📦")}
                    {_metric_card("Events", f"{n_events:,}", "total logged", "#818cf8", "📋")}
                </div>
                """, unsafe_allow_html=True)

        if s is None:
            st.info("⏳ No samples yet — start the collector daemon.")
            return

        st.divider()

        # ── Key system metrics ────────────────────────────────────────────────
        cpu       = safe_float(s, "cpu_total_pct")
        mem_avail = safe_float(s, "mem_available_bytes") / 1024**3
        mem_pct   = safe_float(s, "mem_used_pct")
        swap_pct  = safe_float(s, "swap_used_pct")
        swp_out   = safe_float(s, "swp_out_ps")
        gpu_util  = safe_float(s, "gpu_util_pct")
        gpu_mem   = safe_float(s, "gpu_mem_used_mib")
        gpu_temp  = safe_float(s, "gpu_temp_c")
        load1     = safe_float(s, "load1")

        cols = st.columns(8)
        metrics = [
            ("CPU",       f"{cpu:.1f}%",       "",            "#38bdf8", "💻", cpu > 80),
            ("RAM Avail", f"{mem_avail:.1f}GB", f"Used {mem_pct:.0f}%",  "#22c55e", "🧠", mem_avail < 0.5),
            ("RAM Used",  f"{mem_pct:.0f}%",   "",            "#a78bfa", "📊", mem_pct > 85),
            ("Swap",      f"{swap_pct:.0f}%",  "",            "#fb923c", "💾", swap_pct > 80),
            ("Swap-Out",  f"{swp_out:.0f}",    "pages/s",     "#f43f5e", "⚠️", swp_out > 500),
            ("GPU Util",  f"{gpu_util:.0f}%",  "",            "#818cf8", "🎮", gpu_util > 95),
            ("VRAM",      f"{gpu_mem:.0f}MiB", "",            "#c084fc", "🔮", False),
            ("GPU Temp",  f"{gpu_temp:.0f}°C", "",            "#f97316", "🌡️", gpu_temp > 85),
        ]
        for col, (lbl, val, sub, color, icon, alert) in zip(cols, metrics):
            col.markdown(_metric_card(lbl, val, sub, color, icon, alert), unsafe_allow_html=True)

        st.divider()

        # ── PSI Gauges ────────────────────────────────────────────────────────
        st.subheader("⚡ Pressure Stall Index — Live")
        st.caption("Real-time kernel PSI stall percentages — >20% indicates system under pressure")

        interval = safe_float(s, "interval_s", 2.0)
        def _stall(key: str) -> float:
            v = safe_float(s, key)
            return v / (interval * 1e6) * 100 if interval > 0 and v > 0 else 0.0

        cpu_stall = _stall("psi_cpu_some_stall_us")
        mem_stall = _stall("psi_mem_some_stall_us")
        io_stall  = _stall("psi_io_some_stall_us")

        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(_gauge("CPU Stall",    cpu_stall, warn=15.0), key="psi_cpu_g",  width="stretch")
        g2.plotly_chart(_gauge("Memory Stall", mem_stall, warn=20.0), key="psi_mem_g",  width="stretch")
        g3.plotly_chart(_gauge("IO Stall",     io_stall,  warn=25.0), key="psi_io_g",   width="stretch")

        # Additional GPU gauges if data exists
        if gpu_util > 0 or gpu_temp > 0:
            g4, g5, g6 = st.columns(3)
            g4.plotly_chart(_gauge("GPU Util",   gpu_util, warn=80.0), key="gpu_util_g",  width="stretch")
            g5.plotly_chart(_gauge("GPU Temp",   gpu_temp, warn=80.0, unit="°C"), key="gpu_temp_g",  width="stretch")
            g6.plotly_chart(_gauge("Swap Used",  swap_pct, warn=70.0), key="swap_g",  width="stretch")

    _status_bar()

    # ── 6-tab live charts ─────────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _charts() -> None:
        rows = recent_samples(conn, limit=180)
        if not rows:
            return

        ts      = [r["ts"] for r in rows]
        # Format timestamps for display
        ts_fmt  = [dt.datetime.fromtimestamp(float(t)).strftime("%H:%M:%S") for t in ts]

        def _col(key):   return [r.get(key) for r in rows]
        def _psi(key):
            return [
                (float(r[key]) / ((r.get("interval_s") or 2.0) * 1e6) * 100)
                if r.get(key) else None
                for r in rows
            ]

        cpu_v     = _col("cpu_total_pct")
        swap_v    = _col("swap_used_pct")
        mem_used  = _col("mem_used_pct")
        swp_out_v = _col("swp_out_ps")
        gpu_v     = _col("gpu_util_pct")
        gpu_mem_v = _col("gpu_mem_used_mib")
        gpu_pwr_v = _col("gpu_power_w")
        gpu_tmp_v = _col("gpu_temp_c")
        mem_avl   = [(r.get("mem_available_bytes") or 0) / 1024**3 for r in rows]
        load1_v   = _col("load1")
        load5_v   = _col("load5")
        load15_v  = _col("load15")
        psi_cpu   = _psi("psi_cpu_some_stall_us")
        psi_mem   = _psi("psi_mem_some_stall_us")
        psi_io    = _psi("psi_io_some_stall_us")

        st.subheader("📈 Live Telemetry — Last 5 Minutes")
        tabs = st.tabs(["💻 CPU & GPU", "💾 Memory & Swap", "⚡ PSI All", "⚖️ Load Avg", "🎮 GPU Detail", "🧠 Memory Detail"])

        def _line(fig, x, y, name, color, dash="solid", width=2, yaxis=None, fill=False):
            kw = dict(x=x, y=y, mode="lines", name=name,
                      line=dict(color=color, width=width, dash=dash))
            if yaxis: kw["yaxis"] = yaxis
            if fill:
                kw["fill"] = "tozeroy"
                kw["fillcolor"] = color.replace("rgb", "rgba").replace(")", ",0.06)") if "rgb" in color else f"rgba(56,189,248,0.06)"
            fig.add_scatter(**kw)

        # ── Tab 1: CPU & GPU ──────────────────────────────────────────────────
        with tabs[0]:
            fig = go.Figure()
            _line(fig, ts_fmt, cpu_v,  "CPU %",   "#38bdf8", fill=True)
            if any(gpu_v): _line(fig, ts_fmt, gpu_v, "GPU %", "#818cf8", dash="dot")
            fig.add_hline(y=80, line_color="rgba(239,68,68,0.5)", line_dash="dot",
                         line_width=1, annotation_text="80% warn", annotation_font_color="#ef4444")
            fig.update_layout(**_CHART, title="CPU & GPU Utilisation %", yaxis_range=[0, 105])
            st.plotly_chart(fig, key="c_cpu", width="stretch")

        # ── Tab 2: Memory & Swap ──────────────────────────────────────────────
        with tabs[1]:
            fig = go.Figure()
            _line(fig, ts_fmt, swap_v,    "Swap %",      "#fb923c", fill=True)
            _line(fig, ts_fmt, mem_used,  "RAM Used %",  "#22c55e", dash="dot")
            _line(fig, ts_fmt, swp_out_v, "Swap-Out pg/s","#f43f5e", dash="dashdot", yaxis="y2")
            fig.update_layout(**_CHART, title="Memory & Swap Pressure",
                              yaxis2=dict(overlaying="y", side="right", title="pg/s",
                                          color="#f43f5e", gridcolor="rgba(0,0,0,0)"))
            fig.add_hline(y=80, line_color="rgba(249,115,22,0.4)", line_dash="dot", line_width=1)
            st.plotly_chart(fig, key="c_swap", width="stretch")

        # ── Tab 3: All PSI ────────────────────────────────────────────────────
        with tabs[2]:
            fig = go.Figure()
            _line(fig, ts_fmt, psi_cpu, "CPU Stall %", "#38bdf8")
            _line(fig, ts_fmt, psi_mem, "Mem Stall %", "#a78bfa", fill=True)
            _line(fig, ts_fmt, psi_io,  "IO Stall %",  "#f43f5e")
            fig.add_hline(y=20, line_color="rgba(249,115,22,0.5)", line_dash="dot",
                         line_width=1, annotation_text="warn 20%", annotation_font_color="#f97316")
            fig.update_layout(**_CHART, title="PSI Stall % — CPU / Memory / IO")
            st.plotly_chart(fig, key="c_psi", width="stretch")

        # ── Tab 4: Load Average ───────────────────────────────────────────────
        with tabs[3]:
            ncpu = rows[-1].get("ncpu") or 1
            fig  = go.Figure()
            _line(fig, ts_fmt, load1_v,  "1 min",  "#38bdf8", fill=True)
            _line(fig, ts_fmt, load5_v,  "5 min",  "#818cf8")
            _line(fig, ts_fmt, load15_v, "15 min", "#475569", dash="dot")
            fig.add_hline(y=ncpu, line_color="rgba(239,68,68,0.5)", line_dash="dot",
                         annotation_text=f"nCPU={ncpu}", line_width=1)
            fig.update_layout(**_CHART, title="System Load Average")
            st.plotly_chart(fig, key="c_load", width="stretch")

        # ── Tab 5: GPU Detail (2 charts) ──────────────────────────────────────
        with tabs[4]:
            if not any(gpu_v):
                st.markdown("""
                <div style='text-align:center;padding:40px;color:#334155'>
                    <div style='font-size:3rem;margin-bottom:8px'>🎮</div>
                    <div>No GPU data — NVML not detected or no dedicated GPU</div>
                </div>""", unsafe_allow_html=True)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    fig = go.Figure()
                    _line(fig, ts_fmt, gpu_v,     "GPU Util %",  "#818cf8", fill=True)
                    _line(fig, ts_fmt, gpu_mem_v, "VRAM MiB",    "#c084fc", dash="dot", yaxis="y2")
                    fig.update_layout(**{**_CHART, "height": 260}, title="GPU Util & VRAM",
                                      yaxis2=dict(overlaying="y", side="right", title="MiB",
                                                  color="#c084fc", gridcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig, key="c_gpu_util", width="stretch")
                with c2:
                    fig = go.Figure()
                    _line(fig, ts_fmt, gpu_tmp_v, "Temp °C", "#f97316", fill=True)
                    _line(fig, ts_fmt, gpu_pwr_v, "Power W",  "#fbbf24", dash="dot", yaxis="y2")
                    fig.add_hline(y=85, line_color="rgba(239,68,68,0.5)", line_dash="dot", line_width=1)
                    fig.update_layout(**{**_CHART, "height": 260}, title="GPU Temp & Power",
                                      yaxis2=dict(overlaying="y", side="right", title="W",
                                                  color="#fbbf24", gridcolor="rgba(0,0,0,0)"))
                    st.plotly_chart(fig, key="c_gpu_temp", width="stretch")

        # ── Tab 6: Memory Detail ──────────────────────────────────────────────
        with tabs[5]:
            total_gib = (rows[-1].get("mem_total_bytes") or 0) / 1024**3
            fig = go.Figure()
            _line(fig, ts_fmt, mem_avl,  "Available GiB", "#22c55e", fill=True)
            _line(fig, ts_fmt, mem_used, "Used %",         "#f97316", dash="dot", yaxis="y2")
            if total_gib > 0:
                fig.add_hline(y=0.5, line_color="rgba(239,68,68,0.5)", line_dash="dot",
                             annotation_text="< 500 MiB ⚠", line_width=1)
            fig.update_layout(**_CHART, title=f"RAM Available GiB (Total: {total_gib:.1f} GiB)",
                              yaxis_title="GiB",
                              yaxis2=dict(overlaying="y", side="right", title="%",
                                          color="#f97316", gridcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, key="c_mem", width="stretch")

    _charts()

    # ── Animated event ticker ─────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _event_ticker() -> None:
        st.divider()
        st.subheader("🔔 Live Event Stream")

        try:
            evs = [dict(r) for r in conn.execute(
                "SELECT ts, kind, payload FROM events ORDER BY id DESC LIMIT 20"
            ).fetchall()]
        except Exception:  # noqa: BLE001
            evs = []

        if not evs:
            st.caption("No events yet — waiting for daemon activity.")
            return

        _COLORS = {
            "psi_trigger_fired":      ("#ef4444", "🔥", "PSI SPIKE"),
            "psi_trigger_registered": ("#f97316", "🔔", "PSI MONITOR"),
            "risk_change":            ("#f59e0b", "⚠️", "RISK CHANGE"),
            "proc_birth":             ("#22c55e", "🟢", "NEW PROCESS"),
            "proc_death":             ("#ef4444", "🔴", "PROC EXIT"),
            "process_flagged":        ("#f43f5e", "🚨", "FLAGGED"),
            "rl_train_start":         ("#38bdf8", "🚀", "TRAIN START"),
            "rl_train_step":          ("#818cf8", "📈", "TRAIN STEP"),
            "rl_train_done":          ("#22c55e", "🏁", "TRAIN DONE"),
            "auto_train_triggered":   ("#34d399", "⚡", "AUTO TRAIN"),
            "auto_train_status":      ("#334155", "📊", "SCHEDULER"),
        }

        for i, ev in enumerate(evs[:12]):
            ts_str = dt.datetime.fromtimestamp(float(ev["ts"])).strftime("%H:%M:%S")
            kind   = ev.get("kind", "?")
            color, icon, badge = _COLORS.get(kind, ("#475569", "•", kind.upper()[:12]))

            if kind == "auto_train_status":
                continue  # Skip noisy heartbeats

            try:
                p = json.loads(ev.get("payload") or "{}")
                detail = (p.get("name") or p.get("new_level") or p.get("risk_level") or
                         (f"steps={p.get('steps_done',0):,}" if "steps_done" in p else "") or
                         str(p)[:50])
            except Exception:  # noqa: BLE001
                detail = str(ev.get("payload") or "")[:50]

            delay = i * 0.03
            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:center;padding:7px 12px;"
                f"margin:3px 0;border-radius:8px;"
                f"background:linear-gradient(135deg,rgba(10,20,45,0.8),rgba(15,30,60,0.5));"
                f"border:1px solid {color}22;border-left:3px solid {color};"
                f"animation:slide-in-up {0.3+delay:.2f}s ease-out;'>"
                f"<code style='color:#334155;font-size:10px;white-space:nowrap;"
                f"font-family:JetBrains Mono,monospace'>{ts_str}</code>"
                f"<span style='background:{color}22;color:{color};font-size:9px;"
                f"font-weight:700;letter-spacing:0.5px;padding:2px 6px;"
                f"border-radius:4px;white-space:nowrap'>{badge}</span>"
                f"<span style='font-size:13px'>{icon}</span>"
                f"<span style='color:#64748b;font-size:12px;flex:1'>{detail}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    _event_ticker()
