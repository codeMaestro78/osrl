"""Data Explorer page — browse every sample, feature distributions, auto-train status."""

from __future__ import annotations

import datetime
import json

import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from osrl.dashboard.shared import checkpoint_dir, get_conn, training_available

# All numeric features in the samples table
FEATURES = [
    ("cpu_total_pct",         "CPU Total %"),
    ("load1",                 "Load 1m"),
    ("load5",                 "Load 5m"),
    ("load15",                "Load 15m"),
    ("mem_used_pct",          "RAM Used %"),
    ("swap_used_pct",         "Swap Used %"),
    ("swp_in_ps",             "Swap-In pg/s"),
    ("swp_out_ps",            "Swap-Out pg/s"),
    ("majflt_ps",             "Major Faults/s"),
    ("reclaim_scan_ps",       "Reclaim Scan/s"),
    ("reclaim_steal_ps",      "Reclaim Steal/s"),
    ("allocstall_ps",         "Alloc Stall/s"),
    ("psi_cpu_some_stall_us", "PSI CPU Stall µs"),
    ("psi_mem_some_stall_us", "PSI Mem Stall µs"),
    ("psi_io_some_stall_us",  "PSI IO Stall µs"),
    ("psi_mem_full_stall_us", "PSI Mem Full µs"),
    ("psi_io_full_stall_us",  "PSI IO Full µs"),
    ("gpu_util_pct",          "GPU Util %"),
    ("gpu_mem_used_mib",      "GPU VRAM MiB"),
    ("gpu_temp_c",            "GPU Temp °C"),
    ("gpu_power_w",           "GPU Power W"),
]

_BG = "rgba(0,0,0,0)"
_PLOT = "rgba(6,13,26,0.9)"
_FONT = dict(color="#94a3b8", size=11, family="Inter")
_GRID = "rgba(56,189,248,0.06)"


def render() -> None:
    st.title("🔬 Data Explorer")
    conn    = get_conn()
    refresh = st.session_state.get("refresh_s", 10)

    # ── Auto-train status (fragment) ──────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _auto_train_status() -> None:
        st.subheader("🤖 Auto-Training Scheduler")

        # Get latest auto_train_status event
        row = conn.execute(
            "SELECT ts, payload FROM events WHERE kind='auto_train_status' ORDER BY id DESC LIMIT 1"
        ).fetchone()

        # Also check if triggered recently
        trig = conn.execute(
            "SELECT ts, payload FROM events WHERE kind='auto_train_triggered' ORDER BY id DESC LIMIT 1"
        ).fetchone()

        running_row = conn.execute(
            "SELECT ts FROM events WHERE kind='auto_train_running' ORDER BY id DESC LIMIT 1"
        ).fetchone()

        if row is None:
            st.warning(
                "⏸ Auto-trainer not running.\n\n"
                "Start it with:\n"
                "```bash\n"
                "nohup python -m osrl.rl.auto_train \\\n"
                "  --db ~/os-rl-agent/data/telemetry.db \\\n"
                "  --synthetic \\\n"
                "  --min-samples 500 \\\n"
                "  --interval 3600 \\\n"
                "  > /tmp/osrl-auto-train.log 2>&1 &\n"
                "```"
            )
        else:
            try:
                p = json.loads(row[1] or "{}")
            except Exception:  # noqa: BLE001
                p = {}

            total     = p.get("total_samples", 0)
            new_samp  = p.get("new_samples", 0)
            needed    = p.get("needed_samples", 0)
            wait_s    = p.get("next_run_in_s", 0)
            is_run    = p.get("already_running", False)
            sched_ts  = p.get("scheduler_ts", 0)
            staleness = datetime.datetime.now().timestamp() - float(sched_ts) if sched_ts else 999

            # Status indicator
            if staleness > 300:
                st.warning("⚠️ Scheduler heartbeat stale >5 min — may have stopped")
            elif is_run:
                st.success("🏋️ Training is **currently running**")
            elif trig and float(trig[0]) > float(row[0]) - 120:
                st.info("🚀 Training was triggered recently — starting up…")
            else:
                pct_done = min(100, int(new_samp / max(1, needed + new_samp) * 100)) if needed > 0 else 100
                st.info(f"✅ Scheduler running — waiting for conditions")

            # Metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Samples",   f"{total:,}")
            c2.metric("New Since Last",  f"{new_samp:,}")
            c3.metric("Need More",       f"{needed:,}" if needed > 0 else "✅ Ready")
            c4.metric("Time Until Run",  f"{wait_s // 3600}h {(wait_s % 3600) // 60}m" if wait_s > 0 else "✅ Ready")
            c5.metric("Scheduler",       "🟢 Live" if staleness < 120 else "🔴 Stale")

            # Progress bars
            if needed > 0:
                sample_pct = min(1.0, new_samp / (needed + new_samp))
                st.markdown("**Sample collection progress:**")
                st.progress(sample_pct, text=f"{new_samp:,} / {needed + new_samp:,} samples")

            if wait_s > 0:
                time_pct = 1.0 - min(1.0, wait_s / 3600.0)
                st.markdown("**Time interval progress:**")
                st.progress(time_pct, text=f"Waiting {wait_s // 60} more minutes")

        # Recent auto_train events
        auto_evs = [dict(r) for r in conn.execute(
            "SELECT ts, kind, payload FROM events "
            "WHERE kind LIKE 'auto_train%' OR kind LIKE 'rl_train%' "
            "ORDER BY id DESC LIMIT 20"
        ).fetchall()]

        if auto_evs:
            with st.expander("📋 Auto-Train Event Log", expanded=True):
                _COLOR = {
                    "auto_train_status":    "#475569",
                    "auto_train_triggered": "#22c55e",
                    "auto_train_running":   "#38bdf8",
                    "rl_train_start":       "#38bdf8",
                    "rl_train_step":        "#818cf8",
                    "rl_train_done":        "#22c55e",
                }
                _ICON = {
                    "auto_train_status":    "📊",
                    "auto_train_triggered": "🚀",
                    "auto_train_running":   "🏋️",
                    "rl_train_start":       "🚀",
                    "rl_train_step":        "📈",
                    "rl_train_done":        "🏁",
                }
                for ev in auto_evs:
                    if ev.get("kind") == "auto_train_status":
                        continue  # too noisy — skip status heartbeats
                    ts_str = datetime.datetime.fromtimestamp(float(ev["ts"])).strftime("%H:%M:%S")
                    kind   = ev.get("kind", "?")
                    color  = _COLOR.get(kind, "#64748b")
                    icon   = _ICON.get(kind, "•")
                    try:
                        p = json.loads(ev.get("payload") or "{}")
                    except Exception:  # noqa: BLE001
                        p = {}
                    detail = _format_event(kind, p)
                    st.markdown(
                        f"<div style='display:flex;gap:8px;align-items:flex-start;"
                        f"padding:6px 10px;margin:2px 0;border-radius:6px;"
                        f"background:rgba(15,30,60,0.5);border-left:3px solid {color}'>"
                        f"<code style='color:#475569;font-size:11px;white-space:nowrap'>{ts_str}</code>"
                        f"<span>{icon}</span>"
                        f"<span style='color:{color};font-weight:600;font-size:12px;white-space:nowrap'>{kind}</span>"
                        f"<span style='color:#94a3b8;font-size:11px'>{detail}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Quick-start buttons
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Trigger Training NOW", type="primary",
                         help="Bypass scheduler — launch training immediately"):
                _trigger_now(conn)
        with col2:
            st.caption(f"Min samples: {MIN_SAMPLES} | Min interval: 1h")
        with col3:
            if st.button("🔄 Refresh Status"):
                st.rerun()

    _auto_train_status()

    st.divider()

    # ── Dataset overview (fragment) ───────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _dataset_overview() -> None:
        st.subheader("📦 Training Dataset")

        total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        bounds = conn.execute("SELECT MIN(ts), MAX(ts) FROM samples").fetchone()

        if total == 0:
            st.info("No samples yet.")
            return

        lo = datetime.datetime.fromtimestamp(float(bounds[0])).strftime("%Y-%m-%d %H:%M")
        hi = datetime.datetime.fromtimestamp(float(bounds[1])).strftime("%Y-%m-%d %H:%M")
        duration_h = (float(bounds[1]) - float(bounds[0])) / 3600

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Samples",      f"{total:,}")
        d2.metric("Duration",           f"{duration_h:.1f}h")
        d3.metric("From",               lo)
        d4.metric("To",                 hi)

        # ── Sample rate chart ──────────────────────────────────────────────────
        buckets = conn.execute("""
            SELECT CAST(ts / 300 AS INT) * 300 AS bucket, COUNT(*) as n
            FROM samples
            GROUP BY bucket
            ORDER BY bucket
        """).fetchall()

        if buckets:
            bts = [datetime.datetime.fromtimestamp(float(b[0])) for b in buckets]
            bct = [b[1] for b in buckets]
            fig = go.Figure(go.Bar(x=bts, y=bct, marker_color="#38bdf8", opacity=0.8,
                                  name="Samples per 5-min bucket"))
            fig.update_layout(
                title="Sample Collection Rate (5-min buckets)",
                height=220, margin=dict(t=36, b=30, l=50, r=20),
                paper_bgcolor=_BG, plot_bgcolor=_PLOT, font=_FONT,
                xaxis=dict(gridcolor=_GRID), yaxis=dict(gridcolor=_GRID, title="count"),
                showlegend=False,
            )
            st.plotly_chart(fig, key="sample_rate", width="stretch")

    _dataset_overview()

    # ── Feature distributions (fragment) ─────────────────────────────────────
    @st.fragment(run_every=refresh * 3)
    def _feature_dists() -> None:
        st.divider()
        st.subheader("📊 Feature Distributions")
        st.caption("Histograms and box plots of all observation features the RL policy trains on.")

        # Select features to display
        col1, col2 = st.columns([3, 1])
        with col1:
            feat_names   = [f[1] for f in FEATURES]
            selected_idx = st.multiselect(
                "Features", feat_names,
                default=feat_names[:8],
                key="feat_select",
                label_visibility="collapsed",
                placeholder="Select features to explore…",
            )
        with col2:
            chart_type = st.selectbox("Chart", ["Histogram", "Box", "Violin"],
                                      key="feat_chart_type", label_visibility="collapsed")

        # Lookup column names for selected
        name_to_col = {f[1]: f[0] for f in FEATURES}
        selected_cols = [name_to_col[n] for n in selected_idx if n in name_to_col]

        if not selected_cols:
            st.info("Select features above.")
            return

        # Load data
        placeholders = ",".join(selected_cols)
        rows = conn.execute(f"SELECT ts, {placeholders} FROM samples ORDER BY ts").fetchall()
        if not rows:
            return

        ts_vals = [datetime.datetime.fromtimestamp(float(r[0])) for r in rows]
        data = {col: [r[i+1] for r in rows] for i, col in enumerate(selected_cols)}

        # Draw charts in 2-column grid
        for i in range(0, len(selected_cols), 2):
            c1, c2 = st.columns(2)
            for j, col_name in enumerate(selected_cols[i:i+2]):
                label  = [f[1] for f in FEATURES if f[0] == col_name][0]
                vals   = [v for v in data[col_name] if v is not None]
                target = c1 if j == 0 else c2

                if not vals:
                    target.info(f"No data for {label}")
                    continue

                key_sfx = f"{col_name}_{i}_{j}"

                if chart_type == "Histogram":
                    fig = go.Figure(go.Histogram(
                        x=vals, nbinsx=40, name=label,
                        marker_color="#38bdf8", opacity=0.75,
                    ))
                    fig.update_layout(**_mini_layout(label))
                    target.plotly_chart(fig, key=f"hist_{key_sfx}", width="stretch")

                elif chart_type == "Box":
                    fig = go.Figure(go.Box(
                        y=vals, name=label,
                        marker_color="#38bdf8",
                        boxmean="sd",
                        fillcolor="rgba(56,189,248,0.15)",
                        line=dict(color="#38bdf8"),
                    ))
                    fig.update_layout(**_mini_layout(label))
                    target.plotly_chart(fig, key=f"box_{key_sfx}", width="stretch")

                else:  # Violin
                    fig = go.Figure(go.Violin(
                        y=vals, name=label,
                        fillcolor="rgba(56,189,248,0.15)",
                        line_color="#38bdf8",
                        box_visible=True, meanline_visible=True,
                    ))
                    fig.update_layout(**_mini_layout(label))
                    target.plotly_chart(fig, key=f"vio_{key_sfx}", width="stretch")

    _feature_dists()

    # ── Correlation heatmap (fragment) ────────────────────────────────────────
    @st.fragment(run_every=refresh * 5)
    def _correlation() -> None:
        st.divider()
        st.subheader("🔥 Feature Correlation Matrix")
        st.caption("How correlated are the 21 observation features? Dark = strong correlation.")

        cols = [f[0] for f in FEATURES]
        labels = [f[1] for f in FEATURES]
        placeholders = ",".join(cols)
        rows = conn.execute(f"SELECT {placeholders} FROM samples ORDER BY ts DESC LIMIT 2000").fetchall()

        if len(rows) < 10:
            st.info("Need at least 10 samples to compute correlations.")
            return

        import numpy as np
        mat = np.array([[r[i] if r[i] is not None else 0.0 for i in range(len(cols))] for r in rows], dtype=float)
        # Compute correlation matrix
        with np.errstate(all="ignore"):
            corr = np.corrcoef(mat.T)
            corr = np.nan_to_num(corr, nan=0.0)

        fig = go.Figure(go.Heatmap(
            z=corr, x=labels, y=labels,
            colorscale="RdBu", zmid=0,
            text=[[f"{corr[i][j]:.2f}" for j in range(len(labels))] for i in range(len(labels))],
            texttemplate="%{text}",
            textfont=dict(size=8),
            colorbar=dict(thickness=12),
        ))
        fig.update_layout(
            height=520, margin=dict(t=40, b=80, l=120, r=20),
            paper_bgcolor=_BG, font=_FONT,
            xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig, key="corr_heatmap", width="stretch")

    _correlation()

    # ── Raw sample browser (fragment) ─────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _raw_browser() -> None:
        st.divider()
        st.subheader("🗃️ Raw Sample Browser")

        col1, col2 = st.columns([2, 1])
        with col1:
            limit = st.select_slider("Rows", [25, 50, 100, 200, 500], value=50,
                                     key="raw_limit", label_visibility="collapsed")
        with col2:
            order = st.selectbox("Order", ["Newest first", "Oldest first"],
                                 key="raw_order", label_visibility="collapsed")

        direction = "DESC" if order == "Newest first" else "ASC"
        rows = conn.execute(
            f"SELECT ts, cpu_total_pct, mem_used_pct, swap_used_pct, "
            f"swp_out_ps, psi_mem_some_stall_us, psi_io_some_stall_us, "
            f"gpu_util_pct, gpu_temp_c, load1, interval_s, collect_ms "
            f"FROM samples ORDER BY ts {direction} LIMIT ?",
            (limit,)
        ).fetchall()

        if not rows:
            st.info("No samples yet.")
            return

        table = []
        for r in rows:
            table.append({
                "Timestamp":     datetime.datetime.fromtimestamp(float(r[0])).strftime("%H:%M:%S"),
                "CPU %":         f"{r[1]:.1f}" if r[1] is not None else "—",
                "RAM Used %":    f"{r[2]:.0f}" if r[2] is not None else "—",
                "Swap %":        f"{r[3]:.1f}" if r[3] is not None else "—",
                "Swap-Out pg/s": f"{r[4]:.0f}" if r[4] is not None else "—",
                "PSI Mem µs":    f"{r[5]:.0f}" if r[5] is not None else "—",
                "PSI IO µs":     f"{r[6]:.0f}" if r[6] is not None else "—",
                "GPU %":         f"{r[7]:.0f}" if r[7] is not None else "—",
                "GPU °C":        f"{r[8]:.0f}" if r[8] is not None else "—",
                "Load1":         f"{r[9]:.2f}" if r[9] is not None else "—",
                "Interval s":    f"{r[10]:.1f}" if r[10] is not None else "—",
                "Collect ms":    f"{r[11]:.0f}" if r[11] is not None else "—",
            })

        st.dataframe(table, width="stretch", hide_index=True)

        # Missing value audit
        st.divider()
        st.markdown("**Feature completeness (% non-null)**")
        all_cols = [f[0] for f in FEATURES]
        completeness = []
        for col, lbl in FEATURES:
            n = conn.execute(f"SELECT COUNT(*) FROM samples WHERE {col} IS NOT NULL").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            pct = n / total * 100 if total > 0 else 0
            completeness.append({"Feature": lbl, "Non-Null %": round(pct, 1), "Count": n})

        fig = go.Figure(go.Bar(
            x=[r["Feature"] for r in completeness],
            y=[r["Non-Null %"] for r in completeness],
            marker_color=["#22c55e" if r["Non-Null %"] > 90 else "#f59e0b" if r["Non-Null %"] > 50 else "#ef4444"
                          for r in completeness],
            text=[f"{r['Non-Null %']:.0f}%" for r in completeness],
            textposition="outside",
        ))
        fig.update_layout(
            height=280, margin=dict(t=30, b=80, l=40, r=20),
            paper_bgcolor=_BG, plot_bgcolor=_PLOT, font=_FONT,
            xaxis=dict(tickangle=-35, gridcolor=_GRID, tickfont=dict(size=9)),
            yaxis=dict(gridcolor=_GRID, title="%", range=[0, 110]),
            showlegend=False,
        )
        st.plotly_chart(fig, key="completeness", width="stretch")

    _raw_browser()


MIN_SAMPLES = 500


def _trigger_now(conn) -> None:
    """Force-write an event to trigger training on next scheduler poll."""
    import subprocess, sys, time

    if not training_available():
        st.info(
            "🔒 Training launches are disabled in this deployment. The "
            "Streamlit Cloud demo runs a recorded snapshot without the RL "
            "extra (torch / stable-baselines3)."
        )
        return

    db_path = str(conn.execute("PRAGMA database_list").fetchone()[2])

    try:
        conn.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (time.time(), "auto_train_triggered",
             json.dumps({"source": "dashboard_manual", "new_samples": 0, "time_since_last": 9999})),
        )
        conn.commit()
    except Exception:  # noqa: BLE001
        pass

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "osrl.rl.train",
             "--db", db_path, "--steps", "50000", "--synthetic",
             "--checkpoint-dir",
             str(checkpoint_dir())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        st.success(f"✅ Training launched (PID {proc.pid}) — check Training page for progress.")
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ Failed: {e}")


def _format_event(kind: str, p: dict) -> str:
    if kind == "rl_train_start":
        return f"steps={p.get('n_steps',0):,}  synthetic={p.get('synthetic')}"
    if kind == "rl_train_step":
        v = p.get("validated", False)
        return f"step={p.get('steps_done',0):,}  reward={p.get('eval_mean_reward',0):.4f}  {'✅' if v else '❌'}"
    if kind == "rl_train_done":
        return f"best_reward={p.get('best_reward',0):.4f}  steps={p.get('total_steps',0):,}"
    if kind == "auto_train_triggered":
        return f"new_samples={p.get('new_samples',0)}  steps={p.get('steps',0):,}"
    if kind == "auto_train_running":
        return f"PID={p.get('pid')}  next_check={p.get('next_check_in')}s"
    return str(p)[:80]


def _mini_layout(title: str) -> dict:
    return dict(
        title=title, height=220,
        margin=dict(t=36, b=20, l=40, r=10),
        paper_bgcolor=_BG, plot_bgcolor=_PLOT, font=_FONT,
        xaxis=dict(gridcolor=_GRID, zeroline=False),
        yaxis=dict(gridcolor=_GRID, zeroline=False),
        showlegend=False,
    )

