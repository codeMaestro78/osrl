"""Premium Training page — reward curves, checkpoint manager, live training log."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys

import plotly.graph_objects as go
import streamlit as st

from osrl.dashboard.shared import checkpoint_dir, get_conn, training_available

_BG, _PLOT = "rgba(0,0,0,0)", "rgba(6,13,26,0.95)"
_FONT = dict(color="#64748b", size=11, family="Inter")
_GRID = "rgba(56,189,248,0.05)"
_CHART = dict(
    paper_bgcolor=_BG, plot_bgcolor=_PLOT, font=_FONT,
    margin=dict(t=44, b=32, l=52, r=24), height=300,
    xaxis=dict(gridcolor=_GRID, showline=False, zeroline=False, title=""),
    yaxis=dict(gridcolor=_GRID, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#0f1f3d", font_color="#e2e8f0", bordercolor="rgba(56,189,248,0.4)"),
)


def _chart(**overrides) -> dict:
    """Return a copy of _CHART with any keyword overrides applied (avoids duplicate-key error)."""
    return {**_CHART, **overrides}


def render() -> None:
    st.title("🏋️ RL Training")
    conn    = get_conn()
    refresh = st.session_state.get("refresh_s", 10)
    ckpt_dir = checkpoint_dir()
    can_train = training_available()

    # ── Training metrics fragment ─────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _metrics() -> None:
        rows = [dict(r) for r in conn.execute(
            "SELECT ts, kind, payload FROM events "
            "WHERE kind IN ('rl_train_step','rl_train_done','rl_train_start') "
            "ORDER BY ts"
        ).fetchall()]

        if not rows:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;
                        background:rgba(8,15,31,0.6);border-radius:16px;
                        border:1px dashed rgba(56,189,248,0.2);margin:20px 0'>
                <div style='font-size:4rem;margin-bottom:12px;
                            filter:drop-shadow(0 0 12px rgba(56,189,248,0.4))'>🏋️</div>
                <div style='color:#94a3b8;font-size:16px;font-weight:600;margin-bottom:8px'>
                    No Training Data Yet
                </div>
                <div style='color:#475569;font-size:13px'>
                    Trigger a training run from the button below or wait for auto-training.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            steps_data = [r for r in rows if r["kind"] == "rl_train_step"]
            if steps_data:
                # Parse training runs
                all_steps, all_rewards, all_baselines = [], [], []
                validated_steps, validated_rewards = [], []
                for r in steps_data:
                    try: p = json.loads(r["payload"])
                    except Exception: continue  # noqa: BLE001
                    s = p.get("steps_done", 0)
                    rew = p.get("eval_mean_reward")
                    bas = p.get("baseline_reward")
                    if rew is not None:
                        all_steps.append(s)
                        all_rewards.append(rew)
                        if bas: all_baselines.append(bas)
                        else:   all_baselines.append(None)
                        if p.get("validated"):
                            validated_steps.append(s)
                            validated_rewards.append(rew)

                # Summary metrics
                best_reward = max(all_rewards)
                latest_reward = all_rewards[-1]
                n_steps_total = all_steps[-1] if all_steps else 0
                n_checkpoints = len(list(ckpt_dir.glob("*.zip"))) if ckpt_dir.exists() else 0

                m1, m2, m3, m4, m5 = st.columns(5)
                def _mcol(col, label, val, color="#38bdf8"):
                    col.markdown(f"""
                    <div style='background:rgba(10,20,45,0.9);border:1px solid {color}22;
                                border-radius:12px;padding:14px;text-align:center;
                                animation:slide-in-up 0.4s ease-out'>
                        <div style='color:#334155;font-size:9px;text-transform:uppercase;letter-spacing:1px'>{label}</div>
                        <div style='color:#f1f5f9;font-size:20px;font-weight:800;margin-top:4px;
                                    font-family:JetBrains Mono,monospace;color:{color}'>{val}</div>
                    </div>""", unsafe_allow_html=True)

                _mcol(m1, "Best Reward",   f"{best_reward:.2f}",      "#22c55e")
                _mcol(m2, "Latest Reward", f"{latest_reward:.2f}",    "#38bdf8")
                _mcol(m3, "Total Steps",   f"{n_steps_total:,}",      "#818cf8")
                _mcol(m4, "Checkpoints",   str(n_checkpoints),        "#f59e0b")
                _mcol(m5, "Validated",     str(len(validated_steps)),  "#34d399")

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Reward curve ───────────────────────────────────────────────
                fig = go.Figure()
                # Baseline band
                if any(all_baselines):
                    baselines_clean = [b if b else 0 for b in all_baselines]
                    fig.add_scatter(
                        x=all_steps, y=baselines_clean,
                        name="NOOP Baseline",
                        mode="lines",
                        line=dict(color="rgba(100,116,139,0.6)", dash="dot", width=1.5),
                        fill=None,
                    )
                    # Shade between baseline and reward
                    fig.add_scatter(
                        x=all_steps + all_steps[::-1],
                        y=all_rewards + baselines_clean[::-1],
                        fill="toself",
                        fillcolor="rgba(34,197,94,0.04)",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                        name="Improvement region",
                    )

                # Reward line with gradient-like color
                fig.add_scatter(
                    x=all_steps, y=all_rewards,
                    mode="lines+markers", name="Eval Reward",
                    line=dict(color="#38bdf8", width=2.5),
                    marker=dict(color="#38bdf8", size=5, symbol="circle",
                                line=dict(color="#fff", width=1)),
                )

                # Validated checkpoints
                if validated_steps:
                    fig.add_scatter(
                        x=validated_steps, y=validated_rewards,
                        mode="markers", name="✅ Validated",
                        marker=dict(color="#22c55e", size=12, symbol="star",
                                    line=dict(color="#fff", width=1.5)),
                    )

                fig.update_layout(
                    **_chart(height=340,
                             title="PPO Eval Reward vs NOOP Baseline",
                             xaxis_title="Training Steps",
                             yaxis_title="Mean Reward"),
                )
                st.plotly_chart(fig, key="reward_curve", width="stretch")

                # ── Reward improvement rate ────────────────────────────────────
                if len(all_rewards) >= 4:
                    mid = len(all_rewards) // 2
                    first_half = sum(all_rewards[:mid]) / mid
                    second_half = sum(all_rewards[mid:]) / len(all_rewards[mid:])
                    improvement = (second_half - first_half) / (abs(first_half) + 1e-6) * 100

                    imp_col = "#22c55e" if improvement > 0 else "#ef4444"
                    imp_icon = "📈" if improvement > 0 else "📉"
                    st.markdown(f"""
                    <div style='display:flex;gap:12px;margin:12px 0'>
                        <div style='flex:1;background:rgba(10,20,45,0.8);border-radius:10px;
                                    border:1px solid rgba(34,197,94,0.15);padding:12px 16px;
                                    display:flex;align-items:center;gap:10px'>
                            <span style='font-size:1.8rem'>{imp_icon}</span>
                            <div>
                                <div style='color:#334155;font-size:9px;text-transform:uppercase;letter-spacing:1px'>Reward Trend</div>
                                <div style='color:{imp_col};font-size:18px;font-weight:800;
                                            font-family:JetBrains Mono,monospace'>
                                    {improvement:+.1f}%
                                </div>
                                <div style='color:#475569;font-size:10px'>first half vs second half</div>
                            </div>
                        </div>
                        <div style='flex:1;background:rgba(10,20,45,0.8);border-radius:10px;
                                    border:1px solid rgba(56,189,248,0.15);padding:12px 16px;
                                    display:flex;align-items:center;gap:10px'>
                            <span style='font-size:1.8rem'>🎯</span>
                            <div>
                                <div style='color:#334155;font-size:9px;text-transform:uppercase;letter-spacing:1px'>vs Baseline</div>
                                <div style='color:#38bdf8;font-size:18px;font-weight:800;
                                            font-family:JetBrains Mono,monospace'>
                                    {best_reward - (all_baselines[-1] or 0):+.2f}
                                </div>
                                <div style='color:#475569;font-size:10px'>best reward above NOOP</div>
                            </div>
                        </div>
                        <div style='flex:1;background:rgba(10,20,45,0.8);border-radius:10px;
                                    border:1px solid rgba(129,140,248,0.15);padding:12px 16px;
                                    display:flex;align-items:center;gap:10px'>
                            <span style='font-size:1.8rem'>📊</span>
                            <div>
                                <div style='color:#334155;font-size:9px;text-transform:uppercase;letter-spacing:1px'>Checkpoints</div>
                                <div style='color:#818cf8;font-size:18px;font-weight:800;
                                            font-family:JetBrains Mono,monospace'>{n_checkpoints}</div>
                                <div style='color:#475569;font-size:10px'>{len(validated_steps)} validated</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    _metrics()
    st.divider()

    # ── Reward term breakdown (last step) ─────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _reward_breakdown() -> None:
        st.subheader("🔍 Reward Term Breakdown")
        st.caption("How each component contributed to the last evaluated reward")

        # Try to compute live breakdown
        try:
            from osrl.rl.reward import reward_breakdown
            from osrl.dashboard.shared import recent_samples

            rows = recent_samples(conn, limit=10)
            if not rows:
                st.info("No samples for breakdown.")
                return

            # Build minimal Sample-like objects
            class _S:
                def __init__(self, r):
                    for k, v in r.items():
                        setattr(self, k, v)
                    self.psi_cpu = _PSI(r.get("psi_cpu_some_stall_us"), r.get("psi_cpu_full_stall_us"), r.get("interval_s"))
                    self.psi_mem = _PSI(r.get("psi_mem_some_stall_us"), r.get("psi_mem_full_stall_us"), r.get("interval_s"))
                    self.psi_io  = _PSI(r.get("psi_io_some_stall_us"),  r.get("psi_io_full_stall_us"),  r.get("interval_s"))
                    self.top_procs = []

            class _PSI:
                def __init__(self, some, full, iv):
                    self.some_stall_us = some or 0
                    self.full_stall_us = full or 0
                    self.interval_s    = iv or 2.0

            window = [_S(r) for r in rows]
            bd = reward_breakdown(window)

            terms  = [k for k in bd if k != "total"]
            values = [bd[k] for k in terms]
            colors = ["#22c55e" if v >= 0 else "#ef4444" for v in values]
            labels = {
                "psi_mem": "PSI Mem",  "psi_io": "PSI IO",  "psi_cpu": "PSI CPU",
                "swap_out": "Swap Out", "swap_urgency": "Swap Urgency",
                "headroom": "RAM Free", "swap_free": "Swap Free",
                "fg_share": "Foreground", "oom_penalty": "OOM Risk",
                "stability": "Stability", "trend": "Trend", "action_cost": "Action Cost",
            }
            nice = [labels.get(t, t) for t in terms]

            fig = go.Figure(go.Bar(
                x=nice, y=values, marker_color=colors,
                text=[f"{v:+.3f}" for v in values], textposition="outside",
                textfont=dict(size=10, color="#94a3b8"),
            ))
            fig.add_hline(y=0, line_color="rgba(148,163,184,0.3)", line_width=1)
            total = bd["total"]
            fig.update_layout(
                **_chart(height=280,
                         title=f"Reward Terms — Total: {total:+.4f}",
                         yaxis_title="Contribution",
                         yaxis_range=[min(values + [0]) * 1.2, max(values + [0]) * 1.2]),
            )
            st.plotly_chart(fig, key="reward_bd", width="stretch")
        except Exception as e:  # noqa: BLE001
            st.caption(f"Breakdown unavailable: {e}")

    _reward_breakdown()
    st.divider()

    # ── Checkpoint manager ────────────────────────────────────────────────────
    @st.fragment(run_every=refresh * 2)
    def _checkpoints() -> None:
        st.subheader("💾 Checkpoint Manager")

        zips = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True) if ckpt_dir.exists() else []

        if not zips:
            st.info("No checkpoints found yet.")
        else:
            ckpt_data = []
            for z in zips:
                name = z.stem
                mtime = dt.datetime.fromtimestamp(z.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                size  = z.stat().st_size / 1024
                validated = "VALIDATED" in name
                steps = ""
                if "step" in name:
                    try: steps = f"{int(name.split('step')[1].split('_')[0]):,}"
                    except Exception: pass  # noqa: BLE001
                ckpt_data.append({
                    "Name":      name,
                    "Steps":     steps,
                    "Status":    "✅ VALIDATED" if validated else "⏳ UNVALIDATED",
                    "Saved":     mtime,
                    "Size KB":   f"{size:.0f}",
                })
            st.dataframe(ckpt_data, hide_index=True, width="stretch")

        # Best checkpoint display
        best = ckpt_dir / "best" if ckpt_dir.exists() else None
        if best and (best.exists() or best.is_symlink()):
            resolved = best.resolve().stem if best.is_symlink() else "best"
            st.markdown(f"""
            <div style='background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);
                        border-radius:10px;padding:12px 16px;margin-top:8px;
                        display:flex;align-items:center;gap:10px'>
                <span style='font-size:1.5rem'>🏆</span>
                <div>
                    <div style='color:#4ade80;font-weight:700;font-size:13px'>Best Checkpoint</div>
                    <code style='font-size:11px;color:#22c55e'>{resolved}</code>
                </div>
            </div>""", unsafe_allow_html=True)

    _checkpoints()
    st.divider()

    # ── Training launcher ─────────────────────────────────────────────────────
    st.subheader("🚀 Launch Training Run")
    if not can_train:
        st.info(
            "🔒 Training is disabled in this deployment. The Streamlit Cloud "
            "demo installs dashboard dependencies only (no torch / "
            "stable-baselines3) and replays a recorded snapshot. Run the app "
            "locally with the `.[rl]` extra to launch training."
        )

    with st.form("launch_form"):
        col1, col2, col3 = st.columns(3)
        steps_k   = col1.number_input("Steps (k)", 5, 500, 50, 5)
        synthetic = col2.checkbox("Synthetic stress injection", value=True)
        resume    = col3.checkbox("Resume from best checkpoint", value=True)

        db_path = conn.execute("PRAGMA database_list").fetchone()[2]
        submitted = st.form_submit_button(
            "🏋️ Launch Training", type="primary", width="stretch",
            disabled=not can_train,
        )

    if submitted:
        cmd = [sys.executable, "-m", "osrl.rl.train",
               "--db", db_path, "--steps", str(steps_k * 1000),
               "--checkpoint-dir", str(ckpt_dir)]
        if synthetic: cmd.append("--synthetic")
        best = ckpt_dir / "best"
        if resume and best.exists():
            cmd.extend(["--resume", str(best)])
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True)
            st.success(f"✅ Training started — PID {proc.pid}  ({steps_k}k steps)")
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ {e}")

    st.divider()

    # ── Live training log ─────────────────────────────────────────────────────
    @st.fragment(run_every=refresh)
    def _live_log() -> None:
        st.subheader("📋 Live Training Log")
        evs = [dict(r) for r in conn.execute(
            "SELECT ts, kind, payload FROM events "
            "WHERE kind LIKE 'rl_train%' OR kind LIKE 'auto_train%' "
            "ORDER BY id DESC LIMIT 30"
        ).fetchall()]

        if not evs:
            st.caption("No training events yet.")
            return

        _STYLE = {
            "rl_train_start":       ("#38bdf8", "🚀"),
            "rl_train_step":        ("#818cf8", "📈"),
            "rl_train_done":        ("#22c55e", "🏁"),
            "auto_train_triggered": ("#34d399", "⚡"),
            "auto_train_running":   ("#38bdf8", "🏋️"),
        }
        for ev in evs:
            if ev.get("kind") == "auto_train_status": continue
            ts_s  = dt.datetime.fromtimestamp(float(ev["ts"])).strftime("%H:%M:%S")
            kind  = ev.get("kind", "?")
            color, icon = _STYLE.get(kind, ("#475569", "•"))
            try:
                p = json.loads(ev.get("payload") or "{}")
            except Exception:  # noqa: BLE001
                p = {}

            if kind == "rl_train_step":
                detail = (f"step {p.get('steps_done',0):,} | "
                          f"reward={p.get('eval_mean_reward',0):.4f} | "
                          f"baseline={p.get('baseline_reward',0):.4f} | "
                          f"{'✅ VALIDATED' if p.get('validated') else '❌ not validated'}")
            elif kind == "rl_train_done":
                detail = f"best_reward={p.get('best_reward',0):.4f} | total_steps={p.get('total_steps',0):,}"
            elif kind == "rl_train_start":
                detail = f"steps={p.get('n_steps',0):,} | synthetic={p.get('synthetic')} | resume={p.get('resume_path')}"
            elif kind == "auto_train_triggered":
                detail = f"new_samples={p.get('new_samples',0)} | {p.get('steps',0):,} steps"
            else:
                detail = str(p)[:80]

            st.markdown(
                f"<div style='display:flex;gap:10px;align-items:flex-start;"
                f"padding:8px 12px;margin:2px 0;border-radius:8px;"
                f"background:rgba(8,15,31,0.8);border-left:3px solid {color}'>"
                f"<code style='color:#334155;font-size:10px;white-space:nowrap;"
                f"font-family:JetBrains Mono,monospace;margin-top:1px'>{ts_s}</code>"
                f"<span style='font-size:13px;margin-top:0px'>{icon}</span>"
                f"<span style='color:{color};font-size:11px;font-weight:700;"
                f"white-space:nowrap;margin-top:1px'>{kind}</span>"
                f"<span style='color:#475569;font-size:11px;flex:1'>{detail}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    _live_log()
