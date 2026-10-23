"""OS-RL-Agent Streamlit dashboard — premium animated UI entry point."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="OS-RL Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "**OS-RL Agent** — PSI-aware RL watchdog for Ubuntu laptops.",
    },
)

st.set_option("client.showSidebarNavigation", False)

# ── Premium Animated CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── Keyframe animations ── */
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 8px rgba(56,189,248,0.3), 0 0 0 0 rgba(56,189,248,0.2); }
    50%       { box-shadow: 0 0 20px rgba(56,189,248,0.6), 0 0 30px rgba(56,189,248,0.1); }
}
@keyframes pulse-dot {
    0%, 100% { transform: scale(1); opacity: 1; }
    50%       { transform: scale(1.4); opacity: 0.7; }
}
@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes slide-in-down {
    from { transform: translateY(-16px); opacity: 0; }
    to   { transform: translateY(0);     opacity: 1; }
}
@keyframes slide-in-up {
    from { transform: translateY(16px); opacity: 0; }
    to   { transform: translateY(0);    opacity: 1; }
}
@keyframes fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes shimmer {
    0%   { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}
@keyframes border-spin {
    0%   { border-color: rgba(56,189,248,0.6) rgba(129,140,248,0.2) rgba(167,139,250,0.2) rgba(56,189,248,0.2); }
    25%  { border-color: rgba(56,189,248,0.2) rgba(56,189,248,0.6) rgba(129,140,248,0.2) rgba(167,139,250,0.2); }
    50%  { border-color: rgba(167,139,250,0.2) rgba(56,189,248,0.2) rgba(56,189,248,0.6) rgba(129,140,248,0.2); }
    75%  { border-color: rgba(129,140,248,0.2) rgba(167,139,250,0.2) rgba(56,189,248,0.2) rgba(56,189,248,0.6); }
    100% { border-color: rgba(56,189,248,0.6) rgba(129,140,248,0.2) rgba(167,139,250,0.2) rgba(56,189,248,0.2); }
}
@keyframes count-up {
    from { transform: translateY(4px); opacity: 0; }
    to   { transform: translateY(0);   opacity: 1; }
}

/* ── Global background — animated mesh gradient ── */
.stApp {
    background: #060d1a;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(29,78,216,0.12) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(129,140,248,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(56,189,248,0.06) 0%, transparent 50%);
    color: #e2e8f0;
}
[data-testid="stAppViewContainer"] { background: transparent; }

/* ── Sidebar — animated gradient border ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f1f 0%, #0b1830 50%, #080f1f 100%);
    border-right: 1px solid transparent;
    border-image: linear-gradient(180deg, rgba(56,189,248,0.4), rgba(129,140,248,0.2), rgba(56,189,248,0.4)) 1;
    animation: fade-in 0.6s ease-out;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Metric cards — glass + glow ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg,
        rgba(15,30,60,0.95) 0%,
        rgba(20,40,85,0.8) 100%);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 14px;
    padding: 18px 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: all 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    animation: slide-in-up 0.5s ease-out;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 200%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.04), transparent);
    transition: left 0.6s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(56,189,248,0.55);
    box-shadow: 0 8px 32px rgba(56,189,248,0.15), 0 0 0 1px rgba(56,189,248,0.1);
    transform: translateY(-4px) scale(1.01);
}
[data-testid="stMetric"]:hover::before { left: 100%; }
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 10px !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    animation: count-up 0.4s ease-out;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Buttons — animated gradient ── */
.stButton > button {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 50%, #2563eb 100%);
    background-size: 200% 200%;
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.3px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(29,78,216,0.35);
    font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover {
    background-position: right center;
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(59,130,246,0.5);
    animation: gradient-shift 3s ease infinite;
}
.stButton > button:active { transform: translateY(-1px); }

/* Primary button special ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #059669, #10b981, #34d399);
    background-size: 200% 200%;
    box-shadow: 0 4px 15px rgba(16,185,129,0.4);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 25px rgba(52,211,153,0.5);
}

/* ── Tabs — pill style with glow ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(8,15,31,0.8);
    border-radius: 12px;
    padding: 5px;
    border: 1px solid rgba(56,189,248,0.12);
    gap: 3px;
    backdrop-filter: blur(8px);
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    border-radius: 9px;
    font-weight: 500;
    font-size: 13px;
    transition: all 0.25s ease;
    padding: 8px 16px;
}
.stTabs [data-baseweb="tab"]:hover { color: #94a3b8 !important; background: rgba(56,189,248,0.06) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
    color: white !important;
    box-shadow: 0 3px 12px rgba(29,78,216,0.5), 0 0 0 1px rgba(56,189,248,0.2);
    animation: pulse-glow 3s ease-in-out infinite;
}

/* ── Inputs — glass style ── */
.stSelectbox > div > div,
.stNumberInput > div > div,
.stTextInput > div > div {
    background: rgba(8,15,31,0.9) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s ease;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within {
    border-color: rgba(56,189,248,0.6) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.1) !important;
}
.stSlider > div { color: #94a3b8 !important; }
.stSlider [data-baseweb="slider"] > div:first-child {
    background: linear-gradient(90deg, #1d4ed8, #38bdf8) !important;
}

/* ── Multiselect ── */
[data-baseweb="tag"] {
    background: rgba(29,78,216,0.3) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 6px !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(56,189,248,0.15);
    animation: slide-in-up 0.4s ease-out;
}
iframe[title="st.dataframe"] { border: none !important; background: transparent; }

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #1d4ed8, #38bdf8, #818cf8) !important;
    background-size: 200% 100%;
    border-radius: 4px;
    animation: gradient-shift 2s linear infinite;
}
.stProgress > div { background: rgba(15,30,60,0.6) !important; border-radius: 4px; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(8,15,31,0.8) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(56,189,248,0.12) !important;
    transition: all 0.2s ease;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(56,189,248,0.35) !important;
    background: rgba(15,30,60,0.9) !important;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 12px;
    border: none;
    backdrop-filter: blur(8px);
    animation: slide-in-down 0.4s ease-out;
}
[data-testid="stNotification"] { border-radius: 12px !important; }

/* ── Toasts ── */
[data-testid="stToast"] {
    background: linear-gradient(135deg, #0b1830, #112244) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(56,189,248,0.4) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.5) !important;
    animation: slide-in-down 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

/* ── Dividers ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.2), transparent) !important;
    margin: 1.2rem 0 !important;
}

/* ── Headings ── */
h1 {
    background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 6s ease infinite, slide-in-down 0.5s ease-out;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}
h2 { color: #cbd5e1 !important; font-weight: 700 !important; }
h3 { color: #94a3b8 !important; font-weight: 600 !important; }

/* ── Nav radio (sidebar) ── */
.stRadio > div { gap: 2px !important; }
.stRadio label {
    padding: 10px 16px !important;
    border-radius: 10px !important;
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
    width: 100%;
    font-size: 13px;
    border: 1px solid transparent;
}
.stRadio label:hover {
    background: rgba(56,189,248,0.08) !important;
    border-color: rgba(56,189,248,0.15) !important;
}
div[data-testid="stRadio"] input:checked + label,
.stRadio [aria-checked="true"] label {
    background: linear-gradient(135deg, rgba(29,78,216,0.3), rgba(29,78,216,0.15)) !important;
    border-color: rgba(56,189,248,0.3) !important;
    box-shadow: 0 0 12px rgba(56,189,248,0.1);
}

/* ── Code blocks ── */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: rgba(15,30,60,0.8) !important;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.85em;
    color: #38bdf8 !important;
    border: 1px solid rgba(56,189,248,0.15);
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060d1a; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, rgba(56,189,248,0.4), rgba(129,140,248,0.4));
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(56,189,248,0.8); }

/* ── Plotly chart containers ── */
[data-testid="stPlotlyChart"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(56,189,248,0.08);
    animation: fade-in 0.5s ease-out;
    transition: box-shadow 0.3s ease;
}
[data-testid="stPlotlyChart"]:hover {
    box-shadow: 0 4px 24px rgba(56,189,248,0.08);
    border-color: rgba(56,189,248,0.18);
}

/* ── Form containers ── */
[data-testid="stForm"] {
    background: rgba(8,15,31,0.6);
    border-radius: 14px;
    padding: 20px;
    border: 1px solid rgba(56,189,248,0.12);
    backdrop-filter: blur(8px);
}

/* ── Sidebar bottom status ── */
.live-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 10px;
    font-size: 12px;
    color: #4ade80;
}
.live-dot {
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse-dot 1.5s ease-in-out infinite;
    box-shadow: 0 0 6px #22c55e;
}

/* ── Number inputs ── */
[data-testid="stNumberInputContainer"] {
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Checkbox ── */
.stCheckbox label { font-size: 13px !important; }

/* ── Spinner / loading ── */
.stSpinner > div {
    border-color: rgba(56,189,248,0.3) !important;
    border-top-color: #38bdf8 !important;
}

/* ── Caption ── */
.stCaption { color: #94a3b8 !important; font-size: 12px !important; }

/* Legibility, keyboard focus, narrow layouts and reduced motion. */
[data-testid="stMetricLabel"], .stCaption { color: #94a3b8 !important; }
button:focus-visible, a:focus-visible, input:focus-visible,
[role="tab"]:focus-visible { outline: 2px solid #7dd3fc !important; outline-offset: 3px; }
/* Frequently refreshed data should not replay entrance animations. */
main *, main *::before, main *::after { animation:none !important; }
.telemetry-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }
.telemetry-grid > div { min-width:0; }
@media (max-width:1100px) {
    .telemetry-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
.status-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px; }
[data-testid="stMarkdownContainer"] { overflow-wrap:anywhere; }
@media (max-width:640px) {
    .block-container { padding:1rem 0.75rem; }
    [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:0.75rem; }
    [data-testid="stColumn"] { min-width:0 !important; flex:1 1 100% !important; width:100% !important; }
    .status-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .status-grid > div, .telemetry-grid > div { padding:12px !important; min-width:0; }
    [data-testid="stMetricValue"] { font-size:20px !important; }
    [data-testid="stMarkdownContainer"] div[style*="display:flex"] { flex-wrap:wrap; }
    [data-testid="stMarkdownContainer"] span { white-space:normal !important; }
    .stTabs [data-baseweb="tab-list"] { overflow-x:auto; }
}
@media (prefers-reduced-motion:reduce) {
    *, *::before, *::after { animation:none !important; transition:none !important; scroll-behavior:auto !important; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "page":         "🖥️ Overview",
    "refresh_s":    10,
    "confirm_mode": None,
    "train_steps":  50_000,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Page imports ──────────────────────────────────────────────────────────────
from osrl.dashboard.pages import (  # noqa: E402
    agent,
    data_explorer,
    history,
    evaluation,
    overview,
    processes,
    training,
)
from osrl.dashboard.shared import get_conn, is_demo, snapshot_bounds  # noqa: E402

PAGES = {
    "🖥️ Overview":    overview,
    "⚙️ Processes":   processes,
    "🔬 Data Explorer": data_explorer,
    "🧪 Evaluation": evaluation,
    "🤖 Agent":       agent,
    "🏋️ Training":    training,
    "📅 History":     history,
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Animated logo
    st.markdown("""
    <div style='text-align:center;padding:24px 0 16px;animation:slide-in-down 0.6s ease-out'>
        <div style='font-size:3rem;margin-bottom:6px;
                    filter:drop-shadow(0 0 12px rgba(56,189,248,0.6))
                           drop-shadow(0 0 24px rgba(129,140,248,0.3))'>🧠</div>
        <div style='font-size:1.25rem;font-weight:800;letter-spacing:-0.3px;
                    background:linear-gradient(135deg,#38bdf8,#818cf8,#c084fc);
                    background-size:200% 200%;
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;
                    animation:gradient-shift 4s ease infinite'>OS-RL Agent</div>
        <div style='font-size:0.65rem;color:#94a3b8;margin-top:3px;
                    letter-spacing:2px;text-transform:uppercase;font-weight:600'>
            PSI-AWARE RL WATCHDOG
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation
    page_name = st.radio(
        "Navigation",
        options=list(PAGES.keys()),
        index=list(PAGES.keys()).index(st.session_state.page),
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state.page = page_name

    st.divider()

    # DB path info
    db_path = os.environ.get("OSRL_DB", "~/os-rl-agent/data/telemetry.db")
    db_expanded = str(Path(db_path).expanduser())
    st.markdown(f"""
    <div style='padding:4px 0;animation:fade-in 0.8s ease-out'>
        <div style='color:#94a3b8;text-transform:uppercase;letter-spacing:1px;
                    font-size:9px;font-weight:700;margin-bottom:6px'>Database</div>
        <code style='font-size:10px;color:#38bdf8;word-break:break-all;
                     background:rgba(56,189,248,0.06);border-color:rgba(56,189,248,0.2);
                     padding:4px 6px;display:block;border-radius:6px;
                     font-family:JetBrains Mono,monospace'>{Path(db_expanded).name}</code>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Auto-refresh slider
    st.markdown("""
    <div style='color:#94a3b8;text-transform:uppercase;letter-spacing:1px;
                font-size:9px;font-weight:700;margin-bottom:8px'>Auto-Refresh Interval</div>
    """, unsafe_allow_html=True)
    refresh_s = st.select_slider(
        "refresh",
        options=[5, 10, 30, 60, 300],
        value=st.session_state.refresh_s,
        format_func=lambda v: f"{v}s",
        label_visibility="collapsed",
        key="refresh_slider",
    )
    st.session_state.refresh_s = refresh_s

    # Live status badge
    st.markdown(f"""
    <div style='margin-top:14px;animation:fade-in 1s ease-out'>
        <div class='live-badge'>
            <div class='live-dot'></div>
            <span>Refreshing every <strong>{refresh_s}s</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats
    st.divider()
    try:
        from osrl.dashboard.shared import get_conn
        _conn = get_conn()
        _n = _conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        _ev = _conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        st.markdown(f"""
        <div style='display:flex;gap:8px;animation:fade-in 1.2s ease-out'>
            <div style='flex:1;text-align:center;padding:8px;
                        background:rgba(56,189,248,0.06);border-radius:8px;
                        border:1px solid rgba(56,189,248,0.12)'>
                <div style='font-size:16px;font-weight:800;color:#38bdf8;
                            font-family:JetBrains Mono,monospace'>{_n:,}</div>
                <div style='font-size:9px;color:#94a3b8;text-transform:uppercase;
                            letter-spacing:0.5px;margin-top:2px'>samples</div>
            </div>
            <div style='flex:1;text-align:center;padding:8px;
                        background:rgba(129,140,248,0.06);border-radius:8px;
                        border:1px solid rgba(129,140,248,0.12)'>
                <div style='font-size:16px;font-weight:800;color:#818cf8;
                            font-family:JetBrains Mono,monospace'>{_ev:,}</div>
                <div style='font-size:9px;color:#94a3b8;text-transform:uppercase;
                            letter-spacing:0.5px;margin-top:2px'>events</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:  # noqa: BLE001
        pass

# ── Main content ──────────────────────────────────────────────────────────────
if is_demo():
    import datetime as _dt

    try:
        _b = snapshot_bounds(get_conn())
    except Exception:  # noqa: BLE001
        _b = None
    if _b and _b.get("lo") is not None:
        _lo = _dt.datetime.fromtimestamp(float(_b["lo"])).strftime("%b %d, %Y %H:%M")
        _hi = _dt.datetime.fromtimestamp(float(_b["hi"])).strftime("%b %d, %Y %H:%M")
        _span = (float(_b["hi"]) - float(_b["lo"])) / 3600.0
        _detail = f"{_lo} → {_hi} · {_span:.1f}h · {int(_b['n']):,} samples"
    else:
        _detail = "bundled recorded telemetry"
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:12px;
                background:linear-gradient(135deg,rgba(56,189,248,0.08),rgba(129,140,248,0.05));
                border:1px solid rgba(56,189,248,0.25);border-radius:12px;
                padding:10px 16px;margin-bottom:14px;animation:fade-in 0.5s ease-out'>
        <span style='font-size:1.3rem'>📼</span>
        <div>
            <div style='color:#38bdf8;font-size:12px;font-weight:700;letter-spacing:0.5px'>
                RECORDED DEMO SNAPSHOT
            </div>
            <div style='color:#64748b;font-size:11px'>{_detail}</div>
        </div>
        <div style='margin-left:auto;color:#94a3b8;font-size:10px;max-width:360px;text-align:right'>
            Real telemetry from a developer workstation. Run the collector
            locally for live host monitoring.
        </div>
    </div>
    """, unsafe_allow_html=True)

PAGES[page_name].render()
