"""Shared helpers for the Streamlit dashboard — production version.

Key design decisions:
- DB connection is cached with @st.cache_resource (one conn per process, survives reruns)
- All queries return plain Python dicts so they're picklable by st.cache_data
- Training events are read from the `events` table (kind=rl_train_*) where train.py writes them
- rl_metrics table is for future use; events table is the current source
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import time
from pathlib import Path

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Deployment mode
# ---------------------------------------------------------------------------

def is_demo() -> bool:
    """True when the dashboard is replaying the bundled recorded snapshot."""
    return os.environ.get("OSRL_DEMO") == "1"


def training_available() -> bool:
    """Training launcher is only offered where the RL extra is installed.

    Streamlit Community Cloud intentionally installs only the dashboard
    requirements (no torch / stable-baselines3), so the button is disabled
    instead of silently spawning a doomed subprocess.
    """
    if is_demo():
        return False
    return (
        importlib.util.find_spec("stable_baselines3") is not None
        and importlib.util.find_spec("sb3_contrib") is not None
    )


def checkpoint_dir() -> Path:
    """Resolve the checkpoint directory (env override -> repo -> home)."""
    env = os.environ.get("OSRL_RL_CHECKPOINT_DIR") or os.environ.get("OSRL_CHECKPOINT_DIR")
    if env:
        return Path(env).expanduser()
    repo = _REPO_ROOT / "checkpoints"
    if repo.exists():
        return repo
    return Path("~/os-rl-agent/checkpoints").expanduser()


# ---------------------------------------------------------------------------
# DB connection — cached per process, survives Streamlit reruns
# ---------------------------------------------------------------------------

def _resolve_db() -> str:
    return str(Path(os.environ.get("OSRL_DB", "~/os-rl-agent/data/telemetry.db")).expanduser())


@st.cache_resource
def get_conn() -> sqlite3.Connection:
    """Single shared SQLite connection. Lives for the lifetime of the server process."""
    path = _resolve_db()
    from osrl.dashboard.bootstrap import ensure_schema
    ensure_schema(Path(path))
    conn = sqlite3.connect(path, check_same_thread=False, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def get_db_path() -> Path:
    return Path(_resolve_db())


# ---------------------------------------------------------------------------
# Safe query helpers
# ---------------------------------------------------------------------------

def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    """Execute a SELECT and return list of plain dicts (picklable for st.cache_data)."""
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except sqlite3.OperationalError:
        return []


def _one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> dict | None:
    try:
        r = conn.execute(sql, params).fetchone()
        return dict(r) if r else None
    except sqlite3.OperationalError:
        return None


# ---------------------------------------------------------------------------
# Live data fetchers (called every fragment refresh)
# ---------------------------------------------------------------------------

def latest_sample(conn: sqlite3.Connection) -> dict | None:
    return _one(conn, "SELECT * FROM samples ORDER BY ts DESC LIMIT 1")


def recent_samples(conn: sqlite3.Connection, limit: int = 300) -> list[dict]:
    rows = _rows(conn, "SELECT * FROM samples ORDER BY ts DESC LIMIT ?", (limit,))
    return list(reversed(rows))


def heartbeat_row(conn: sqlite3.Connection) -> dict | None:
    return _one(conn, "SELECT * FROM heartbeat WHERE id=1")


def agent_mode(conn: sqlite3.Connection) -> str:
    r = _one(conn, "SELECT value FROM meta WHERE key='agent_mode'")
    return str(r["value"]) if r else "observe"


def set_agent_mode(conn: sqlite3.Connection, mode: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('agent_mode', ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (mode,),
    )
    conn.commit()


def recent_events(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    return _rows(conn, "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))


def training_events(conn: sqlite3.Connection) -> list[dict]:
    """Training writes start/step/done events to the events table."""
    rows = _rows(
        conn,
        "SELECT ts, kind, payload FROM events WHERE kind LIKE 'rl_train%' ORDER BY id ASC",
    )
    result = []
    for r in rows:
        try:
            payload = json.loads(r["payload"] or "{}")
        except Exception:  # noqa: BLE001
            payload = {}
        result.append({**r, **payload})
    return result


def recent_rollups(conn: sqlite3.Connection, limit: int = 700) -> list[dict]:
    rows = _rows(conn, "SELECT * FROM rollups ORDER BY window_start DESC LIMIT ?", (limit,))
    return list(reversed(rows))


def recent_rl_metrics(conn: sqlite3.Connection, limit: int = 500) -> list[dict]:
    """Read from rl_metrics table (future) + events table (current training output)."""
    # Primary: events table (where train.py actually writes)
    return training_events(conn)


def recent_actions(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    return _rows(conn, "SELECT * FROM actions_ledger ORDER BY created_ts DESC LIMIT ?", (limit,))


def top_procs(conn: sqlite3.Connection) -> list[dict]:
    row = latest_sample(conn)
    if row is None:
        return []
    try:
        return json.loads(row.get("top_procs_json") or "[]")
    except Exception:  # noqa: BLE001
        return []


def sample_count(conn: sqlite3.Connection) -> int:
    r = _one(conn, "SELECT COUNT(*) as n FROM samples")
    return int(r["n"]) if r else 0


def snapshot_bounds(conn: sqlite3.Connection) -> dict | None:
    """Recorded-data extent, used by the demo banner."""
    return _one(
        conn,
        "SELECT MIN(ts) AS lo, MAX(ts) AS hi, COUNT(*) AS n FROM samples",
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_checkpoints() -> list[dict]:
    """Return list of checkpoint dicts sorted newest-first."""
    ckpt_dir = checkpoint_dir()
    if not ckpt_dir.exists():
        return []
    ckpts = []
    for p in ckpt_dir.rglob("*.zip"):
        if p.suffix == ".zip" and p.is_file():
            from osrl.rl.artifacts import is_validated
            validated = is_validated(p)
            steps = 0
            if "step" in p.name:
                try:
                    steps = int(p.name.split("step")[1].split("_")[0])
                except (IndexError, ValueError):
                    steps = 0
            ckpts.append({
                "path": str(p),
                "name": p.name,
                "validated": validated,
                "steps": steps,
                "size_mb": p.stat().st_size / (1024 * 1024),
                "mtime": p.stat().st_mtime,
            })
    return sorted(ckpts, key=lambda x: x["mtime"], reverse=True)


def best_checkpoint() -> dict | None:
    ckpts = find_checkpoints()
    for name in ("best", "latest"):
        link = checkpoint_dir() / name
        if link.exists():
            selected = next((c for c in ckpts if Path(c["path"]).resolve() == link.resolve()), None)
            if selected:
                return selected
    # Prefer VALIDATED, otherwise latest
    validated = [c for c in ckpts if c["validated"]]
    return validated[0] if validated else (ckpts[0] if ckpts else None)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def safe_float(row: dict | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    v = row.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    import datetime
    return datetime.datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")


def ago(ts: float | None) -> str:
    if ts is None:
        return "never"
    s = int(time.time() - float(ts))
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m {s % 60}s ago"
    return f"{s // 3600}h ago"


# ---------------------------------------------------------------------------
# Rollup computation (on-the-fly when table is empty)
# ---------------------------------------------------------------------------

def ensure_rollups(conn: sqlite3.Connection) -> None:
    """Compute missing rollup windows from raw samples (15-min buckets)."""
    n = _one(conn, "SELECT COUNT(*) as n FROM rollups")
    if n and n["n"] > 0:
        return   # already have rollup data

    # Get sample range
    bounds = _one(conn, "SELECT MIN(ts) as lo, MAX(ts) as hi FROM samples")
    if not bounds or bounds["lo"] is None:
        return

    lo = float(bounds["lo"])
    hi = float(bounds["hi"])
    window = 900.0  # 15 minutes

    # Snap lo down to nearest window boundary
    lo = (lo // window) * window
    t = lo
    while t < hi:
        t_end = t + window
        rows = _rows(
            conn,
            "SELECT * FROM samples WHERE ts >= ? AND ts < ?",
            (t, t_end),
        )
        if rows:
            def _avg(key: str) -> float | None:
                vals = [r[key] for r in rows if r.get(key) is not None]
                return sum(vals) / len(vals) if vals else None

            def _max(key: str) -> float | None:
                vals = [r[key] for r in rows if r.get(key) is not None]
                return max(vals) if vals else None

            def _sum(key: str) -> float:
                vals = [r[key] for r in rows if r.get(key) is not None]
                return sum(vals) if vals else 0.0

            agg = {
                "window_start": t,
                "window_end": t_end,
                "sample_count": len(rows),
                "cpu_total_pct_avg": _avg("cpu_total_pct"),
                "cpu_total_pct_max": _max("cpu_total_pct"),
                "mem_used_pct_avg": _avg("mem_used_pct"),
                "mem_available_bytes_avg": _avg("mem_available_bytes"),
                "swap_used_pct_avg": _avg("swap_used_pct"),
                "swap_used_pct_max": _max("swap_used_pct"),
                "swp_out_ps_avg": _avg("swp_out_ps"),
                "swp_out_ps_max": _max("swp_out_ps"),
                "psi_mem_stall_s_sum": _sum("psi_mem_some_stall_us") / 1e6,
                "psi_io_stall_s_sum": _sum("psi_io_some_stall_us") / 1e6,
                "extra_json": "{}",
            }
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO rollups (window_start,window_end,sample_count,"
                    "cpu_total_pct_avg,cpu_total_pct_max,mem_used_pct_avg,mem_available_bytes_avg,"
                    "swap_used_pct_avg,swap_used_pct_max,swp_out_ps_avg,swp_out_ps_max,"
                    "psi_mem_stall_s_sum,psi_io_stall_s_sum,extra_json) VALUES "
                    "(:window_start,:window_end,:sample_count,:cpu_total_pct_avg,"
                    ":cpu_total_pct_max,:mem_used_pct_avg,:mem_available_bytes_avg,"
                    ":swap_used_pct_avg,:swap_used_pct_max,:swp_out_ps_avg,:swp_out_ps_max,"
                    ":psi_mem_stall_s_sum,:psi_io_stall_s_sum,:extra_json)",
                    agg,
                )
            except Exception:  # noqa: BLE001
                pass
        t = t_end
    conn.commit()


def render_action_ledger(conn: sqlite3.Connection, limit: int = 30) -> None:
    """Display the real schema, including failures and guard decisions."""
    rows = recent_actions(conn, limit)
    st.caption("Guarded worker attempts. Execution requires an explicit worker invocation with --execute.")
    if not rows:
        st.info("No action attempts recorded. Changing agent mode alone does not start execution.")
        return
    st.dataframe([{
        "Time": fmt_ts(r["created_ts"]), "PID": r["target_pid"],
        "Action": r["kind"], "Status": r["status"],
        "Nice before": r["before_nice"], "Nice applied": r["applied_nice"],
        "Reason": r["error"] or "", "Intent": r["intent_id"],
    } for r in rows], hide_index=True, width="stretch")
    with st.expander("Guard audit details"):
        st.json(rows)
