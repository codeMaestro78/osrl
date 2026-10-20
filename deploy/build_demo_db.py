#!/usr/bin/env python3
"""Build a compact, real-telemetry demo snapshot for Streamlit Community Cloud.

Streamlit Community Cloud runs the dashboard in a container that cannot read
your laptop's ``/proc`` and has an ephemeral filesystem, so the Cloud app ships
a recorded snapshot instead of live telemetry. This script produces that
snapshot from a real collector database -- nothing is fabricated:

- copies the entire DB (schema, migration version, rollups, events, ledger),
- optionally trims old samples/events to the last ``--hours``,
- compacts the per-process JSON blobs (keeps the top ``--max-procs``),
- drops GPU process blobs (VRAM totals remain in the scalar columns),
- VACUUMs and switches the copy back to rollback journal so the committed file
  is self-contained (no ``-wal`` sidecar needed at runtime).

Usage::

    python deploy/build_demo_db.py \
        --src data/telemetry.db \
        --dst data/demo_telemetry.db

Then commit ``data/demo_telemetry.db`` (the repo's .gitignore has an
exception for it) and point Streamlit Community Cloud at ``streamlit_app.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

_HOME_RE = re.compile(r"/home/[^/\s\"']+/")


def _sanitize(text: str | None) -> str | None:
    """Replace /home/<user>/ with ~/ so a public demo does not leak a home path."""
    return _HOME_RE.sub("~/", text) if text else text


def _compact_procs(raw: str | None, limit: int) -> str | None:
    if not raw:
        return raw
    try:
        procs = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if not isinstance(procs, list):
        return raw
    procs = sorted(procs, key=lambda p: float(p.get("cpu_pct") or 0), reverse=True)[:limit]
    compact = []
    for p in procs:
        item: dict[str, object] = {}
        if "name" in p:
            item["name"] = p["name"]
        if "pid" in p:
            item["pid"] = p["pid"]
        if "cpu_pct" in p and p["cpu_pct"] is not None:
            item["cpu_pct"] = round(float(p["cpu_pct"]), 2)
        if "rss_mib" in p and p["rss_mib"] is not None:
            item["rss_mib"] = round(float(p["rss_mib"]), 1)
        if "start_time" in p:
            item["start_time"] = p["start_time"]
        compact.append(item)
    return json.dumps(compact, separators=(",", ":"))


def _compact_gpu_procs(raw: str | None, limit: int = 5) -> str | None:
    if not raw:
        return raw
    try:
        procs = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if not isinstance(procs, list):
        return raw
    procs = sorted(procs, key=lambda p: int(p.get("mem_bytes") or 0), reverse=True)[:limit]
    return json.dumps(
        [{"pid": p.get("pid"), "mem_bytes": p.get("mem_bytes")} for p in procs],
        separators=(",", ":"),
    )


def build(src: Path, dst: Path, hours: float, max_procs: int) -> dict[str, object]:
    if not src.exists():
        raise SystemExit(f"source database not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    for suffix in ("-wal", "-shm"):
        Path(str(dst) + suffix).unlink(missing_ok=True)

    src_conn = sqlite3.connect(f"file:{src.resolve()}?mode=ro", uri=True)
    dst_conn = sqlite3.connect(str(dst))
    try:
        src_conn.backup(dst_conn)
    finally:
        src_conn.close()

    hi = dst_conn.execute("SELECT MAX(ts) FROM samples").fetchone()[0]
    if hi is None:
        raise SystemExit(f"source database has no samples: {src}")

    cutoff = float(hi) - hours * 3600.0 if hours > 0 else None
    if cutoff is not None:
        dst_conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
        dst_conn.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        dst_conn.execute("DELETE FROM rollups WHERE window_end <= ?", (cutoff,))

    rows = dst_conn.execute(
        "SELECT id, top_procs_json, gpu_top_procs_json FROM samples"
    ).fetchall()
    updates = [
        (
            _compact_procs(top, max_procs),
            _compact_gpu_procs(gpu_top),
            sample_id,
        )
        for sample_id, top, gpu_top in rows
    ]
    dst_conn.executemany(
        "UPDATE samples SET top_procs_json = ?, gpu_top_procs_json = ? WHERE id = ?",
        updates,
    )

    sanitized = 0
    event_rows = dst_conn.execute("SELECT id, payload FROM events").fetchall()
    for event_id, payload in event_rows:
        clean = _sanitize(payload)
        if clean != payload:
            dst_conn.execute("UPDATE events SET payload = ? WHERE id = ?", (clean, event_id))
            sanitized += 1

    samples = dst_conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    events = dst_conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    lo, hi = dst_conn.execute("SELECT MIN(ts), MAX(ts) FROM samples").fetchone()
    dst_conn.execute(
        "INSERT INTO meta (key, value) VALUES ('demo_snapshot', ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (
            json.dumps(
                {
                    "source": src.name,
                    "generated_ts": time.time(),
                    "samples": samples,
                    "events": events,
                    "from_ts": lo,
                    "to_ts": hi,
                },
                separators=(",", ":"),
            ),
        ),
    )
    dst_conn.commit()

    dst_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    dst_conn.execute("VACUUM")
    dst_conn.execute("PRAGMA journal_mode=DELETE")
    dst_conn.commit()
    dst_conn.close()

    size_mb = dst.stat().st_size / (1024 * 1024)
    return {
        "dst": str(dst),
        "samples": samples,
        "events": events,
        "sanitized_events": sanitized,
        "from_ts": lo,
        "to_ts": hi,
        "span_hours": round((float(hi) - float(lo)) / 3600.0, 2),
        "size_mb": round(size_mb, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="data/telemetry.db", help="recorded collector DB")
    p.add_argument("--dst", default="data/demo_telemetry.db", help="output snapshot")
    p.add_argument(
        "--hours",
        type=float,
        default=0.0,
        help="keep only the last N hours (0 = keep the whole recording)",
    )
    p.add_argument(
        "--max-procs",
        type=int,
        default=8,
        help="per-sample process rows to keep (dashboard shows 15; 8 keeps size small)",
    )
    args = p.parse_args(argv)

    summary = build(Path(args.src).expanduser(), Path(args.dst).expanduser(),
                    args.hours, args.max_procs)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
