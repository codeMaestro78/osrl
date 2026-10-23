"""Database bootstrap shared by the Streamlit Cloud entrypoint and tests.

Kept free of Streamlit imports so it can be exercised directly:

    path = bootstrap_database(Path("/repo"))

Resolution order:
1. ``OSRL_DB`` when explicitly set (live deployment / secrets),
2. ``<root>/data/telemetry.db`` when present (local runs with a collector),
3. bundled ``<root>/data/demo_telemetry.db`` copied to a writable temp dir
   (Streamlit Community Cloud, ephemeral filesystem),
4. an empty v3 store in the temp dir (first-run placeholder).
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import uuid
import tempfile
from pathlib import Path


def ensure_schema(path: Path) -> None:
    """Create an empty v3 store so every page can query instead of crashing."""
    from osrl.store import Store

    store = Store(path)
    store.close()


def _work_dir() -> Path:
    return Path(tempfile.gettempdir()) / "osrl-dashboard"


def bootstrap_database(root: Path | None = None) -> str:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    os.environ.pop("OSRL_DEMO", None)

    explicit = os.environ.get("OSRL_DB")
    if explicit:
        chosen = Path(explicit).expanduser()
        if not chosen.exists() or chosen.stat().st_size == 0:
            chosen.parent.mkdir(parents=True, exist_ok=True)
            ensure_schema(chosen)
        try:
            with sqlite3.connect(f"file:{chosen}?mode=ro", uri=True) as conn:
                if conn.execute("SELECT 1 FROM meta WHERE key='demo_snapshot'").fetchone():
                    os.environ["OSRL_DEMO"] = "1"
        except sqlite3.Error:
            pass
        # Keep auto-selected demo mode stable over Streamlit reruns too.
        if chosen.resolve() == (_work_dir() / "demo.db").resolve():
            os.environ["OSRL_DEMO"] = "1"
        return str(chosen)

    local_live = root / "data" / "telemetry.db"
    if local_live.exists() and local_live.stat().st_size > 0:
        return str(local_live)

    demo_src = root / "data" / "demo_telemetry.db"
    work = _work_dir()
    work.mkdir(parents=True, exist_ok=True)
    if demo_src.exists() and demo_src.stat().st_size > 0:
        # The Cloud filesystem is ephemeral; the snapshot is copied so mode
        # changes and rollup writes never touch the repository.
        demo_dst = work / "demo.db"
        stale = (
            not demo_dst.exists()
            or demo_dst.stat().st_size != demo_src.stat().st_size
            or int(demo_dst.stat().st_mtime) != int(demo_src.stat().st_mtime)
        )
        if stale:
            # Atomic replace so concurrent first sessions never read a partial file.
            tmp = work / f".demo.{uuid.uuid4().hex}.tmp"
            shutil.copy2(demo_src, tmp)
            os.replace(tmp, demo_dst)
        os.environ["OSRL_DEMO"] = "1"
        return str(demo_dst)

    empty = work / "empty.db"
    if not empty.exists() or empty.stat().st_size == 0:
        ensure_schema(empty)
    return str(empty)
