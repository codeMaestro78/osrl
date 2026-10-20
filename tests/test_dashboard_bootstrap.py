"""Deployment bootstrap tests: DB selection order for Streamlit Cloud/local."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from osrl.dashboard import bootstrap


def _set_work_dir(monkeypatch, tmp_path: Path) -> Path:
    work = tmp_path / "work"
    monkeypatch.setattr(bootstrap, "_work_dir", lambda: work)
    return work


def test_explicit_osrl_db_wins_and_is_initialized(tmp_path, monkeypatch):
    monkeypatch.delenv("OSRL_DEMO", raising=False)
    target = tmp_path / "nested" / "live.db"
    monkeypatch.setenv("OSRL_DB", str(target))

    resolved = bootstrap.bootstrap_database(tmp_path)

    assert Path(resolved) == target
    assert target.exists()
    assert os.environ.get("OSRL_DEMO") is None
    conn = sqlite3.connect(target)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
    assert conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0


def test_local_live_db_takes_precedence_over_demo(tmp_path, monkeypatch):
    monkeypatch.delenv("OSRL_DB", raising=False)
    monkeypatch.delenv("OSRL_DEMO", raising=False)
    _set_work_dir(monkeypatch, tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "telemetry.db").write_bytes(b"live-db-placeholder")
    (data / "demo_telemetry.db").write_bytes(b"demo-db-placeholder")

    resolved = bootstrap.bootstrap_database(tmp_path)

    assert Path(resolved) == data / "telemetry.db"
    assert os.environ.get("OSRL_DEMO") is None


def test_demo_snapshot_is_copied_to_writable_work_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("OSRL_DB", raising=False)
    monkeypatch.delenv("OSRL_DEMO", raising=False)
    work = _set_work_dir(monkeypatch, tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "demo_telemetry.db").write_bytes(b"snapshot-bytes" * 8)

    resolved = bootstrap.bootstrap_database(tmp_path)

    assert Path(resolved) == work / "demo.db"
    assert Path(resolved).read_bytes() == (data / "demo_telemetry.db").read_bytes()
    assert os.environ.get("OSRL_DEMO") == "1"

    # A second call reuses the copy and keeps demo mode sticky for the process.
    assert Path(bootstrap.bootstrap_database(tmp_path)) == Path(resolved)
    assert os.environ.get("OSRL_DEMO") == "1"


def test_missing_data_bootstraps_empty_schema(tmp_path, monkeypatch):
    monkeypatch.delenv("OSRL_DB", raising=False)
    monkeypatch.delenv("OSRL_DEMO", raising=False)
    work = _set_work_dir(monkeypatch, tmp_path)

    resolved = Path(bootstrap.bootstrap_database(tmp_path))

    assert resolved == work / "empty.db"
    assert os.environ.get("OSRL_DEMO") is None
    conn = sqlite3.connect(resolved)
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"samples", "events", "rollups", "actions_ledger", "intents"} <= tables
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3
