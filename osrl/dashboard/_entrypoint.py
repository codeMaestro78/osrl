"""Thin CLI entrypoint — launches Streamlit without importing app.py at module level."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Launch the Streamlit dashboard as a subprocess."""
    import argparse
    import subprocess

    p = argparse.ArgumentParser(prog="osrl-dashboard")
    p.add_argument("--db", default="~/os-rl-agent/data/telemetry.db",
                   help="SQLite DB path")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--host", default="localhost")
    args = p.parse_args(argv)

    app_path = Path(__file__).parent / "app.py"
    env = {**os.environ, "OSRL_DB": str(Path(args.db).expanduser())}
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true",
        "--theme.base", "dark",
        "--theme.primaryColor", "#38bdf8",
        "--theme.backgroundColor", "#0d1117",
        "--theme.secondaryBackgroundColor", "#1e293b",
        "--theme.textColor", "#e2e8f0",
    ]
    print(f"[osrl-dashboard] Launching at http://{args.host}:{args.port}")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
