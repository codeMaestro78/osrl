"""Auto-training scheduler for OS-RL-Agent.

Runs as a background daemon that watches the telemetry DB and automatically
triggers PPO training when conditions are met:

  1. Enough NEW samples have been collected since the last training run
     (default: MIN_SAMPLES_FOR_TRAINING = 500).
  2. Enough wall-clock time has passed since the last training run
     (default: MIN_TRAIN_INTERVAL_S = 3600 = 1 hour).
  3. No training process is already running.

On each triggered run:
  - Runs at os.nice(15) — lowest possible priority.
  - Saves checkpoints, writes rl_train_* events to the DB.
  - Writes auto_train_schedule events so the dashboard can show next-run time.

Usage:
    python -m osrl.rl.auto_train --db ~/os-rl-agent/data/telemetry.db
    osrl-auto-train --db ~/os-rl-agent/data/telemetry.db [--steps 50000]
                    [--interval 3600] [--min-samples 500] [--synthetic]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s auto_train] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("auto_train")

# ── Defaults ──────────────────────────────────────────────────────────────────
MIN_SAMPLES_FOR_TRAINING = 500    # samples since last run before we trigger
MIN_TRAIN_INTERVAL_S     = 3600   # minimum 1 hour between runs
POLL_INTERVAL_S          = 60     # check DB every 60 seconds
DEFAULT_STEPS            = 50_000
DEFAULT_CHECKPOINT_DIR   = Path("~/os-rl-agent/checkpoints").expanduser()

_STOP = False


def _handle_signal(sig, _frame):
    global _STOP
    log.info("Received signal %s — stopping after current poll.", sig)
    _STOP = True


def _write_event(conn, kind: str, payload: dict) -> None:
    try:
        conn.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(payload, sort_keys=True)),
        )
        conn.commit()
    except Exception:  # noqa: BLE001
        pass


def _sample_count(conn) -> int:
    try:
        return conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    except Exception:  # noqa: BLE001
        return 0


def _last_train_ts(conn) -> float:
    """Timestamp of the last rl_train_done event (0 if never trained)."""
    try:
        row = conn.execute(
            "SELECT ts FROM events WHERE kind='rl_train_done' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return float(row[0]) if row else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _samples_since(conn, since_ts: float) -> int:
    """How many samples have been collected after a given timestamp."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM samples WHERE ts > ?", (since_ts,)
        ).fetchone()
        return int(row[0]) if row else 0
    except Exception:  # noqa: BLE001
        return 0


def _is_training_running() -> bool:
    """Check if an osrl-train / rl.train process is already running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "osrl.rl.train|osrl-train"],
            capture_output=True, text=True,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _launch_training(
    db_path: str,
    steps: int,
    synthetic: bool,
    checkpoint_dir: Path,
    resume: bool,
) -> subprocess.Popen:
    """Launch training as a detached subprocess."""
    best_link = checkpoint_dir / "best"
    cmd = [
        sys.executable, "-m", "osrl.rl.train",
        "--db", db_path,
        "--steps", str(steps),
        "--checkpoint-dir", str(checkpoint_dir),
    ]
    if synthetic:
        cmd.append("--synthetic")
    if resume and best_link.exists():
        cmd.extend(["--resume", str(best_link)])

    log.info("Launching training: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return proc


def run_scheduler(
    db_path: str,
    steps: int = DEFAULT_STEPS,
    synthetic: bool = True,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    min_samples: int = MIN_SAMPLES_FOR_TRAINING,
    min_interval_s: float = MIN_TRAIN_INTERVAL_S,
    poll_s: float = POLL_INTERVAL_S,
    resume: bool = True,
) -> None:
    """Main scheduler loop — runs forever until SIGTERM/SIGINT."""
    import sqlite3

    global _STOP
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    checkpoint_dir = Path(checkpoint_dir).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Lower own priority so we never interfere with desktop
    try:
        os.nice(5)
    except OSError:
        pass

    log.info("Auto-trainer started. DB=%s  steps=%d  min_samples=%d  interval=%ds",
             db_path, steps, min_samples, int(min_interval_s))

    active_proc: subprocess.Popen | None = None
    db_path_expanded = str(Path(db_path).expanduser())

    while not _STOP:
        try:
            conn = sqlite3.connect(db_path_expanded, timeout=10)
            conn.row_factory = sqlite3.Row

            # ── Check if active training subprocess is still alive ─────────
            if active_proc is not None:
                ret = active_proc.poll()
                if ret is None:
                    log.info("Training still running (PID %d) …", active_proc.pid)
                    _write_event(conn, "auto_train_running", {
                        "pid": active_proc.pid,
                        "next_check_in": int(poll_s),
                    })
                    conn.close()
                    time.sleep(poll_s)
                    continue
                else:
                    log.info("Training subprocess finished (exit=%d).", ret)
                    active_proc = None

            # ── Eligibility checks ────────────────────────────────────────
            now            = time.time()
            last_train     = _last_train_ts(conn)
            time_since     = now - last_train
            n_new_samples  = _samples_since(conn, last_train)
            total_samples  = _sample_count(conn)
            already_running= _is_training_running()

            next_run_in    = max(0.0, min_interval_s - time_since)
            needed_samples = max(0, min_samples - n_new_samples)

            log.info(
                "State: total=%d  new_since_last_train=%d/%d  "
                "time_since_last=%.0fs/%.0fs  running=%s",
                total_samples, n_new_samples, min_samples,
                time_since, min_interval_s, already_running,
            )

            # Write schedule status so dashboard can show it
            _write_event(conn, "auto_train_status", {
                "total_samples":   total_samples,
                "new_samples":     n_new_samples,
                "needed_samples":  needed_samples,
                "time_since_last": int(time_since),
                "next_run_in_s":   int(next_run_in),
                "already_running": already_running,
                "scheduler_ts":    now,
            })

            if already_running:
                log.info("Training already running (external). Skipping trigger.")
            elif n_new_samples >= min_samples and time_since >= min_interval_s:
                log.info(
                    "✅ Conditions met — triggering training! "
                    "new_samples=%d  time_since=%.0fs",
                    n_new_samples, time_since,
                )
                _write_event(conn, "auto_train_triggered", {
                    "new_samples": n_new_samples,
                    "time_since_last": int(time_since),
                    "steps": steps,
                    "synthetic": synthetic,
                })
                conn.close()
                active_proc = _launch_training(
                    db_path_expanded, steps, synthetic, checkpoint_dir, resume
                )
                time.sleep(poll_s)
                continue
            else:
                reasons = []
                if n_new_samples < min_samples:
                    reasons.append(f"need {needed_samples} more samples")
                if time_since < min_interval_s:
                    reasons.append(f"wait {next_run_in:.0f}s more")
                log.info("Not yet eligible: %s", ", ".join(reasons))

            conn.close()

        except Exception as exc:  # noqa: BLE001
            log.warning("Scheduler poll error: %s", exc)

        time.sleep(poll_s)

    log.info("Auto-trainer stopped.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="osrl-auto-train",
        description="Auto-training scheduler — triggers PPO training when enough data is collected.",
    )
    p.add_argument("--db", required=True, help="SQLite telemetry DB path")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                   help=f"PPO training steps per run (default {DEFAULT_STEPS:,})")
    p.add_argument("--synthetic", action="store_true",
                   help="Enable synthetic stress injection")
    p.add_argument("--interval", type=int, default=int(MIN_TRAIN_INTERVAL_S),
                   help="Minimum seconds between training runs (default 3600 = 1h)")
    p.add_argument("--min-samples", type=int, default=MIN_SAMPLES_FOR_TRAINING,
                   help=f"New samples needed to trigger training (default {MIN_SAMPLES_FOR_TRAINING})")
    p.add_argument("--poll", type=int, default=int(POLL_INTERVAL_S),
                   help=f"DB poll interval in seconds (default {POLL_INTERVAL_S})")
    p.add_argument("--no-resume", action="store_true",
                   help="Always train from scratch, never resume best checkpoint")
    p.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    args = p.parse_args(argv)

    run_scheduler(
        db_path=args.db,
        steps=args.steps,
        synthetic=args.synthetic,
        checkpoint_dir=Path(args.checkpoint_dir),
        min_samples=args.min_samples,
        min_interval_s=float(args.interval),
        poll_s=float(args.poll),
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
