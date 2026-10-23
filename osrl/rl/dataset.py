"""Chronological replay splits and coverage, without importing the ML stack."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass

from osrl.twin import sessions_from_rows


@dataclass
class ReplaySplit:
    train: list[list]
    evaluation: list[list]
    summary: dict


def split_rows(rows: list, train_fraction: float = 0.8, window: int = 10) -> ReplaySplit:
    """Split on time, with a full decision-window embargo at the boundary.

    Splitting within a long session is allowed, but no window or observation
    crosses the split or a telemetry outage. Tiny fragments are discarded.
    """
    if not 0 < train_fraction < 1 or window < 1:
        raise ValueError("train_fraction must be in (0, 1) and window must be positive")
    rows = sorted(rows, key=lambda r: float(r["ts"]))
    if len(rows) < window * 5:
        raise ValueError("Need at least five decision windows of real telemetry")
    cut = int(len(rows) * train_fraction)
    train_end = float(rows[cut - 1]["ts"])
    eval_start = float(rows[min(cut + window, len(rows) - 1)]["ts"])
    train = [
        s.rows
        for s in sessions_from_rows([r for r in rows if r["ts"] <= train_end])
        if len(s.rows) >= window * 2
    ]
    evaluation = [
        s.rows
        for s in sessions_from_rows([r for r in rows if r["ts"] >= eval_start])
        if len(s.rows) >= window * 2
    ]
    if not train or not evaluation:
        raise ValueError("Insufficient contiguous real data for separate training and evaluation")
    return ReplaySplit(
        train,
        evaluation,
        {
            "train_samples": sum(map(len, train)),
            "eval_samples": sum(map(len, evaluation)),
            "train_sessions": len(train),
            "eval_sessions": len(evaluation),
            "train_end_ts": train[-1][-1]["ts"],
            "eval_start_ts": evaluation[0][0]["ts"],
            "data_end_ts": evaluation[-1][-1]["ts"],
            "embargo_samples": window,
        },
    )


def load_rows(db_path: str) -> list[dict]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM samples ORDER BY ts, id")]


def quality_summary(rows: list) -> dict:
    """Coverage and observed stalls; missing and non-finite readings cast no vote."""

    def finite(v):
        return v is not None and math.isfinite(float(v))

    valid = [
        r
        for r in rows
        if finite(r["interval_s"]) and r["interval_s"] > 0 and finite(r["psi_mem_some_stall_us"])
    ]
    pressure = sum(r["psi_mem_some_stall_us"] / (r["interval_s"] * 1e6) >= 0.2 for r in valid)
    return {
        "samples": len(rows),
        "psi_valid_samples": len(valid),
        "psi_coverage": len(valid) / len(rows) if rows else 0.0,
        "pressure_samples": pressure,
        "sessions": len(sessions_from_rows(rows)) if rows else 0,
    }


def fingerprint(rows: list) -> str:
    """Content identity for reproducible evaluation after retention or edits."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(dict(row), sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()
