"""SQLite WAL store: telemetry, rollups, events, heartbeat, intent channel.

Operation rules implemented here (brief G):

- WAL journal mode + synchronous=NORMAL (documented-safe pairing per the
  SQLite WAL paper); explicit short transactions only, batched writes.
  Callers pass complete batches so each transaction stays short.
- Intent states are explicit: pending -> claimed -> applied/rejected/failed,
  or claimed-again via lease expiry -> pending -> ... -> expired. Claim and
  finish use compare-and-set UPDATEs inside BEGIN IMMEDIATE transactions, so
  retries and competing workers are safe (no lost/double intents).
- `sample_staleness` is the primitive later risk gates will use: stale or
  missing telemetry may NOT enable intervention behavior downstream.
- Indexes on timestamps; retention prunes raw telemetry and transient events.
  Schema changes are tracked with PRAGMA user_version.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from osrl.collector import Sample

_INTENT_OPEN_STATES = ("pending", "claimed")
_INTENT_FINAL_STATES = ("applied", "rejected", "failed", "expired")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    interval_s REAL, ncpu INTEGER,
    cpu_total_pct REAL, cpu_percpu_json TEXT,
    load1 REAL, load5 REAL, load15 REAL,
    mem_total_bytes INTEGER, mem_available_bytes INTEGER, mem_used_pct REAL,
    swap_total_bytes INTEGER, swap_used_bytes INTEGER, swap_used_pct REAL,
    swp_in_ps REAL, swp_out_ps REAL, majflt_ps REAL,
    reclaim_scan_ps REAL, reclaim_steal_ps REAL, allocstall_ps REAL,
    psi_cpu_some_total_us INTEGER, psi_cpu_some_stall_us INTEGER,
    psi_cpu_full_total_us INTEGER, psi_cpu_full_stall_us INTEGER,
    psi_mem_some_total_us INTEGER, psi_mem_some_stall_us INTEGER,
    psi_mem_full_total_us INTEGER, psi_mem_full_stall_us INTEGER,
    psi_io_some_total_us INTEGER, psi_io_some_stall_us INTEGER,
    psi_io_full_total_us INTEGER, psi_io_full_stall_us INTEGER,
    gpu_util_pct REAL, gpu_mem_used_mib REAL, gpu_mem_total_mib REAL,
    gpu_temp_c REAL, gpu_power_w REAL,
   gpu_top_procs_json TEXT,
    top_procs_json TEXT, collect_ms REAL
);
CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts);

CREATE TABLE IF NOT EXISTS rollups (
    window_start REAL PRIMARY KEY,
    window_end REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    cpu_total_pct_avg REAL, cpu_total_pct_max REAL,
    mem_used_pct_avg REAL, mem_available_bytes_avg REAL,
    swap_used_pct_avg REAL, swap_used_pct_max REAL,
    swp_out_ps_avg REAL, swp_out_ps_max REAL,
    psi_mem_stall_s_sum REAL, psi_io_stall_s_sum REAL,
    extra_json TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS heartbeat (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    ts REAL NOT NULL,
    pid INTEGER,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS intents (
    id TEXT PRIMARY KEY,
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','claimed','applied','rejected','failed','expired')),
    claimed_by TEXT,
    claimed_ts REAL,
    tries INTEGER NOT NULL DEFAULT 0,
    result TEXT
);

CREATE TABLE IF NOT EXISTS actions_ledger (
    id TEXT PRIMARY KEY,
    created_ts REAL NOT NULL,
    proposed_by TEXT NOT NULL,
    kind TEXT NOT NULL,
    target_pid INTEGER,
    target_start_time REAL,
    target_nice INTEGER,
    before_nice INTEGER,
    applied_nice INTEGER,
    parent_action_id TEXT,
    intent_id TEXT,
    guard_checks TEXT,
    payload TEXT,
    error TEXT,
    status TEXT NOT NULL DEFAULT 'proposed'
        CHECK (status IN ('proposed','approved','rejected','applied','failed',
                          'restored','expired')),
    ts_applied REAL,
    ts_restored REAL
);
CREATE INDEX IF NOT EXISTS idx_actions_pid_ts
    ON actions_ledger(target_pid, created_ts);

CREATE TABLE IF NOT EXISTS action_outcomes (
    intent_id TEXT PRIMARY KEY,
    action_id TEXT UNIQUE,
    created_ts REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN
        ('pending','measured','insufficient_telemetry','not_applied')),
    details TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_action_outcomes_ts ON action_outcomes(created_ts);
"""

# Columns added in schema v2 (idempotent ALTER list for existing v1 dbs).
_V2_ACTION_COLUMNS = (
    "target_nice INTEGER", "before_nice INTEGER", "applied_nice INTEGER",
    "parent_action_id TEXT", "intent_id TEXT", "guard_checks TEXT",
    "error TEXT", "ts_applied REAL", "ts_restored REAL",
)
_V2_SAMPLE_COLUMNS = ("gpu_top_procs_json TEXT",)


# Schema v3 additions: RL metrics table + agent mode meta key
_V3_ADDITIONS = """
CREATE TABLE IF NOT EXISTS rl_metrics (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    checkpoint_id TEXT,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rl_metrics_ts ON rl_metrics(ts);
"""


class Store:
    SCHEMA_VERSION = 4

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.path), timeout=5.0, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        mode = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            self._conn.close()
            raise RuntimeError(f"SQLite WAL unavailable in {self.path}: got {mode!r}")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        self._migrate(self._conn)
        self._conn.executescript(_V3_ADDITIONS)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotent version bump to SCHEMA_VERSION.

        v1 -> v2: adds actions_ledger columns (before they were dead schema)
        and samples.gpu_top_procs_json; existing dbs are ALTERed in place,
        fresh dbs already carry the v2 layout and need only the marker.
        v2 -> v3: adds rl_metrics table + agent_mode default meta row.
        v3 -> v4: adds durable intervention outcome records.
        An unknown FUTURE db version is a hard error (defensive).
        The intent-column index lives here (not in _SCHEMA) because legacy
        tables cannot take an index on a column they don't have yet.
        """
        self._recreate_ledger_indexes(conn)
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > self.SCHEMA_VERSION:
            raise RuntimeError(
                f"db {self.path} has schema v{version}, newer than supported "
                f"v{self.SCHEMA_VERSION}"
            )
        if version == self.SCHEMA_VERSION:
            return
        table_cols = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(actions_ledger)"
            ).fetchall()
        }
        for coldef in _V2_ACTION_COLUMNS:
            name = coldef.split()[0]
            if name not in table_cols:
                conn.execute(
                    f"ALTER TABLE actions_ledger ADD COLUMN {coldef}"
                )
        sample_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        for coldef in _V2_SAMPLE_COLUMNS:
            name = coldef.split()[0]
            if name not in sample_cols:
                conn.execute(f"ALTER TABLE samples ADD COLUMN {coldef}")
        self._recreate_ledger_indexes(conn)
        # v3: rl_metrics is created by _V3_ADDITIONS (executescript after migrate)
        # Insert default agent_mode if absent
        conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('agent_mode', 'observe')"
        )
        conn.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")

    def _recreate_ledger_indexes(self, conn: sqlite3.Connection) -> None:
        # created lazily because legacy CREATE prevented early ALTERs
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_actions_pid_ts"
                " ON actions_ledger(target_pid, created_ts);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_actions_intent"
                " ON actions_ledger(intent_id);"
            )
        except sqlite3.Error:
            pass  # indexes are performance-only

    def wal_checkpoint(self, mode: str = "PASSIVE") -> None:
        try:
            self._conn.execute(f"PRAGMA wal_checkpoint({mode})")
        except sqlite3.Error:
            pass  # checkpoint failure must never kill sampling

    def close(self) -> None:
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error:
            pass
        self._conn.close()

    # ---------------------------------------------------------- transaction

    class _Txn:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def __enter__(self) -> sqlite3.Connection:
            self._conn.execute("BEGIN IMMEDIATE")
            return self._conn

        def __exit__(self, exc_type, exc, tb) -> None:
            if exc_type is None:
                self._conn.execute("COMMIT")
            else:
                self._conn.execute("ROLLBACK")

    def _txn(self) -> Store._Txn:
        return Store._Txn(self._conn)

    # ------------------------------------------------------------- samples

    def insert_samples(self, batch: list[Sample]) -> int:
        if not batch:
            return 0
        rows = [_sample_to_row(s) for s in batch]
        with self._txn() as conn:
            conn.executemany(
                "INSERT INTO samples ("
                " ts, interval_s, ncpu, cpu_total_pct, cpu_percpu_json,"
                " load1, load5, load15,"
                " mem_total_bytes, mem_available_bytes, mem_used_pct,"
                " swap_total_bytes, swap_used_bytes, swap_used_pct,"
                " swp_in_ps, swp_out_ps, majflt_ps,"
                " reclaim_scan_ps, reclaim_steal_ps, allocstall_ps,"
                " psi_cpu_some_total_us, psi_cpu_some_stall_us,"
                " psi_cpu_full_total_us, psi_cpu_full_stall_us,"
                " psi_mem_some_total_us, psi_mem_some_stall_us,"
                " psi_mem_full_total_us, psi_mem_full_stall_us,"
                " psi_io_some_total_us, psi_io_some_stall_us,"
                " psi_io_full_total_us, psi_io_full_stall_us,"
                " gpu_util_pct, gpu_mem_used_mib, gpu_mem_total_mib,"
                " gpu_temp_c, gpu_power_w,"
                " gpu_top_procs_json, top_procs_json, collect_ms"
                ") VALUES ("
                ":ts, :interval_s, :ncpu, :cpu_total_pct, :cpu_percpu_json,"
                ":load1, :load5, :load15,"
                ":mem_total_bytes, :mem_available_bytes, :mem_used_pct,"
                ":swap_total_bytes, :swap_used_bytes, :swap_used_pct,"
                ":swp_in_ps, :swp_out_ps, :majflt_ps,"
                ":reclaim_scan_ps, :reclaim_steal_ps, :allocstall_ps,"
                ":psi_cpu_some_total_us, :psi_cpu_some_stall_us,"
                ":psi_cpu_full_total_us, :psi_cpu_full_stall_us,"
                ":psi_mem_some_total_us, :psi_mem_some_stall_us,"
                ":psi_mem_full_total_us, :psi_mem_full_stall_us,"
                ":psi_io_some_total_us, :psi_io_some_stall_us,"
                ":psi_io_full_total_us, :psi_io_full_stall_us,"
                ":gpu_util_pct, :gpu_mem_used_mib, :gpu_mem_total_mib,"
                ":gpu_temp_c, :gpu_power_w,"
                ":gpu_top_procs_json, :top_procs_json, :collect_ms)",
                rows,
            )
        return len(rows)

    def count_samples(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM samples").fetchone()
        return int(row["n"])

    def recent_samples(self, limit: int = 300) -> list[sqlite3.Row]:
        """Return the latest telemetry rows in chronological order."""
        if limit < 1 or limit > 5000:
            raise ValueError("sample limit must be between 1 and 5000")
        rows = self._conn.execute(
            "SELECT * FROM samples ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return list(reversed(rows))

    def latest_sample_ts(self) -> float | None:
        row = self._conn.execute("SELECT MAX(ts) AS m FROM samples").fetchone()
        return None if row["m"] is None else float(row["m"])

    def sample_staleness(self, now: float | None = None) -> float | None:
        """Seconds since the newest sample; None = no samples at all. This is
        the foundation of the 'stale telemetry -> abstain' gate (needs the
        risk layer, Phase 1, to actually enforce it)."""
        latest = self.latest_sample_ts()
        if latest is None:
            return None
        now = time.time() if now is None else now
        return now - latest

    # ------------------------------------------------------ rollups/retention

    def rollup_until(self) -> float | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='rollup_until'"
        ).fetchone()
        return None if row is None else float(row["value"])

    def set_rollup_until(self, ts: float) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES ('rollup_until', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (repr(ts),),
        )

    def compute_rollup(self, window_start: float, window_end: float) -> dict | None:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n,"
            " AVG(cpu_total_pct) AS cpu_avg, MAX(cpu_total_pct) AS cpu_max,"
            " AVG(mem_used_pct) AS mem_pct_avg,"
            " AVG(mem_available_bytes) AS mem_avail_avg,"
            " AVG(swap_used_pct) AS swap_pct_avg, MAX(swap_used_pct) AS swap_pct_max,"
            " AVG(swp_out_ps) AS swp_out_avg, MAX(swp_out_ps) AS swp_out_max,"
            " SUM(psi_mem_some_stall_us) AS psi_mem_sum,"
            " SUM(psi_io_some_stall_us) AS psi_io_sum"
            " FROM samples WHERE ts >= ? AND ts < ?",
            (window_start, window_end),
        ).fetchone()
        if row["n"] == 0:
            return None
        psi_mem = row["psi_mem_sum"]
        psi_io = row["psi_io_sum"]
        return {
            "window_start": window_start,
            "window_end": window_end,
            "sample_count": int(row["n"]),
            "cpu_total_pct_avg": row["cpu_avg"],
            "cpu_total_pct_max": row["cpu_max"],
            "mem_used_pct_avg": row["mem_pct_avg"],
            "mem_available_bytes_avg": row["mem_avail_avg"],
            "swap_used_pct_avg": row["swap_pct_avg"],
            "swap_used_pct_max": row["swap_pct_max"],
            "swp_out_ps_avg": row["swp_out_avg"],
            "swp_out_ps_max": row["swp_out_max"],
            # us -> seconds, plainly labeled: reward/experiments consume stall
            # seconds, never swap occupancy, per brief E.
            "psi_mem_stall_s_sum": None if psi_mem is None else psi_mem / 1e6,
            "psi_io_stall_s_sum": None if psi_io is None else psi_io / 1e6,
        }

    def insert_rollup(self, agg: dict) -> None:
        self._conn.execute(
            "INSERT INTO rollups ("
            " window_start, window_end, sample_count,"
            " cpu_total_pct_avg, cpu_total_pct_max,"
            " mem_used_pct_avg, mem_available_bytes_avg,"
            " swap_used_pct_avg, swap_used_pct_max,"
            " swp_out_ps_avg, swp_out_ps_max,"
            " psi_mem_stall_s_sum, psi_io_stall_s_sum)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(window_start) DO UPDATE SET"
            " window_end=excluded.window_end, sample_count=excluded.sample_count,"
            " cpu_total_pct_avg=excluded.cpu_total_pct_avg,"
            " cpu_total_pct_max=excluded.cpu_total_pct_max,"
            " mem_used_pct_avg=excluded.mem_used_pct_avg,"
            " mem_available_bytes_avg=excluded.mem_available_bytes_avg,"
            " swap_used_pct_avg=excluded.swap_used_pct_avg,"
            " swap_used_pct_max=excluded.swap_used_pct_max,"
            " swp_out_ps_avg=excluded.swp_out_ps_avg,"
            " swp_out_ps_max=excluded.swp_out_ps_max,"
            " psi_mem_stall_s_sum=excluded.psi_mem_stall_s_sum,"
            " psi_io_stall_s_sum=excluded.psi_io_stall_s_sum",
            (
                agg["window_start"], agg["window_end"], agg["sample_count"],
                agg["cpu_total_pct_avg"], agg["cpu_total_pct_max"],
                agg["mem_used_pct_avg"], agg["mem_available_bytes_avg"],
                agg["swap_used_pct_avg"], agg["swap_used_pct_max"],
                agg["swp_out_ps_avg"], agg["swp_out_ps_max"],
                agg["psi_mem_stall_s_sum"], agg["psi_io_stall_s_sum"],
            ),
        )

    def prune(self, now: float, raw_days: int, events_days: int) -> dict[str, int]:
        raw_cutoff = now - raw_days * 86400.0
        ev_cutoff = now - events_days * 86400.0
        with self._txn() as conn:
            cur_s = conn.execute(
                "DELETE FROM samples WHERE ts < ?", (raw_cutoff,)
            )
            cur_e = conn.execute(
                "DELETE FROM events WHERE ts < ?", (ev_cutoff,)
            )
            cur_l = conn.execute(
                "DELETE FROM actions_ledger WHERE created_ts < ?", (ev_cutoff,)
            )
            return {
                "samples_deleted": cur_s.rowcount,
                "events_deleted": cur_e.rowcount,
                "actions_deleted": cur_l.rowcount,
            }

    # -------------------------------------------------- events / heartbeat

    def record_event(self, ts: float, kind: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (ts, kind, json.dumps(payload, sort_keys=True)),
        )

    def recent_events(self, limit: int = 50) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT ts, kind, payload FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def heartbeat(self, ts: float, pid: int, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO heartbeat (id, ts, pid, payload) VALUES (1, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET ts=excluded.ts,"
            " pid=excluded.pid, payload=excluded.payload",
            (ts, pid, json.dumps(payload, sort_keys=True)),
        )

    def heartbeat_age(self, now: float | None = None) -> float:
        """Seconds since heartbeat; inf when none exists. The dashboard's
        red-banner logic keys off this (Phase 5)."""
        row = self._conn.execute("SELECT ts FROM heartbeat WHERE id=1").fetchone()
        if row is None:
            return math.inf
        now = time.time() if now is None else now
        return now - float(row["ts"])

    # --------------------------------------------------------------- intents

    # ------------------------------------------------------- actions ledger

    _ACTION_COLUMNS = frozenset({
        "id", "created_ts", "proposed_by", "kind",
        "target_pid", "target_start_time", "target_nice", "before_nice",
        "applied_nice", "parent_action_id", "intent_id", "guard_checks",
        "payload", "error", "status", "ts_applied", "ts_restored",
    })

    def insert_action(self, **cols) -> bool:
        """Strict-column insert; unknown keys are a programming error."""
        unknown = set(cols) - self._ACTION_COLUMNS
        if unknown:
            raise ValueError(f"unknown ledger columns: {sorted(unknown)}")
        fields = sorted(cols)
        sql = (
            f"INSERT INTO actions_ledger ({', '.join(fields)})"
            f" VALUES ({', '.join(':' + f for f in fields)})"
        )
        try:
            with self._txn() as conn:
                conn.execute(sql, {f: cols[f] for f in fields})
            return True
        except sqlite3.IntegrityError:
            return False

    def mark_action_applied(self, id: str, ts: float, applied_nice: int) -> bool:
        with self._txn() as conn:
            cur = conn.execute(
                "UPDATE actions_ledger SET status='applied', applied_nice=?,"
                " ts_applied=? WHERE id=? AND status IN ('proposed','approved')",
                (applied_nice, ts, id),
            )
            return cur.rowcount == 1

    def mark_action_failed(self, id: str, ts: float, error: str) -> bool:
        with self._txn() as conn:
            cur = conn.execute(
                "UPDATE actions_ledger SET status='failed', error=?,"
                " ts_restored=? WHERE id=? AND status IN ('proposed','approved')",
                (error, ts, id),
            )
            return cur.rowcount == 1
    def action_row(self, id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM actions_ledger WHERE id=?", (id,)
        ).fetchone()
        return None if row is None else dict(row)

    def last_action_ts(self, pid: int, kind: str) -> float | None:
        """Most recent ledger entry (any state) for (pid, kind): the rate
        limiter counts every attempt so a spam loop cannot retry itself
        into a per-millisecond renice."""
        row = self._conn.execute(
            "SELECT MAX(created_ts) AS m FROM actions_ledger"
            " WHERE target_pid=? AND kind=?",
            (pid, kind),
        ).fetchone()
        return None if row["m"] is None else float(row["m"])

    def actions_for_intent(self, intent_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM actions_ledger WHERE intent_id=? ORDER BY created_ts",
            (intent_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_action_outcome(
        self,
        intent_id: str,
        action_id: str | None,
        status: str,
        ts: float,
        details: dict[str, Any],
    ) -> bool:
        """Persist one attempt; applied actions wait for their post-window."""
        if status not in {"applied", "rejected", "failed"}:
            raise ValueError(f"invalid action outcome status: {status!r}")
        stored_status = "pending" if status == "applied" else "not_applied"
        payload = {**details, "attempt_status": status}
        try:
            with self._txn() as conn:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO action_outcomes"
                    " (intent_id, action_id, created_ts, status, details)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (intent_id, action_id, ts, stored_status,
                     json.dumps(payload, sort_keys=True)),
                )
            return cur.rowcount == 1
        except sqlite3.IntegrityError:
            return False

    def finalize_action_outcomes(
        self, now: float, limit: int = 50, risk_config=None,
        flush_grace_s: float = 15.0,
    ) -> list[dict]:
        """Measure matured applied attempts against their fixed telemetry window."""
        from osrl.outcomes import (
            DEFAULT_POST_WINDOW_S,
            DEFAULT_PRE_WINDOW_S,
            DEFAULT_WARMUP_S,
            measure_action_outcome,
        )
        if not math.isfinite(flush_grace_s) or flush_grace_s < 0:
            raise ValueError("flush_grace_s must be a finite non-negative number")

        rows = self._conn.execute(
            "SELECT o.intent_id, o.action_id, o.created_ts, o.details,"
            " a.target_pid, a.ts_applied, a.kind"
            " FROM action_outcomes o LEFT JOIN actions_ledger a ON a.id=o.action_id"
            " WHERE o.status='pending' AND o.created_ts + ? + ? <= ?"
            " ORDER BY o.created_ts, o.intent_id LIMIT ?",
            (DEFAULT_POST_WINDOW_S, flush_grace_s, now, limit),
        ).fetchall()
        finalized = []
        for row in rows:
            action = dict(row)
            applied_ts = action.get("ts_applied")
            if applied_ts is None:
                measurement = {
                    "status": "insufficient_telemetry",
                    "reason": "missing_action_ledger_or_apply_timestamp",
                    "foreground_impact": None,
                    "foreground_impact_note": (
                        "Active foreground application identity is not collected."
                    ),
                }
            else:
                sample_rows = self._conn.execute(
                    "SELECT * FROM samples WHERE ts >= ? AND ts <= ? ORDER BY ts",
                    (applied_ts - DEFAULT_PRE_WINDOW_S - DEFAULT_WARMUP_S,
                     applied_ts + DEFAULT_POST_WINDOW_S),
                ).fetchall()
                measurement = measure_action_outcome(
                    action,
                    [dict(sample) for sample in sample_rows],
                    risk_config=risk_config,
                )
            status = measurement.pop("status")
            details = json.loads(row["details"] or "{}")
            details["measurement"] = measurement
            with self._txn() as conn:
                changed = conn.execute(
                    "UPDATE action_outcomes SET status=?, details=?"
                    " WHERE intent_id=? AND status='pending'",
                    (status, json.dumps(details, sort_keys=True), row["intent_id"]),
                ).rowcount
            if changed:
                finalized.append({"intent_id": row["intent_id"], "status": status})
        return finalized

    def recent_action_outcomes(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT intent_id, action_id, created_ts, status, details"
            " FROM action_outcomes ORDER BY created_ts DESC, intent_id LIMIT ?",
            (limit,),
        ).fetchall()
        out = []
        for row in rows:
            details = json.loads(row["details"] or "{}")
            measurement = details.get("measurement", {})
            out.append({
                **dict(row),
                "details_json": row["details"],
                "details": details,
                "measured": row["status"] == "measured",
                **measurement,
            })
        return out

    def relevant_outcome_count(self, action: str, risk_level: str) -> int:
        """Count measured outcomes for the same policy action and risk band."""
        rows = self._conn.execute(
            "SELECT details FROM action_outcomes WHERE status='measured'"
        ).fetchall()
        count = 0
        for row in rows:
            try:
                context = json.loads(row["details"]).get("recommendation_context", {})
            except (TypeError, ValueError):
                continue
            if context.get("action") == action and context.get("risk_level") == risk_level:
                count += 1
        return count

    # --------------------------------------------------------------- intents

    def pending_intent_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM intents WHERE status='pending'"
        ).fetchone()
        return int(row["n"])

    def insert_intent(self, id: str, kind: str, payload: dict, created_ts: float) -> bool:
        """Idempotent insert: duplicate ids return False, never raise."""
        try:
            self._conn.execute(
                "INSERT INTO intents"
                " (id, created_ts, updated_ts, kind, payload, status)"
                " VALUES (?, ?, ?, ?, ?, 'pending')",
                (id, created_ts, created_ts, kind, json.dumps(payload, sort_keys=True)),
            )
            return True
        except sqlite3.IntegrityError:
            return False

    def claim_next_pending(
        self, kind: str | None, worker: str, now: float
    ) -> sqlite3.Row | None:
        """Atomically claim the oldest pending intent of a kind.

        BEGIN IMMEDIATE keeps the select+cas-update atomic; a competing claim
        returns None (never two owners), safe under retries and restarts.
        """
        with self._txn() as conn:
            row = conn.execute(
                "SELECT id FROM intents WHERE status='pending'"
                + (" AND kind=?" if kind else "")
                + " ORDER BY created_ts, id LIMIT 1",
                (kind,) if kind else (),
            ).fetchone()
            if row is None:
                return None
            cur = conn.execute(
                "UPDATE intents SET status='claimed', claimed_by=?, claimed_ts=?,"
                " updated_ts=?, tries=tries+1 WHERE id=? AND status='pending'",
                (worker, now, now, row["id"]),
            )
            if cur.rowcount != 1:
                return None
            updated = conn.execute(
                "SELECT * FROM intents WHERE id=?", (row["id"],)
            ).fetchone()
        return updated

    def finish_intent(
        self, id: str, status: str, result: dict | str | None, now: float
    ) -> bool:
        """Move a CLAIMED intent to a final status (compare-and-set).

        Only claimed intents can finalize: an unclaimed intent must not be
        consumable by a worker that never owned it, and terminal states
        cannot be re-entered (double-apply protection). Use
        reconcile_intents to recover abandoned claims instead.
        """
        if status not in _INTENT_FINAL_STATES:
            raise ValueError(f"invalid final intent status: {status!r}")
        payload = (
            json.dumps(result, sort_keys=True)
            if isinstance(result, dict)
            else (str(result) if result is not None else None)
        )
        with self._txn() as conn:
            cur = conn.execute(
                "UPDATE intents SET status=?, result=?, updated_ts=?"
                " WHERE id=? AND status='claimed'",
                (status, payload, now, id),
            )
            return cur.rowcount == 1

    def release_claim(self, intent_id: str, now: float, reason: str) -> bool:
        """Return a CLAIMED intent to pending without consuming it.

        Used when a worker aborts before acting (e.g. invocation limit):
        the intent must stay actionable, its tries counter already reflects
        the claim (incremented atomically by claim_next_pending).
        """
        with self._txn() as conn:
            cur = conn.execute(
                "UPDATE intents SET status='pending', claimed_by=NULL,"
                " claimed_ts=NULL, updated_ts=? WHERE id=? AND status='claimed'",
                (now, intent_id),
            )
            if cur.rowcount == 1:
                conn.execute(
                    "UPDATE intents SET result=? WHERE id=?",
                    (f"released: {reason}", intent_id),
                )
            return cur.rowcount == 1

    def reconcile_intents(
        self, now: float, lease_s: float = 120.0, max_tries: int = 3
    ) -> list[tuple[str, str]]:
        """Recover intents whose claimant died within the lease window.

        Stale claims go back to 'pending' (tries already incremented by the
        claim); past max_tries they expire so a poison message cannot loop
        forever. Run this periodically from the daemon.
        """
        actions: list[tuple[str, str]] = []
        with self._txn() as conn:
            rows = conn.execute(
                "SELECT id, tries, claimed_ts FROM intents"
                " WHERE status='claimed' AND claimed_ts <= ?",
                (now - lease_s,),
            ).fetchall()
            for row in rows:
                if row["tries"] >= max_tries:
                    conn.execute(
                        "UPDATE intents SET status='expired', updated_ts=? WHERE id=?",
                        (now, row["id"]),
                    )
                    actions.append((row["id"], "expired"))
                else:
                    conn.execute(
                        "UPDATE intents SET status='pending', claimed_by=NULL,"
                        " claimed_ts=NULL, updated_ts=? WHERE id=?",
                        (now, row["id"]),
                    )
                    actions.append((row["id"], "back_to_pending"))
        return actions


    # ---------------------------------------------------------------- RL helpers

    def get_agent_mode(self) -> str:
        """Read current agent mode from meta ('observe' | 'advise' | 'autopilot')."""
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='agent_mode'"
        ).fetchone()
        return str(row[0]) if row else "observe"

    def set_agent_mode(self, mode: str) -> None:
        """Set agent mode (written by Streamlit dashboard)."""
        if mode not in ("observe", "advise", "autopilot"):
            raise ValueError(f"invalid agent_mode: {mode!r}")
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES ('agent_mode', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (mode,),
        )

    def insert_rl_metric(self, kind: str, payload: dict, checkpoint_id: str | None = None) -> None:
        """Record a training metric (train step, eval result, mode change)."""
        try:
            self._conn.execute(
                "INSERT INTO rl_metrics (ts, checkpoint_id, kind, payload)"
                " VALUES (?, ?, ?, ?)",
                (time.time(), checkpoint_id, kind,
                 json.dumps(payload, sort_keys=True)),
            )
        except Exception:  # noqa: BLE001 - metrics must never kill sampling
            pass

    def recent_rl_metrics(self, limit: int = 100) -> list[sqlite3.Row]:
        """Most recent RL training metrics for the dashboard."""
        return self._conn.execute(
            "SELECT ts, checkpoint_id, kind, payload FROM rl_metrics"
            " ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()


def _sample_to_row(s: Sample) -> dict:
    """Sample -> insertable row dict. Column names mirror _SCHEMA; every
    'unknown' stays None rather than becoming a zero."""
    return {
        "ts": s.ts,
        "interval_s": s.interval_s,
        "ncpu": s.ncpu,
        "cpu_total_pct": s.cpu_total_pct,
        "cpu_percpu_json": json.dumps(s.cpu_percpu_pct),
        "load1": s.load1,
        "load5": s.load5,
        "load15": s.load15,
        "mem_total_bytes": s.mem_total_bytes,
        "mem_available_bytes": s.mem_available_bytes,
        "mem_used_pct": s.mem_used_pct,
        "swap_total_bytes": s.swap_total_bytes,
        "swap_used_bytes": s.swap_used_bytes,
        "swap_used_pct": s.swap_used_pct,
        "swp_in_ps": s.swp_in_ps,
        "swp_out_ps": s.swp_out_ps,
        "majflt_ps": s.majflt_ps,
        "reclaim_scan_ps": s.reclaim_scan_ps,
        "reclaim_steal_ps": s.reclaim_steal_ps,
        "allocstall_ps": s.allocstall_ps,
        "psi_cpu_some_total_us": s.psi_cpu.some_total_us,
        "psi_cpu_some_stall_us": s.psi_cpu.some_stall_us,
        "psi_cpu_full_total_us": s.psi_cpu.full_total_us,
        "psi_cpu_full_stall_us": s.psi_cpu.full_stall_us,
        "psi_mem_some_total_us": s.psi_mem.some_total_us,
        "psi_mem_some_stall_us": s.psi_mem.some_stall_us,
        "psi_mem_full_total_us": s.psi_mem.full_total_us,
        "psi_mem_full_stall_us": s.psi_mem.full_stall_us,
        "psi_io_some_total_us": s.psi_io.some_total_us,
        "psi_io_some_stall_us": s.psi_io.some_stall_us,
        "psi_io_full_total_us": s.psi_io.full_total_us,
        "psi_io_full_stall_us": s.psi_io.full_stall_us,
        "gpu_util_pct": None if s.gpu is None else s.gpu.util_pct,
        "gpu_mem_used_mib": None if s.gpu is None else s.gpu.mem_used_mib,
        "gpu_mem_total_mib": None if s.gpu is None else s.gpu.mem_total_mib,
        "gpu_temp_c": None if s.gpu is None else s.gpu.temp_c,
        "gpu_power_w": None if s.gpu is None else s.gpu.power_w,
        "gpu_top_procs_json": json.dumps(s.gpu_top_procs, sort_keys=True),
        "top_procs_json": json.dumps(s.top_procs, sort_keys=True),
        "collect_ms": s.collect_ms,
    }
