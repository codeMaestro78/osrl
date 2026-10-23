"""Configuration with environment-variable overrides.

Every knob Phase 0 needs lives here so it is inspectable and testable.
Environment variables read by :func:`load_config` (unset -> default):

    OSRL_DB                  path to the SQLite database
    OSRL_INTERVAL            collector sample period, seconds
    OSRL_RAW_DAYS            retention of raw 2 s samples (days)
    OSRL_EVENTS_DAYS         retention of process/anomaly events (days)
    OSRL_ROLLUP_SECONDS      rollup window size (default 900 = 15 min)
    OSRL_HEARTBEAT_S         daemon heartbeat period, seconds
    OSRL_BATCH_SIZE          flush after N samples
    OSRL_BATCH_INTERVAL_S    flush after T seconds regardless of batch size
    OSRL_WAL_CHECKPOINT_S    WAL checkpoint period, seconds

Phase 1/2 additions (defaults documented in the dataclasses):

    OSRL_HEAVY_CPU_PCT / OSRL_HEAVY_RSS_MIB   heavyweight birth thresholds
    OSRL_RISK_*                               risk estimator thresholds
    OSRL_GUARD_*                              execution guard thresholds
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path(name: str, default: str) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default).expanduser()


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


@dataclass(frozen=True)
class CollectorConfig:
    interval_s: float = 2.0
    # Heavyweight birth/death thresholds: evidence-tuned 2026-10-20 from 88
    # recorded births / 0.7 h (hold-test session): at (25%, 200 MiB) the CPU
    # side never fired (newborn CPU% is ~0) and the RSS floor caught pure
    # Chrome-utility spawn noise (median 273 MiB). (512 MiB, 50%) keeps 13%
    # of that noise while retaining the meaningful names (chrome renderers,
    # code, next-server). Overridable via OSRL_HEAVY_* for further tuning.
    heavy_cpu_pct: float = 50.0
    heavy_rss_mib: float = 512.0
    top_k_procs: int = 15


@dataclass(frozen=True)
class RetentionConfig:
    """DESIGN.md section 7: 2 s raw data 7 days, 15 min rollups forever."""

    raw_days: int = 7
    events_days: int = 30
    rollup_seconds: int = 900
    heartbeat_s: float = 30.0
    batch_size: int = 10
    batch_interval_s: float = 10.0
    wal_checkpoint_s: float = 3600.0


@dataclass(frozen=True)
class RiskConfig:
    """Deterministic thrash-risk thresholds (Phase 1). Initial values are
    hypotheses to be retuned from recorded data, per DESIGN §11.7."""

    staleness_s: float = 6.0  # >3x default interval -> 'unknown'
    enter_n: int = 5          # consecutive bad samples to raise a level (10 s)
    exit_n: int = 15          # consecutive good samples to demote (30 s)
    # memory PSI: fraction of the interval during which tasks were stalled
    psi_mem_stall_warn: float = 0.20
    psi_mem_stall_crit: float = 0.50
    # /proc/vmstat delta rates, pages/s
    swp_out_warn_ps: float = 512.0
    swp_out_crit_ps: float = 2048.0
    allocstall_warn_ps: float = 5.0
    allocstall_crit_ps: float = 25.0
    # MemAvailable low-water marks, MiB
    memavail_warn_mib: float = 768.0
    memavail_crit_mib: float = 256.0
    # trigger fire decay: the kernel rate-limits notifications to one per
    # tracking window, so absence of a second notification is NOT proof of
    # recovery. Two windows of grace is a hypothesis to be tuned on data.
    trigger_freshness_s: float = 4.0


@dataclass(frozen=True)
class PsiTriggerConfig:
    """Event-driven PSI monitors (Phase 1). Windows are 2 s = the smallest
    unprivileged-legal multiple (kernel: EINVAL otherwise); thresholds
    chosen conservatively -- they generate events, never actions."""

    enabled: bool = True
    memory_some_us: int = 150_000          # 150 ms partial memory stall
    memory_window_us: int = 2_000_000      # 2 s, multiple of 2 s
    io_full_us: int = 50_000               # 50 ms full IO stall
    io_window_us: int = 2_000_000
    probe_interval_s: float = 0.5


@dataclass(frozen=True)
class GuardConfig:
    """Execution guard (Phase 2). No RL anywhere: the guard is a total
    function of machine state (see osrl/guard.py). `restore_capable` is
    FALSE on this deployment BY EVIDENCE, not convention: unprivileged
    nice-lowering returns EPERM both cross-process and self (probed
    2026-10-20, RLIMIT_NICE=(0,0)); execution must also be explicitly
    enabled per worker invocation (--execute): the daemon never runs it."""

    staleness_max_s: float = 12.0   # abstain when telemetry older than this
    renice_step: int = 10           # nice units added per renice_back
    renice_max_nice: int = 19       # hard ceiling; never exceed
    min_spacing_s: float = 300.0    # per (pid, kind) action spacing
    # empirical EPERM; flip only with a PROVEN elevated-helper deployment
    restore_capable: bool = False


@dataclass(frozen=True)
class RLConfig:
    """RL policy configuration (Phase 4/5). All values env-tunable."""

    n_obs_steps: int = 10           # samples per observation window
    n_train_steps: int = 100_000    # default PPO timesteps per run
    checkpoint_dir: Path = Path("~/os-rl-agent/checkpoints")
    eval_gate_mae_ratio: float = 0.05   # must beat NOOP by this fraction
    advise_cooldown_steps: int = 15     # min steps between RENICE_BACK suggestions
    # Synthetic stress injection (for pretraining before real events exist)
    synthetic_stress: bool = True
    stress_probability: float = 0.15
    stress_psi_ratio: float = 0.40
    stress_swp_out_ps: float = 1500.0


@dataclass(frozen=True)
class Config:
    db_path: Path
    collector: CollectorConfig = CollectorConfig()
    retention: RetentionConfig = RetentionConfig()
    risk: RiskConfig = RiskConfig()
    psi_trigger: PsiTriggerConfig = PsiTriggerConfig()
    guard: GuardConfig = GuardConfig()
    rl: RLConfig = RLConfig()


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def load_config() -> Config:
    return Config(
        db_path=_path("OSRL_DB", "~/os-rl-agent/data/telemetry.db"),
        collector=CollectorConfig(
            interval_s=_float("OSRL_INTERVAL", 2.0),
            heavy_cpu_pct=_float("OSRL_HEAVY_CPU_PCT", 50.0),
            heavy_rss_mib=_float("OSRL_HEAVY_RSS_MIB", 512.0),
            top_k_procs=_int("OSRL_TOP_K_PROCS", 15),
        ),
        retention=RetentionConfig(
            raw_days=_int("OSRL_RAW_DAYS", 7),
            events_days=_int("OSRL_EVENTS_DAYS", 30),
            rollup_seconds=_int("OSRL_ROLLUP_SECONDS", 900),
            heartbeat_s=_float("OSRL_HEARTBEAT_S", 30.0),
            batch_size=_int("OSRL_BATCH_SIZE", 10),
            batch_interval_s=_float("OSRL_BATCH_INTERVAL_S", 10.0),
            wal_checkpoint_s=_float("OSRL_WAL_CHECKPOINT_S", 3600.0),
        ),
        risk=RiskConfig(
            staleness_s=_float("OSRL_RISK_STALENESS_S", 6.0),
            enter_n=_int("OSRL_RISK_ENTER_N", 5),
            exit_n=_int("OSRL_RISK_EXIT_N", 15),
            psi_mem_stall_warn=_float("OSRL_RISK_PSI_WARN", 0.20),
            psi_mem_stall_crit=_float("OSRL_RISK_PSI_CRIT", 0.50),
            swp_out_warn_ps=_float("OSRL_RISK_SWAP_WARN", 512.0),
            swp_out_crit_ps=_float("OSRL_RISK_SWAP_CRIT", 2048.0),
            allocstall_warn_ps=_float("OSRL_RISK_ALLOCSTALL_WARN", 5.0),
            allocstall_crit_ps=_float("OSRL_RISK_ALLOCSTALL_CRIT", 25.0),
            memavail_warn_mib=_float("OSRL_RISK_MEMAVAIL_WARN", 768.0),
            memavail_crit_mib=_float("OSRL_RISK_MEMAVAIL_CRIT", 256.0),
            trigger_freshness_s=_float("OSRL_RISK_TRIGGER_FRESH_S", 4.0),
        ),
        psi_trigger=PsiTriggerConfig(
            enabled=_bool("OSRL_PSI_TRIGGER_ENABLED", True),
            memory_some_us=_int("OSRL_PSI_MEMORY_SOME_US", 150_000),
            memory_window_us=_int("OSRL_PSI_MEMORY_WINDOW_US", 2_000_000),
            io_full_us=_int("OSRL_PSI_IO_FULL_US", 50_000),
            io_window_us=_int("OSRL_PSI_IO_WINDOW_US", 2_000_000),
            probe_interval_s=_float("OSRL_PSI_PROBE_INTERVAL_S", 0.5),
        ),
        guard=GuardConfig(
            staleness_max_s=_float("OSRL_GUARD_STALENESS_MAX_S", 12.0),
            renice_step=_int("OSRL_GUARD_RENICE_STEP", 10),
            renice_max_nice=_int("OSRL_GUARD_RENICE_MAX_NICE", 19),
            min_spacing_s=_float("OSRL_GUARD_MIN_SPACING_S", 300.0),
            restore_capable=_bool("OSRL_GUARD_RESTORE_CAPABLE", False),
        ),
        rl=RLConfig(
            n_obs_steps=_int("OSRL_RL_OBS_STEPS", 10),
            n_train_steps=_int("OSRL_RL_TRAIN_STEPS", 100_000),
            checkpoint_dir=_path("OSRL_RL_CHECKPOINT_DIR",
                                  "~/os-rl-agent/checkpoints"),
            eval_gate_mae_ratio=_float("OSRL_RL_EVAL_GATE", 0.05),
            advise_cooldown_steps=_int("OSRL_RL_COOLDOWN_STEPS", 15),
            synthetic_stress=_bool("OSRL_RL_SYNTHETIC", True),
            stress_probability=_float("OSRL_RL_STRESS_PROB", 0.15),
            stress_psi_ratio=_float("OSRL_RL_STRESS_PSI", 0.40),
            stress_swp_out_ps=_float("OSRL_RL_STRESS_SWP", 1500.0),
        ),
    )
