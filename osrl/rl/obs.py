"""Observation encoder: sliding window of Sample objects → flat float32 vector.

Design rules (matching the project-wide abstention discipline):
- None fields → 0.0 in the obs vector; a companion binary "valid" mask
  channel lets the network distinguish "zero" from "unknown".
- The vector size is FIXED (OBS_DIM) regardless of missing inputs.
- Per-core CPU list is padded / truncated to NCPU_MAX so shape is stable
  even if psutil reports a different count across restarts.
- All values are pre-clipped to physically-reasonable ranges before
  normalisation so outliers cannot saturate the network.
- Running normalisation (VecNormalize) is layered on top at training time;
  this module only does the structural encoding.

Observation layout (OBS_DIM = 114 = 57 features + 57 validity mask bits):
  [0:5]    system scalars  (cpu_total, load1, load5, iowait_pct, ncpu_norm)
  [5:13]   memory          (avail_mib, swap_used_pct, swp_in, swp_out,
                            majflt, allocstall, reclaim_scan, reclaim_steal)
  [13:21]  PSI             (3 some-stall-ratio + 2 full-stall-ratio + 3 total)
  [21:26]  GPU             (util_pct, mem_used_mib, mem_total_mib, temp_c, power_w)
  [26:39]  per-core CPU    (12 cores padded to NCPU_MAX=13)
  [39:45]  risk encoding   (level 0..3, bad_streak, good_streak,
                            psi_recent, trigger_recent, pad)
  [45:51]  deltas          (cpu_delta, swap_delta, psi_mem_delta,
                            swp_out_delta, mem_avail_delta, allocstall_delta)
  [51:57]  agent history   (last_action one-hot 5, steps_since_action_norm)
  [57:114] validity mask   (1.0 if original value was not-None, 0.0 if missing)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from osrl.collector import Sample

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NCPU_MAX = 13           # i5-12450HX has 12; +1 padding slot
N_OBS_STEPS = 10        # samples per observation window (20 s at 2 s cadence)
N_ACTIONS = 5           # NOOP/ALERT/RENICE_BACK/REQUEST_APPROVAL/TUNE_THRESHOLDS

# Layout sizes (must sum to OBS_DIM)
_SZ_SYSTEM   = 5
_SZ_MEMORY   = 8
_SZ_PSI      = 8   # 3 some-ratio + 2 full-ratio (mem/io) + 3 total (cpu/mem/io)
_SZ_GPU      = 5
_SZ_PERCPU   = NCPU_MAX
_SZ_RISK     = 6
_SZ_DELTAS   = 6
_SZ_HISTORY  = 6        # 5 action one-hot + steps_since norm
_SZ_FEATURES = (_SZ_SYSTEM + _SZ_MEMORY + _SZ_PSI + _SZ_GPU
                + _SZ_PERCPU + _SZ_RISK + _SZ_DELTAS + _SZ_HISTORY)
_SZ_MASK     = _SZ_FEATURES   # one validity bit per feature

OBS_DIM = _SZ_FEATURES + _SZ_MASK   # 57 + 57 = 114

# Normalisation ranges — clip then scale to [0, 1]
_MEM_TOTAL_MIB   = 15 * 1024.0       # 15 GiB
_SWAP_TOTAL_MIB  = 4 * 1024.0        # 4 GiB
_GPU_MEM_MIB     = 6 * 1024.0        # 6 GiB RTX 3050
_MAX_LOAD        = 24.0               # 2× thread count
_MAX_SWP_PS      = 50_000.0           # pages/s
_MAX_STALL_RATIO = 1.0                # fraction 0..1
_MAX_ALLOC_PS    = 200.0
_MAX_MAJFLT_PS   = 5_000.0
_MAX_STREAK      = 50.0               # bad/good streak cap
_MAX_GPU_TEMP    = 100.0
_MAX_GPU_POWER   = 200.0

RISK_LEVEL_MAP = {"none": 0.0, "elevated": 0.5, "critical": 1.0, "unknown": -1.0}


def _clip01(v: float | None, lo: float, hi: float) -> tuple[float, float]:
    """Returns (normalised_value, valid_flag)."""
    if v is None or not math.isfinite(v):
        return 0.0, 0.0
    return float(np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)), 1.0


def _psi_ratio(sample: Sample, resource: str) -> tuple[float, float]:
    """PSI some-stall ratio for resource 'cpu'|'mem'|'io'."""
    psi_attr = {"cpu": "psi_cpu", "mem": "psi_mem", "io": "psi_io"}[resource]
    psi = getattr(sample, psi_attr, None)
    if psi is None:
        return 0.0, 0.0
    stall = getattr(psi, "some_stall_us", None)
    interval = sample.interval_s
    if (stall is None or interval is None or not math.isfinite(stall)
            or not math.isfinite(interval) or interval <= 0 or stall < 0):
        return 0.0, 0.0
    ratio = stall / (interval * 1e6)
    return float(np.clip(ratio, 0.0, 1.0)), 1.0


@dataclass
class AgentState:
    """Mutable agent-side state fed into observation encoding."""
    last_action: int = 0                     # action index 0..N_ACTIONS-1
    steps_since_last_action: int = 999       # large = long time ago
    risk_level: str = "none"
    bad_streak: int = 0
    good_streak: int = 0
    psi_recent: bool = False
    trigger_recent: bool = False
    # rolling delta tracking
    _prev_features: dict = field(default_factory=dict)


def encode_obs(window: list[Sample], agent: AgentState) -> np.ndarray:
    """Encode a sliding window (up to N_OBS_STEPS) into the obs vector.

    Uses the MOST RECENT sample for instantaneous values; earlier samples
    inform the delta block. If the window is empty, returns an all-zeros
    vector (unknown state).
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    valid = np.zeros(_SZ_FEATURES, dtype=np.float32)

    if not window:
        obs[_SZ_FEATURES:] = valid
        return obs

    s = window[-1]   # most recent sample

    idx = 0  # write cursor into obs[:_SZ_FEATURES]
    vidx = 0  # write cursor into valid[]

    # ------------------------------------------------------------------
    # Block 0: System scalars [0:5]
    # ------------------------------------------------------------------
    v, f = _clip01(s.cpu_total_pct, 0.0, 100.0)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(s.load1, 0.0, _MAX_LOAD)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(s.load5, 0.0, _MAX_LOAD)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    # iowait: derive from per-cpu if available
    iowait = None
    if s.cpu_percpu_pct:
        pass  # iowait not tracked per-core; use total as proxy
    v, f = _clip01(iowait, 0.0, 100.0)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    # ncpu normalised (0..1 over [1, NCPU_MAX])
    v, f = _clip01(float(s.ncpu), 1.0, float(NCPU_MAX))
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    # ------------------------------------------------------------------
    # Block 1: Memory [5:13]
    # ------------------------------------------------------------------
    mem_avail_mib = (
        s.mem_available_bytes / (1024.0 * 1024.0)
        if s.mem_available_bytes is not None else None
    )
    v, f = _clip01(mem_avail_mib, 0.0, _MEM_TOTAL_MIB)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(s.swap_used_pct, 0.0, 100.0)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(s.swp_in_ps, 0.0, _MAX_SWP_PS)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(s.swp_out_ps, 0.0, _MAX_SWP_PS)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(s.majflt_ps, 0.0, _MAX_MAJFLT_PS)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(s.allocstall_ps, 0.0, _MAX_ALLOC_PS)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(s.reclaim_scan_ps, 0.0, _MAX_SWP_PS)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(s.reclaim_steal_ps, 0.0, _MAX_SWP_PS)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    # ------------------------------------------------------------------
    # Block 2: PSI [13:22]
    # ------------------------------------------------------------------
    for resource in ("cpu", "mem", "io"):
        v, f = _psi_ratio(s, resource)
        obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    # PSI avg10 (from PsiInterval — we proxy via the stall ratio already;
    # use full-stall ratio as second PSI channel for mem and io)
    for resource in ("mem", "io"):
        psi_attr = {"mem": "psi_mem", "io": "psi_io"}[resource]
        psi = getattr(s, psi_attr, None)
        full_stall = getattr(psi, "full_stall_us", None) if psi else None
        interval = s.interval_s
        if (full_stall is not None and interval and interval > 0
                and math.isfinite(full_stall) and math.isfinite(interval)):
            ratio = full_stall / (interval * 1e6)
            obs[idx] = float(np.clip(ratio, 0.0, 1.0))
            valid[vidx] = 1.0
        idx += 1; vidx += 1

    # PSI total stall delta (normalised over the interval, 3rd PSI channel)
    for resource in ("cpu", "mem", "io"):
        psi_attr = {"cpu": "psi_cpu", "mem": "psi_mem", "io": "psi_io"}[resource]
        psi = getattr(s, psi_attr, None)
        total = getattr(psi, "some_total_us", None) if psi else None
        v, f = _clip01(total, 0.0, 1e9) if total is not None else (0.0, 0.0)
        obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    # ------------------------------------------------------------------
    # Block 3: GPU [22:27]
    # ------------------------------------------------------------------
    gpu = s.gpu
    v, f = _clip01(gpu.util_pct if gpu else None, 0.0, 100.0)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(gpu.mem_used_mib if gpu else None, 0.0, _GPU_MEM_MIB)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(gpu.mem_total_mib if gpu else None, 0.0, _GPU_MEM_MIB)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(gpu.temp_c if gpu else None, 0.0, _MAX_GPU_TEMP)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(gpu.power_w if gpu else None, 0.0, _MAX_GPU_POWER)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    # ------------------------------------------------------------------
    # Block 4: Per-core CPU [27:40]
    # ------------------------------------------------------------------
    percpu = s.cpu_percpu_pct or []
    for i in range(NCPU_MAX):
        if i < len(percpu) and percpu[i] is not None and math.isfinite(percpu[i]):
            obs[idx] = float(np.clip(percpu[i] / 100.0, 0.0, 1.0))
            valid[vidx] = 1.0
        idx += 1; vidx += 1

    # ------------------------------------------------------------------
    # Block 5: Risk encoding [40:46]
    # ------------------------------------------------------------------
    obs[idx] = (RISK_LEVEL_MAP.get(agent.risk_level, -1.0) + 1.0) / 2.0
    valid[vidx] = 1.0; idx += 1; vidx += 1

    v, f = _clip01(float(agent.bad_streak), 0.0, _MAX_STREAK)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    v, f = _clip01(float(agent.good_streak), 0.0, _MAX_STREAK)
    obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1

    obs[idx] = 1.0 if agent.psi_recent else 0.0
    valid[vidx] = 1.0; idx += 1; vidx += 1

    obs[idx] = 1.0 if agent.trigger_recent else 0.0
    valid[vidx] = 1.0; idx += 1; vidx += 1

    # padding slot
    idx += 1; vidx += 1

    # ------------------------------------------------------------------
    # Block 6: Deltas [46:52]
    # ------------------------------------------------------------------
    first = window[0] if len(window) > 1 else None
    previous = {} if first is None else {
        "cpu": first.cpu_total_pct, "swap_pct": first.swap_used_pct,
        "psi_mem": _psi_ratio(first, "mem")[0] if _psi_ratio(first, "mem")[1] else None,
        "swp_out": first.swp_out_ps,
        "mem_avail": first.mem_available_bytes / 1024**2
                     if first.mem_available_bytes is not None else None,
        "alloc": first.allocstall_ps,
    }

    def _delta_norm(key: str, cur: float | None, lo: float, hi: float
                    ) -> tuple[float, float]:
        prev_val = previous.get(key)
        if cur is None or prev_val is None or not math.isfinite(cur) or not math.isfinite(prev_val):
            return 0.0, 0.0
        delta = cur - prev_val
        rng = max(hi - lo, 1e-9)
        return float(np.clip((delta + rng) / (2.0 * rng), 0.0, 1.0)), 1.0

    deltas = [
        ("cpu",       s.cpu_total_pct,            0.0,          100.0),
        ("swap_pct",  s.swap_used_pct,             0.0,          100.0),
        ("psi_mem",   _psi_ratio(s, "mem")[0] if _psi_ratio(s, "mem")[1] else None,     0.0,          1.0),
        ("swp_out",   s.swp_out_ps,                0.0,          _MAX_SWP_PS),
        ("mem_avail", mem_avail_mib,               0.0,          _MEM_TOTAL_MIB),
        ("alloc",     s.allocstall_ps,             0.0,          _MAX_ALLOC_PS),
    ]
    for key, cur_val, lo, hi in deltas:
        v, f = _delta_norm(key, cur_val, lo, hi)
        obs[idx] = v; valid[vidx] = f; idx += 1; vidx += 1
    # update rolling prev for next call
    # Encoding is pure: repeated rendering/inference cannot change features.

    # ------------------------------------------------------------------
    # Block 7: Agent history [52:58]
    # ------------------------------------------------------------------
    # One-hot of last action (5 dims)
    for a in range(N_ACTIONS):
        obs[idx] = 1.0 if a == agent.last_action else 0.0
        valid[vidx] = 1.0; idx += 1; vidx += 1

    # steps since last action (normalised; cap at 100 steps)
    v, _ = _clip01(float(agent.steps_since_last_action), 0.0, 100.0)
    obs[idx] = v; valid[vidx] = 1.0; idx += 1; vidx += 1

    # ------------------------------------------------------------------
    # Validity mask block [_SZ_FEATURES : OBS_DIM]
    # ------------------------------------------------------------------
    obs[_SZ_FEATURES:] = valid

    assert idx == _SZ_FEATURES, f"obs encoder cursor mismatch: {idx}"
    return obs
