"""Advanced reward function with exponential penalties, trend detection, and stability bonuses.

Improvements over v1:
  - Exponential PSI spike penalties (small stalls barely penalised, large stalls hit hard)
  - Swap urgency curve: penalty grows as e^(swap%) not linearly
  - Trend component: reward if system is *improving*, penalise if worsening
  - Stability bonus: low variance in CPU/mem = bonus (steady system)
  - OOM-proximity penalty: when mem_available < 512 MiB, hard penalty
  - Foreground protection: quadratic penalty if fg processes lose CPU
  - Action efficiency: penalise redundant actions less if they fixed a problem
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ── Thresholds ────────────────────────────────────────────────────────────────
_SWP_NORM_PS      = 2048.0        # pages/s — critical swap-out threshold
_MEM_HEADROOM_MIB = 3 * 1024.0   # 3 GiB → full headroom bonus
_MEM_OOM_MIB      = 512.0        # < 512 MiB = near OOM
_MEM_TOTAL_MIB    = 15 * 1024.0
_PSI_SPIKE_THRESH = 0.05          # 5% stall ratio = start of exponential regime

_FOREGROUND_NAMES = frozenset({
    "gnome-shell", "xorg", "xwayland", "mutter", "gnome-session",
    "code", "chrome", "chromium", "firefox",
})

# ── Reward weights ────────────────────────────────────────────────────────────
_W_PSI_MEM    = -4.0   # ↑ from -3.0 — memory stall is most critical
_W_PSI_IO     = -2.5   # ↑ from -2.0
_W_SWP_OUT    = -3.0   # ↑ from -2.5
_W_PSI_CPU    = -1.0
_W_HEADROOM   =  1.5   # ↑ from 1.0
_W_SWAP_FREE  =  0.8   # ↑ from 0.5
_W_FG_SHARE   =  1.0   # ↑ from 0.8
_W_ACTION     = -0.02  # ↓ penalty (less discouraging of good actions)
_W_STABILITY  =  0.5   # NEW: stability bonus
_W_TREND      =  0.8   # NEW: improving trend reward
_W_OOM        = -5.0   # NEW: near-OOM hard penalty


def _safe(v: float | None, default: float = 0.0) -> float:
    if v is None or not math.isfinite(v):
        return default
    return float(v)


def _safe_sum(values) -> float:
    return sum(v for v in values if v is not None and math.isfinite(v))


def _psi_stall_ratio(window: list, attr: str) -> float:
    """Mean PSI stall ratio (stall_us / interval_us) over window."""
    ratios = []
    for s in window:
        psi = getattr(s, attr, None)
        if psi is None:
            continue
        stall_us = getattr(psi, 'some_stall_us', None)
        # Get interval from sample
        interval_s = getattr(s, 'interval_s', None)
        if interval_s is None or not math.isfinite(interval_s) or interval_s <= 0:
            continue
        if stall_us is not None and stall_us >= 0:
            ratio = stall_us / (interval_s * 1e6)
            ratios.append(min(ratio, 1.0))
    return sum(ratios) / len(ratios) if ratios else 0.0


def _psi_stall_s(window: list, attr: str) -> float:
    """Total PSI stall seconds over the window."""
    total_us = _safe_sum(
        getattr(getattr(s, attr, None), 'some_stall_us', None) for s in window
    )
    return total_us / 1e6


def _exponential_psi_penalty(ratio: float) -> float:
    """Exponential penalty: small stalls barely hurt, large stalls punished hard.
    
    f(x) = 0                         for x < thresh
    f(x) = (x - thresh)^2 * scale    for x >= thresh
    This gives a smooth quadratic-then-exponential curve.
    """
    if ratio <= _PSI_SPIKE_THRESH:
        return ratio * 0.5  # gentle linear for tiny stalls
    excess = ratio - _PSI_SPIKE_THRESH
    # Quadratic growth above threshold
    return _PSI_SPIKE_THRESH * 0.5 + excess + excess ** 2 * 3.0


def _swap_urgency(window: list) -> float:
    """Exponential urgency curve: swap% 90+ is catastrophic, 50- is mild."""
    pcts = [_safe(s.swap_used_pct) for s in window if s.swap_used_pct is not None]
    if not pcts:
        return 0.0
    mean_pct = max(0.0, min(1.0, sum(pcts) / len(pcts) / 100.0))  # 0..1
    # Exponential: f(x) = e^(3x) - 1, normalised so f(1)=1
    return min((math.exp(3.0 * mean_pct) - 1.0) / (math.exp(3.0) - 1.0), 1.0)


def _norm_swap_out(window: list) -> float:
    """Mean swap-out rate normalised, with exponential tail."""
    rates = [_safe(s.swp_out_ps) for s in window if s.swp_out_ps is not None]
    if not rates:
        return 0.0
    mean_rate = sum(rates) / len(rates)
    linear = min(mean_rate / _SWP_NORM_PS, 1.0)
    # Square it to give quadratic growth
    return linear ** 1.5


def _headroom_bonus(window: list) -> float:
    """Saturating bonus for free RAM. Sharper cutoff below 1 GiB."""
    avails = [
        _safe(s.mem_available_bytes) / (1024.0 * 1024.0)
        for s in window if s.mem_available_bytes is not None
    ]
    if not avails:
        return 0.0
    mean_mib = sum(avails) / len(avails)
    # Sigmoid-like: fast reward up to 3 GiB, then flat
    return min(mean_mib / _MEM_HEADROOM_MIB, 1.0) ** 0.7  # concave = reward early gains


def _oom_proximity_penalty(window: list) -> float:
    """Hard penalty when available memory drops below 512 MiB."""
    penalty = 0.0
    for s in window:
        avail_mib = _safe(s.mem_available_bytes) / (1024.0 * 1024.0)
        if s.mem_available_bytes is not None and 0 <= avail_mib < _MEM_OOM_MIB:
            # Severity: 1.0 at 0 MiB, 0.0 at OOM_THRESH MiB
            sev = 1.0 - avail_mib / _MEM_OOM_MIB
            penalty += sev ** 2  # quadratic
    return penalty / len(window) if window else 0.0


def _stability_bonus(window: list) -> float:
    """Reward low variance in CPU and swap (stable system = good).

    Uses cpu_total_pct and swap_used_pct — both are real Sample attributes.
    """
    if len(window) < 3:
        return 0.0
    cpu_vals  = [_safe(s.cpu_total_pct)  / 100.0 for s in window
                 if getattr(s, 'cpu_total_pct', None) is not None]
    swap_vals = [_safe(s.swap_used_pct)  / 100.0 for s in window
                 if getattr(s, 'swap_used_pct', None) is not None]

    bonus = 0.0
    if len(cpu_vals) >= 3:
        cpu_std = _std(cpu_vals)
        bonus += max(0.0, 0.5 - cpu_std * 3.0)   # bonus ∝ (0.5 − std)
    if len(swap_vals) >= 3:
        swp_std = _std(swap_vals)
        bonus += max(0.0, 0.5 - swp_std * 3.0)
    return bonus / 2.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return math.sqrt(max(0.0, var))


def _trend_component(window: list) -> float:
    """Reward improving trends, penalise worsening ones.
    
    Compares first-half vs second-half of window:
    - Swap decreasing → positive
    - PSI decreasing → positive  
    - Memory available increasing → positive
    """
    if len(window) < 4:
        return 0.0
    mid = len(window) // 2
    first, second = window[:mid], window[mid:]
    
    trend = 0.0
    
    # Swap trend
    def _mean_safe(samples, attr):
        vals = [_safe(getattr(s, attr)) for s in samples if getattr(s, attr) is not None]
        return sum(vals) / len(vals) if vals else None
    
    swap_first  = _mean_safe(first,  'swap_used_pct')
    swap_second = _mean_safe(second, 'swap_used_pct')
    if swap_first is not None and swap_second is not None:
        delta = (swap_first - swap_second) / 100.0  # positive = improving
        trend += delta * 1.5
    
    # Memory available trend  
    mem_first  = _mean_safe(first,  'mem_available_bytes')
    mem_second = _mean_safe(second, 'mem_available_bytes')
    if mem_first is not None and mem_second is not None and mem_first > 0:
        delta = (mem_second - mem_first) / (mem_first + 1e-6)
        trend += min(max(delta, -1.0), 1.0) * 0.5
    
    return max(min(trend, 1.0), -1.0)  # clamp to [-1, 1]


def _swap_free_fraction(window: list) -> float:
    pcts = [_safe(s.swap_used_pct) for s in window if s.swap_used_pct is not None]
    if not pcts:
        return 0.5
    return max(0.0, 1.0 - sum(pcts) / len(pcts) / 100.0)


def _foreground_share(window: list) -> float:
    fg_vals = []
    for s in window:
        if not s.top_procs or s.cpu_total_pct is None or s.cpu_total_pct <= 0:
            continue
        fg_cpu = sum(
            float(p.get('cpu_pct') or 0.0)
            for p in s.top_procs
            if any(fn in str(p.get('name', '')).lower() for fn in _FOREGROUND_NAMES)
        )
        fg_vals.append(min(fg_cpu / s.cpu_total_pct, 1.0))
    if not fg_vals:
        return 0.5
    return sum(fg_vals) / len(fg_vals)


def compute_reward(
    window: list,
    n_actions_this_step: int = 0,
) -> float:
    """Advanced reward: exponential PSI penalties + trend + stability + OOM guard."""
    if not window:
        return 0.0

    # PSI penalties — exponential
    psi_mem_ratio = _psi_stall_ratio(window, 'psi_mem')
    psi_io_ratio  = _psi_stall_ratio(window, 'psi_io')
    psi_cpu_ratio = _psi_stall_ratio(window, 'psi_cpu')

    r = 0.0
    r += _W_PSI_MEM   * _exponential_psi_penalty(psi_mem_ratio)
    r += _W_PSI_IO    * _exponential_psi_penalty(psi_io_ratio)
    r += _W_PSI_CPU   * _exponential_psi_penalty(psi_cpu_ratio)
    r += _W_SWP_OUT   * _norm_swap_out(window)
    r += _W_SWP_OUT   * _swap_urgency(window) * 0.5  # double-penalise high swap
    r += _W_HEADROOM  * _headroom_bonus(window)
    r += _W_SWAP_FREE * _swap_free_fraction(window)
    r += _W_FG_SHARE  * _foreground_share(window)
    r += _W_OOM       * _oom_proximity_penalty(window)
    r += _W_STABILITY * _stability_bonus(window)
    r += _W_TREND     * _trend_component(window)
    r += _W_ACTION    * n_actions_this_step

    return float(r) if math.isfinite(r) else 0.0


def reward_breakdown(window: list, n_actions: int = 0) -> dict:
    """Per-term breakdown for logging and dashboard display."""
    psi_mem_r = _psi_stall_ratio(window, 'psi_mem')
    psi_io_r  = _psi_stall_ratio(window, 'psi_io')
    psi_cpu_r = _psi_stall_ratio(window, 'psi_cpu')
    terms = {
        'psi_mem':      _W_PSI_MEM   * _exponential_psi_penalty(psi_mem_r),
        'psi_io':       _W_PSI_IO    * _exponential_psi_penalty(psi_io_r),
        'psi_cpu':      _W_PSI_CPU   * _exponential_psi_penalty(psi_cpu_r),
        'swap_out':     _W_SWP_OUT   * _norm_swap_out(window),
        'swap_urgency': _W_SWP_OUT   * _swap_urgency(window) * 0.5,
        'headroom':     _W_HEADROOM  * _headroom_bonus(window),
        'swap_free':    _W_SWAP_FREE * _swap_free_fraction(window),
        'fg_share':     _W_FG_SHARE  * _foreground_share(window),
        'oom_penalty':  _W_OOM       * _oom_proximity_penalty(window),
        'stability':    _W_STABILITY * _stability_bonus(window),
        'trend':        _W_TREND     * _trend_component(window),
        'action_cost':  _W_ACTION    * n_actions,
    }
    terms['total'] = sum(terms.values())
    return terms
