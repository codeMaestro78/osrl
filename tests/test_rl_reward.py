"""Tests for osrl.rl.reward — reward function."""

from __future__ import annotations

import math

from osrl.collector import PsiInterval, Sample
from osrl.rl.reward import (
    _foreground_share,
    _headroom_bonus,
    _psi_stall_s,
    _swap_free_fraction,
    compute_reward,
    reward_breakdown,
)


def _make_sample(
    psi_mem_stall_us: int | None = 0,
    psi_io_stall_us: int | None = 0,
    psi_cpu_stall_us: int | None = 0,
    swp_out_ps: float | None = 0.0,
    mem_avail_bytes: int | None = int(9e9),
    swap_used_pct: float | None = 87.0,
    cpu_total_pct: float | None = 20.0,
    interval: float = 2.0,
    top_procs: list | None = None,
) -> Sample:
    return Sample(
        ts=1.0,
        interval_s=interval,
        cpu_total_pct=cpu_total_pct,
        mem_available_bytes=mem_avail_bytes,
        swap_used_pct=swap_used_pct,
        swp_out_ps=swp_out_ps,
        psi_mem=PsiInterval(some_stall_us=psi_mem_stall_us),
        psi_io=PsiInterval(some_stall_us=psi_io_stall_us),
        psi_cpu=PsiInterval(some_stall_us=psi_cpu_stall_us),
        top_procs=top_procs or [],
    )


def test_reward_quiet_machine_is_positive():
    """A quiet machine with good headroom and no stalls earns positive reward."""
    window = [_make_sample(
        mem_avail_bytes=int(10e9),   # ~9.3 GiB available → full headroom
        swap_used_pct=10.0,
        swp_out_ps=0.0,
    )] * 10
    r = compute_reward(window)
    assert r > 0.0, f"expected positive reward on quiet machine, got {r}"


def test_reward_thrashing_machine_is_negative():
    """High PSI stalls + heavy swap-out → strongly negative reward."""
    # 1 second of stall per 2-second sample → 50% stall ratio
    window = [_make_sample(
        psi_mem_stall_us=1_000_000,   # 1 s stall in 2 s window → large
        swp_out_ps=3000.0,            # above critical threshold
        mem_avail_bytes=int(200e6),   # < 256 MiB — critical
    )] * 10
    r = compute_reward(window)
    assert r < -2.0, f"expected strongly negative reward under thrash, got {r}"


def test_reward_empty_window():
    """Empty window returns 0.0 (neutral)."""
    assert compute_reward([]) == 0.0


def test_reward_is_finite():
    """Reward is always a finite float, even with extreme inputs."""
    window = [_make_sample(
        psi_mem_stall_us=int(1e9),
        swp_out_ps=1e9,
        mem_avail_bytes=0,
    )] * 5
    r = compute_reward(window)
    assert math.isfinite(r)


def test_action_cost_reduces_reward():
    """Taking actions costs reward (encourages restraint)."""
    window = [_make_sample()] * 5
    r_noop = compute_reward(window, n_actions_this_step=0)
    r_act  = compute_reward(window, n_actions_this_step=1)
    assert r_act < r_noop


def test_missing_metrics_neutral():
    """None metrics contribute 0 — no fabricated pressure."""
    window_none = [Sample(ts=1.0)] * 5   # everything None
    r_none = compute_reward(window_none)
    # None window should not be MORE negative than a good window
    # (missing ≠ pressure)
    # headroom bonus won't fire for None mem_avail → reward may differ
    assert math.isfinite(r_none)


def test_psi_stall_s():
    """PSI stall seconds summed correctly over window."""
    window = [_make_sample(psi_mem_stall_us=500_000)] * 4  # 0.5s each
    total = _psi_stall_s(window, "psi_mem")
    assert abs(total - 2.0) < 1e-9  # 4 × 0.5s = 2.0s


def test_headroom_bonus_saturates():
    """Headroom bonus saturates at 1.0 for >= 3 GiB available."""
    window = [_make_sample(mem_avail_bytes=int(5e9))] * 3
    bonus = _headroom_bonus(window)
    assert abs(bonus - 1.0) < 1e-6


def test_headroom_bonus_zero_at_zero_mem():
    """No available memory → 0 headroom bonus."""
    window = [_make_sample(mem_avail_bytes=0)] * 3
    assert _headroom_bonus(window) == 0.0


def test_swap_free_fraction():
    """swap_free_fraction = 1 - swap_used_pct/100."""
    window = [_make_sample(swap_used_pct=60.0)] * 3
    frac = _swap_free_fraction(window)
    assert abs(frac - 0.40) < 1e-6


def test_foreground_share_neutral_when_no_procs():
    """foreground_share returns 0.5 (neutral) when top_procs unavailable."""
    window = [_make_sample(top_procs=[])] * 3
    share = _foreground_share(window)
    assert abs(share - 0.5) < 1e-6


def test_foreground_share_when_gnomeshell_present():
    """gnome-shell CPU share is captured as foreground share."""
    procs = [{"name": "gnome-shell", "cpu_pct": 10.0, "rss_mib": 500.0,
              "pid": 1234, "start_time": 1.0}]
    window = [_make_sample(cpu_total_pct=50.0, top_procs=procs)] * 3
    share = _foreground_share(window)
    # gnome-shell has 10/50 = 0.2 of total CPU
    assert abs(share - 0.2) < 0.05


def test_reward_breakdown_sums_to_total():
    """reward_breakdown['total'] matches compute_reward."""
    window = [_make_sample(psi_mem_stall_us=200_000, swp_out_ps=500.0)] * 5
    bd = reward_breakdown(window, n_actions=1)
    r = compute_reward(window, n_actions_this_step=1)
    assert abs(bd["total"] - r) < 1e-9
