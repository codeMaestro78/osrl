"""Tests for osrl.rl.env — Gymnasium environment."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from osrl.rl.env import OSResourceEnv, StressScenario, _row_to_sample
from osrl.rl.obs import N_ACTIONS, OBS_DIM
from osrl.store import Store

# ---------------------------------------------------------------------------
# Fixture: minimal in-memory DB with enough samples for the env to work
# ---------------------------------------------------------------------------

def _make_test_db() -> str:
    """Create a temp DB with synthetic samples and return its path."""
    tmp = tempfile.mktemp(suffix=".db")
    store = Store(tmp)
    # Insert 60 samples (enough for 3 sessions of 20 steps each)

    from osrl.collector import PsiInterval, Sample
    t = 1_700_000_000.0
    batch = []
    for i in range(60):
        s = Sample(
            ts=t, interval_s=2.0, ncpu=12,
            cpu_total_pct=float(20 + i % 10),
            mem_available_bytes=int(9e9),
            swap_used_pct=87.0,
            swp_out_ps=float(i * 5),
            psi_mem=PsiInterval(some_stall_us=i * 1000),
            psi_io=PsiInterval(some_stall_us=0),
            psi_cpu=PsiInterval(some_stall_us=0),
        )
        batch.append(s)
        t += 2.0
    store.insert_samples(batch)
    store.close()
    return tmp


@pytest.fixture(scope="module")
def db_path():
    path = _make_test_db()
    yield path
    Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_env_resets_correctly(db_path):
    """reset() returns obs of correct shape and info dict."""
    env = OSResourceEnv(db_path, synthetic_stress=False, seed=0)
    obs, info = env.reset()
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert "session_len" in info
    env.close()


def test_env_step_returns_correct_types(db_path):
    """step() returns (obs, reward, done, truncated, info) with correct types."""
    env = OSResourceEnv(db_path, synthetic_stress=False, seed=1)
    env.reset()
    obs, reward, done, truncated, info = env.step(0)  # NOOP
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert "action" in info
    env.close()


def test_env_episode_terminates(db_path):
    """Episode terminates (done=True) when session is exhausted."""
    env = OSResourceEnv(db_path, synthetic_stress=False, seed=2)
    env.reset()
    done = False
    steps = 0
    while not done and steps < 1000:
        _, _, done, _, _ = env.step(0)
        steps += 1
    assert done, "episode never terminated"
    assert steps < 200   # should not take forever
    env.close()


def test_action_masks_shape(db_path):
    """action_masks() returns bool array of length N_ACTIONS."""
    env = OSResourceEnv(db_path, synthetic_stress=False, seed=3)
    env.reset()
    mask = env.action_masks()
    assert mask.shape == (N_ACTIONS,)
    assert mask.dtype == bool
    env.close()


def test_action_masks_noop_always_available(db_path):
    """NOOP (action 0) is always in the mask."""
    env = OSResourceEnv(db_path, synthetic_stress=False, seed=4)
    env.reset()
    mask = env.action_masks()
    assert mask[0], "NOOP should always be available"
    env.close()


def test_env_reward_is_finite(db_path):
    """All rewards from a full episode are finite floats."""
    env = OSResourceEnv(db_path, synthetic_stress=False, seed=5)
    env.reset()
    done = False
    while not done:
        _, r, done, _, _ = env.step(0)
        assert np.isfinite(r), f"non-finite reward: {r}"
    env.close()


def test_obs_values_in_range(db_path):
    """All obs values should be in approximately [-1, 1]."""
    env = OSResourceEnv(db_path, synthetic_stress=False, seed=6)
    obs, _ = env.reset()
    for _ in range(5):
        obs, _, done, _, _ = env.step(0)
        assert obs.max() <= 1.0 + 1e-5
        assert obs.min() >= -1.0 - 1e-5
        if done:
            break
    env.close()


def test_synthetic_stress_injects_pressure(db_path):
    """Synthetic stress mode produces more negative rewards than no-stress."""
    rewards_no_stress = []
    rewards_stressed = []

    for use_stress in (False, True):
        env = OSResourceEnv(
            db_path, synthetic_stress=use_stress,
            stress_scenario=StressScenario(probability=1.0),  # always inject
            seed=42
        )
        env.reset()
        done = False
        ep_r = 0.0
        steps = 0
        while not done and steps < 20:
            _, r, done, _, _ = env.step(0)
            ep_r += r
            steps += 1
        (rewards_stressed if use_stress else rewards_no_stress).append(ep_r)
        env.close()

    # Stressed episodes should yield lower total reward
    assert sum(rewards_stressed) <= sum(rewards_no_stress) + 0.1, (
        "synthetic stress should decrease reward"
    )


def test_env_works_with_empty_db():
    """Env falls back to synthetic session when DB is empty."""
    tmp = tempfile.mktemp(suffix=".db")
    try:
        store = Store(tmp)
        store.close()
        env = OSResourceEnv(tmp, synthetic_stress=False, seed=0)
        obs, _ = env.reset()
        assert obs.shape == (OBS_DIM,)
        env.close()
    finally:
        Path(tmp).unlink(missing_ok=True)


def test_multiple_resets_work(db_path):
    """Multiple reset() calls work without errors."""
    env = OSResourceEnv(db_path, synthetic_stress=False, seed=7)
    for _ in range(5):
        obs, _ = env.reset()
        assert obs.shape == (OBS_DIM,)
    env.close()


def test_row_to_sample_handles_none_fields():
    """_row_to_sample handles rows with None fields gracefully."""
    # Create a minimal sqlite3.Row-like dict
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE samples (ts REAL, interval_s REAL, ncpu INTEGER,"
        " cpu_total_pct REAL, cpu_percpu_json TEXT,"
        " load1 REAL, load5 REAL, load15 REAL,"
        " mem_total_bytes INTEGER, mem_available_bytes INTEGER,"
        " mem_used_pct REAL, swap_total_bytes INTEGER, swap_used_bytes INTEGER,"
        " swap_used_pct REAL, swp_in_ps REAL, swp_out_ps REAL, majflt_ps REAL,"
        " reclaim_scan_ps REAL, reclaim_steal_ps REAL, allocstall_ps REAL,"
        " psi_cpu_some_stall_us INTEGER, psi_cpu_some_total_us INTEGER,"
        " psi_cpu_full_stall_us INTEGER, psi_cpu_full_total_us INTEGER,"
        " psi_mem_some_stall_us INTEGER, psi_mem_some_total_us INTEGER,"
        " psi_mem_full_stall_us INTEGER, psi_mem_full_total_us INTEGER,"
        " psi_io_some_stall_us INTEGER, psi_io_some_total_us INTEGER,"
        " psi_io_full_stall_us INTEGER, psi_io_full_total_us INTEGER,"
        " top_procs_json TEXT, gpu_top_procs_json TEXT)"
    )
    conn.execute(
        "INSERT INTO samples (ts, interval_s, ncpu) VALUES (1.0, 2.0, 12)"
    )
    row = conn.execute("SELECT * FROM samples").fetchone()
    sample = _row_to_sample(row)
    assert sample.ts == 1.0
    assert sample.interval_s == 2.0
    assert sample.ncpu == 12
    assert sample.cpu_total_pct is None
    conn.close()
