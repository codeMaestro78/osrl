"""Tests for osrl.rl.mask — action masker."""

from __future__ import annotations

import numpy as np

from osrl.rl.mask import (
    ALERT,
    NOOP,
    RENICE_BACK,
    REQUEST_APPROVAL,
    TUNE_THRESHOLDS,
    compute_action_mask,
)
from osrl.rl.obs import N_ACTIONS, AgentState


def _agent(**kw) -> AgentState:
    defaults = dict(risk_level="elevated", steps_since_last_action=999)
    defaults.update(kw)
    return AgentState(**defaults)


def test_mask_shape():
    """Mask is always bool array of length N_ACTIONS."""
    mask = compute_action_mask(_agent())
    assert mask.shape == (N_ACTIONS,)
    assert mask.dtype == bool


def test_noop_always_available():
    """NOOP is never masked."""
    for level in ("none", "elevated", "critical", "unknown"):
        mask = compute_action_mask(_agent(risk_level=level))
        assert mask[NOOP], f"NOOP masked at risk={level}"


def test_alert_always_available():
    """ALERT is never masked (no OS mutation)."""
    for level in ("none", "elevated", "critical", "unknown"):
        mask = compute_action_mask(_agent(risk_level=level))
        assert mask[ALERT], f"ALERT masked at risk={level}"


def test_request_approval_always_available():
    """REQUEST_APPROVAL is never masked (dashboard-only)."""
    mask = compute_action_mask(_agent())
    assert mask[REQUEST_APPROVAL]


def test_tune_thresholds_always_available():
    """TUNE_THRESHOLDS is never masked (self-adjustment)."""
    mask = compute_action_mask(_agent())
    assert mask[TUNE_THRESHOLDS]


def test_renice_back_available_at_elevated():
    """RENICE_BACK is legal at elevated risk when past cooldown."""
    mask = compute_action_mask(_agent(
        risk_level="elevated", steps_since_last_action=999
    ))
    assert mask[RENICE_BACK]


def test_renice_back_available_at_critical():
    """RENICE_BACK is legal at critical risk when past cooldown."""
    mask = compute_action_mask(_agent(
        risk_level="critical", steps_since_last_action=999
    ))
    assert mask[RENICE_BACK]


def test_renice_back_masked_at_none():
    """RENICE_BACK is masked when risk is 'none' (no pressure)."""
    mask = compute_action_mask(_agent(risk_level="none"))
    assert not mask[RENICE_BACK]


def test_renice_back_masked_at_unknown():
    """RENICE_BACK is masked when risk is 'unknown' (stale telemetry)."""
    mask = compute_action_mask(_agent(risk_level="unknown"))
    assert not mask[RENICE_BACK]


def test_renice_back_masked_during_cooldown():
    """RENICE_BACK is masked within the cooldown window."""
    mask = compute_action_mask(_agent(
        risk_level="elevated",
        steps_since_last_action=3,  # well within default cooldown=15
    ), cooldown_steps=15)
    assert not mask[RENICE_BACK]


def test_renice_back_available_after_cooldown():
    """RENICE_BACK is legal after the cooldown expires."""
    mask = compute_action_mask(_agent(
        risk_level="elevated",
        steps_since_last_action=15,  # exactly at cooldown boundary
    ), cooldown_steps=15)
    assert mask[RENICE_BACK]


def test_custom_cooldown_respected():
    """Custom cooldown_steps parameter controls the threshold."""
    agent = _agent(risk_level="elevated", steps_since_last_action=5)
    mask_tight = compute_action_mask(agent, cooldown_steps=10)
    mask_loose = compute_action_mask(agent, cooldown_steps=4)
    assert not mask_tight[RENICE_BACK]   # 5 < 10 → masked
    assert mask_loose[RENICE_BACK]       # 5 >= 4 → available


def test_mask_is_numpy_bool_array():
    """Mask dtype is bool for sb3-contrib compatibility."""
    mask = compute_action_mask(_agent())
    assert mask.dtype == bool
    assert isinstance(mask, np.ndarray)
