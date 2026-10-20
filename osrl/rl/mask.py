"""Action masker: conservative pre-policy filter mirroring guard semantics.

The mask is computed from observable state BEFORE the policy acts.
The guard remains the authoritative post-action veto; the mask is an
efficiency layer that prevents the policy from wasting exploration budget
on structurally illegal moves.

Action indices (must match env.py's action space definition):
  0  NOOP              — always legal
  1  ALERT             — always legal (no OS mutation)
  2  RENICE_BACK       — masked under several conditions (see below)
  3  REQUEST_APPROVAL  — always legal (no OS mutation)
  4  TUNE_THRESHOLDS   — always legal (self-adjustment only)

RENICE_BACK masking conditions (any one is sufficient):
  - risk_level == 'none'    : no evidence of pressure → no need to act
  - risk_level == 'unknown' : stale telemetry → abstain (matches guard check 5)
  - cooldown_active         : fewer than cooldown_steps steps since last action
  - restore_capable=False   : RESTORE is always masked (empirical EPERM)

Note: RESTORE is not currently an action in v1's Discrete(5) space.
      If the action space is later expanded to include RESTORE, add:
      mask[RESTORE_IDX] = False  when not restore_capable.
"""

from __future__ import annotations

import numpy as np

from osrl.rl.obs import N_ACTIONS, AgentState

# Action indices
NOOP              = 0
ALERT             = 1
RENICE_BACK       = 2
REQUEST_APPROVAL  = 3
TUNE_THRESHOLDS   = 4

# Default cooldown: mirror guard's min_spacing_s=300s / 20s decision cadence
DEFAULT_COOLDOWN_STEPS = 15   # 300s / 20s per step


def compute_action_mask(
    agent: AgentState,
    *,
    cooldown_steps: int = DEFAULT_COOLDOWN_STEPS,
    restore_capable: bool = False,
) -> np.ndarray:
    """Return a bool[N_ACTIONS] mask. True = action is legal this step.

    The mask is intentionally conservative: it may mask actions the guard
    would permit, but it must never UNMASK actions the guard would deny.
    "When in doubt, mask it out."
    """
    mask = np.ones(N_ACTIONS, dtype=bool)

    # RENICE_BACK: conditional
    renice_ok = True
    if agent.risk_level in ("none", "unknown"):
        renice_ok = False   # no pressure → no need; unknown → abstain
    if agent.steps_since_last_action < cooldown_steps:
        renice_ok = False   # within cooldown window
    mask[RENICE_BACK] = renice_ok

    # RESTORE: always masked in v1 (restore_capable=False empirically)
    # Not in current action space but kept here for future extension
    _ = restore_capable  # reserved

    return mask


def action_name(action: int) -> str:
    """Human-readable name for an action index."""
    names = {
        NOOP: "NOOP",
        ALERT: "ALERT",
        RENICE_BACK: "RENICE_BACK",
        REQUEST_APPROVAL: "REQUEST_APPROVAL",
        TUNE_THRESHOLDS: "TUNE_THRESHOLDS",
    }
    return names.get(action, f"UNKNOWN({action})")
