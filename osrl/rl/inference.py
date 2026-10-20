"""Policy inference + intent bridge: loads a checkpoint and drives the
Observe / Advise / Autopilot mode gate.

Mode gate (stored in store.meta['agent_mode'], written by Streamlit):
  observe   — inference runs, suggestions recorded to events, NO intents written
  advise    — suggestions written to intents table (status='pending');
              human approves from dashboard
  autopilot — intent written AND immediately claimed by the inference worker;
              guard/worker/executor chain is the last safety gate

Structural guarantees:
  - This module NEVER imports osrl.guard, osrl.worker, osrl.executor.
  - The guard chain is called only by osrl.worker (independent process).
  - Advisory suggestions write to intents; execution is always mediated by
    the guard chain even in autopilot mode.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as e:
    raise ImportError(
        "sb3-contrib is required. Install with: pip install 'osrl-agent[rl]'"
    ) from e

if TYPE_CHECKING:
    from osrl.store import Store

from osrl.rl.mask import NOOP, RENICE_BACK, action_name, compute_action_mask
from osrl.rl.obs import OBS_DIM, AgentState, encode_obs

# ---------------------------------------------------------------------------

VALID_MODES = ("observe", "advise", "autopilot")


@dataclass
class ActionSuggestion:
    """Policy output with full audit trail."""
    action: int
    action_name: str
    probability: float         # policy's confidence
    mask: list[bool]           # which actions were legal this step
    obs_vector: list[float]    # raw obs for dashboard display
    agent_mode: str            # 'observe' | 'advise' | 'autopilot'
    validated: bool            # whether the loaded checkpoint is VALIDATED
    checkpoint_path: str
    ts: float


class PolicyInference:
    """Loads a MaskablePPO checkpoint and runs advisory inference.

    Typical lifecycle:
        inf = PolicyInference(checkpoint_dir)
        suggestion = inf.suggest(window, agent_state, store)
        if written := inf.write_intent_if_advised(suggestion, store, top_procs):
            print(f"Intent {written} queued for approval")
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        vecnorm_path: str | Path | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir).expanduser()
        self._model: MaskablePPO | None = None
        self._vecnorm: VecNormalize | None = None
        self._validated = False
        self._ckpt_path = ""
        self._load(vecnorm_path)

    # ----------------------------------------------------------------- loading

    def _load(self, vecnorm_path: str | Path | None) -> None:
        best = self.checkpoint_dir / "best"
        if not best.exists():
            return   # no checkpoint yet; inference will return NOOP
        ckpt = str(best)
        self._validated = "VALIDATED" in (best.resolve().name)
        self._ckpt_path = str(best.resolve())

        # Minimal dummy env to satisfy SB3 load API (obs/action space only)
        import gymnasium as gym
        from gymnasium import spaces

        class _MinimalEnv(gym.Env):
            observation_space = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
            action_space = spaces.Discrete(5)
            def reset(self, **kw): return np.zeros(OBS_DIM, np.float32), {}
            def step(self, a): return np.zeros(OBS_DIM, np.float32), 0.0, True, False, {}

        dummy = DummyVecEnv([lambda: _MinimalEnv()])
        vn_path = vecnorm_path or self.checkpoint_dir / "vecnorm.pkl"
        if Path(str(vn_path)).exists():
            self._vecnorm = VecNormalize.load(str(vn_path), dummy)
            self._vecnorm.training = False
            self._vecnorm.norm_reward = False

        try:
            self._model = MaskablePPO.load(ckpt, device="cpu")
        except Exception:  # noqa: BLE001
            self._model = None

    # ---------------------------------------------------------------- inference

    def suggest(
        self,
        window: list,           # list[Sample]
        agent: AgentState,
        mode: str = "observe",
    ) -> ActionSuggestion:
        """Run policy inference for the current observation window."""
        obs = encode_obs(window, agent)
        mask = compute_action_mask(agent)

        if self._model is None or not any(mask):
            # No model or all actions masked → NOOP
            return ActionSuggestion(
                action=NOOP, action_name="NOOP",
                probability=1.0, mask=mask.tolist(),
                obs_vector=obs.tolist(),
                agent_mode=mode, validated=self._validated,
                checkpoint_path=self._ckpt_path, ts=time.time(),
            )

        # Normalise observation if vecnorm is available
        obs_in = obs.reshape(1, -1)
        if self._vecnorm is not None:
            obs_in = self._vecnorm.normalize_obs(obs_in)

        action, _states = self._model.predict(
            obs_in, action_masks=mask.reshape(1, -1), deterministic=False
        )
        action_idx = int(action[0]) if hasattr(action, "__len__") else int(action)

        # Extract action probability for audit trail
        try:
            import torch
            with torch.no_grad():
                obs_t = torch.tensor(obs_in, dtype=torch.float32)
                dist = self._model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.numpy().flatten()
            prob = float(probs[action_idx])
        except Exception:  # noqa: BLE001
            prob = float("nan")

        return ActionSuggestion(
            action=action_idx, action_name=action_name(action_idx),
            probability=prob, mask=mask.tolist(),
            obs_vector=obs.tolist(),
            agent_mode=mode, validated=self._validated,
            checkpoint_path=self._ckpt_path, ts=time.time(),
        )

    # --------------------------------------------------------- intent writing

    def write_intent_if_advised(
        self,
        suggestion: ActionSuggestion,
        store: Store,
        top_procs: list | None = None,
    ) -> str | None:
        """Write an intent to the store if mode allows it.

        Returns the intent id if written, None otherwise.

        RENICE_BACK intents include the top-CPU process from top_procs as the
        target. The guard chain validates ownership and identity before any
        execution.
        """
        if suggestion.agent_mode == "observe":
            return None  # advisory only; record suggestion to events below
        if suggestion.action == NOOP:
            return None  # NOOP never writes an intent
        if suggestion.action != RENICE_BACK:
            # ALERT and REQUEST_APPROVAL are dashboard-only (events table)
            return None

        # RENICE_BACK: identify top-CPU non-protected process as candidate
        target = _pick_renice_target(top_procs or [])
        if target is None:
            return None

        intent_id = str(uuid.uuid4())
        payload = {
            "kind": "renice_back",
            "pid": target["pid"],
            "start_time": target["start_time"],
            "proposed_by": "policy",
            "policy_action_name": suggestion.action_name,
            "checkpoint": suggestion.checkpoint_path,
            "validated": suggestion.validated,
        }
        store.insert_intent(
            id=intent_id,
            kind="renice_back",
            payload=payload,
            created_ts=suggestion.ts,
        )
        return intent_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Protected process name fragments that must never be targeted
_PROTECTED_NAMES = frozenset({
    "gnome-shell", "xorg", "xwayland", "systemd", "kworker",
    "ksoftirqd", "osrl", "python",  # never renice ourselves
})


def _pick_renice_target(top_procs: list[dict]) -> dict | None:
    """Pick the highest-CPU process that is not protected and has nice >= 0."""
    for proc in top_procs:
        name = str(proc.get("name", "")).lower()
        if any(p in name for p in _PROTECTED_NAMES):
            continue
        cpu = proc.get("cpu_pct")
        if cpu is None or float(cpu) < 5.0:
            continue
        return proc
    return None


def current_agent_mode(store: Store) -> str:
    """Read the current agent mode from store.meta. Defaults to 'observe'."""
    try:
        row = store._conn.execute(
            "SELECT value FROM meta WHERE key='agent_mode'"
        ).fetchone()
        if row and row[0] in VALID_MODES:
            return str(row[0])
    except Exception:  # noqa: BLE001
        pass
    return "observe"
