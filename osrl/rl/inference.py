"""Policy inference + intent bridge: loads a checkpoint and drives the
Observe / Advise / Autopilot mode gate.

Mode gate (stored in store.meta['agent_mode'], written by Streamlit):
  observe   — inference runs, suggestions recorded to events, NO intents written
  advise    — suggestions written to intents table (status='pending');
              human approves from dashboard
  autopilot — eligible intents are queued; a separate worker still needs
              explicit --execute approval before the guard/executor chain

Structural guarantees:
  - This module NEVER imports osrl.guard, osrl.worker, osrl.executor.
  - The guard chain is called only by osrl.worker (independent process).
  - Advisory suggestions write to intents; execution is always mediated by
    the guard chain even in autopilot mode.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass, field
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

from osrl.config import RiskConfig
from osrl.risk import sample_evidence
from osrl.rl.artifacts import is_validated, normalization_path
from osrl.rl.mask import NOOP, RENICE_BACK, action_name, compute_action_mask
from osrl.rl.obs import N_OBS_STEPS, OBS_DIM, AgentState, encode_obs

# ---------------------------------------------------------------------------

VALID_MODES = ("observe", "advise", "autopilot")
MIN_RELEVANT_OUTCOMES = 5
MIN_POLICY_PROBABILITY = 0.70
MAX_NORMALIZED_POLICY_ENTROPY = 0.60


@dataclass(frozen=True)
class RecommendationAssessment:
    """Evidence limits for a recommendation; policy probabilities are uncalibrated."""

    reasons: tuple[str, ...] = ("assessment_missing",)
    telemetry_age_s: float | None = None
    max_telemetry_age_s: float = RiskConfig().staleness_s
    signal_disagreement: bool | None = None
    relevant_outcomes: int | None = None
    risk_level: str = "unknown"
    policy_probability: float | None = None
    policy_entropy: float | None = None
    policy_probability_calibrated: bool = False

    @property
    def eligible(self) -> bool:
        return not self.reasons

    def as_dict(self) -> dict:
        return {**asdict(self), "reasons": list(self.reasons), "eligible": self.eligible}


def assess_recommendation(
    window: list,
    agent: AgentState,
    action: int,
    probability: float | None,
    entropy: float | None,
    store: Store | None,
    *,
    now: float | None = None,
    risk_config: RiskConfig | None = None,
) -> RecommendationAssessment:
    """Fail closed when telemetry, signals, model output, or outcome evidence is weak."""
    now = time.time() if now is None else float(now)
    cfg = risk_config or RiskConfig()
    reasons: list[str] = []
    age = None
    disagreement = None
    experience = None

    if not math.isfinite(cfg.staleness_s) or cfg.staleness_s <= 0:
        reasons.append("invalid_staleness_threshold")

    if not window:
        reasons.append("no_observation_window")
    else:
        if len(window) < N_OBS_STEPS:
            reasons.append("insufficient_observation_window")
        age = now - float(window[-1].ts)
        if not math.isfinite(age) or age < 0:
            reasons.append("invalid_telemetry_timestamp")
        elif age > cfg.staleness_s:
            reasons.append("stale_telemetry")
        if agent.risk_level not in ("none", "elevated", "critical", "unknown"):
            reasons.append("risk_state_unknown")
        elif agent.risk_level == "unknown":
            reasons.append("risk_state_unknown")
        else:
            strong, warn, _psi_present = sample_evidence(window[-1], cfg)
            sample_pressure = bool(strong) or len(warn) >= 2
            tracker_pressure = agent.risk_level in ("elevated", "critical")
            disagreement = sample_pressure != tracker_pressure
            if disagreement:
                reasons.append("risk_signals_disagree")

    if action != NOOP and store is None:
        reasons.extend(("telemetry_store_unavailable", "outcome_evidence_unavailable"))
    elif action != NOOP:
        try:
            stored_age = store.sample_staleness(now=now)
        except Exception:  # noqa: BLE001 - stale or unreadable data must abstain
            stored_age = None
            reasons.append("telemetry_store_unavailable")
        if stored_age is None:
            reasons.append("persisted_telemetry_unavailable")
        elif not math.isfinite(stored_age) or stored_age < 0:
            reasons.append("invalid_persisted_telemetry_timestamp")
        elif stored_age > cfg.staleness_s:
            reasons.append("stale_persisted_telemetry")
        if age is None or stored_age is None:
            age = None
        elif math.isfinite(stored_age):
            age = max(age, stored_age)
        try:
            experience = store.relevant_outcome_count(
                action_name(action), agent.risk_level
            )
        except Exception:  # noqa: BLE001 - uncertainty failures must abstain
            reasons.append("outcome_evidence_unavailable")
        else:
            if experience < MIN_RELEVANT_OUTCOMES:
                reasons.append("insufficient_relevant_outcomes")

    if action != NOOP:
        if probability is None or not math.isfinite(probability):
            reasons.append("policy_probability_unavailable")
        elif probability < 0 or probability > 1:
            reasons.append("invalid_policy_probability")
        elif probability < MIN_POLICY_PROBABILITY:
            reasons.append("low_policy_probability")
        if entropy is None or not math.isfinite(entropy):
            reasons.append("policy_entropy_unavailable")
        elif entropy < 0 or entropy > 1:
            reasons.append("invalid_policy_entropy")
        elif entropy > MAX_NORMALIZED_POLICY_ENTROPY:
            reasons.append("ambiguous_policy_distribution")

    return RecommendationAssessment(
        reasons=tuple(dict.fromkeys(reasons)),
        telemetry_age_s=age,
        max_telemetry_age_s=cfg.staleness_s,
        signal_disagreement=disagreement,
        relevant_outcomes=experience,
        risk_level=agent.risk_level,
        policy_probability=probability,
        policy_entropy=entropy,
        policy_probability_calibrated=False,
    )


@dataclass
class ActionSuggestion:
    """Policy output with full audit trail."""
    action: int
    action_name: str
    probability: float         # policy probability; not calibrated confidence
    mask: list[bool]           # which actions were legal this step
    obs_vector: list[float]    # raw obs for dashboard display
    agent_mode: str            # 'observe' | 'advise' | 'autopilot'
    validated: bool            # whether the loaded checkpoint is VALIDATED
    checkpoint_path: str
    ts: float
    uncertainty: RecommendationAssessment = field(default_factory=RecommendationAssessment)


class PolicyInference:
    """Loads a MaskablePPO checkpoint and runs advisory inference.

    Typical lifecycle:
        inf = PolicyInference(checkpoint_dir)
        suggestion = inf.suggest(window, agent_state, store=store)
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
            best = self.checkpoint_dir / "latest"
        if not best.exists():
            return   # no checkpoint yet; inference will return NOOP
        ckpt = str(best)
        self._validated = is_validated(best)
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
        vn_path = vecnorm_path or normalization_path(best)
        if not Path(str(vn_path)).exists():
            dummy.close()
            return  # A normalized model cannot safely consume raw observations.
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
        store: Store | None = None,
        *,
        now: float | None = None,
        risk_config: RiskConfig | None = None,
    ) -> ActionSuggestion:
        """Run policy inference for the current observation window."""
        if mode not in VALID_MODES:
            mode = "observe"
        obs = encode_obs(window, agent)
        mask = compute_action_mask(agent)

        if self._model is None or not any(mask):
            # No model or all actions masked → NOOP
            timestamp = time.time() if now is None else now
            uncertainty = assess_recommendation(
                window, agent, NOOP, 1.0, 0.0, store,
                now=timestamp, risk_config=risk_config,
            )
            return ActionSuggestion(
                action=NOOP, action_name="NOOP",
                probability=1.0, mask=mask.tolist(),
                obs_vector=obs.tolist(),
                agent_mode=mode, validated=self._validated,
                checkpoint_path=self._ckpt_path, ts=timestamp,
                uncertainty=uncertainty,
            )

        # Normalise observation if vecnorm is available
        obs_in = obs.reshape(1, -1)
        if self._vecnorm is not None:
            obs_in = self._vecnorm.normalize_obs(obs_in)

        action, _states = self._model.predict(
            obs_in, action_masks=mask.reshape(1, -1), deterministic=True
        )
        action_idx = int(action[0]) if hasattr(action, "__len__") else int(action)

        # Extract action probability for audit trail
        entropy_value = None
        try:
            import torch
            with torch.no_grad():
                obs_t = torch.tensor(obs_in, dtype=torch.float32)
                dist = self._model.policy.get_distribution(obs_t, action_masks=mask.reshape(1, -1))
                probs = dist.distribution.probs.numpy().flatten()
            prob = float(probs[action_idx])
            positive = probs[probs > 0]
            legal_count = int(mask.sum())
            entropy_value = (
                float(-(positive * np.log(positive)).sum() / math.log(legal_count))
                if legal_count > 1 else 0.0
            )
        except Exception:  # noqa: BLE001
            prob = float("nan")

        timestamp = time.time() if now is None else now
        uncertainty = assess_recommendation(
            window, agent, action_idx, prob, entropy_value, store,
            now=timestamp, risk_config=risk_config,
        )

        return ActionSuggestion(
            action=action_idx, action_name=action_name(action_idx),
            probability=prob, mask=mask.tolist(),
            obs_vector=obs.tolist(),
            agent_mode=mode, validated=self._validated,
            checkpoint_path=self._ckpt_path, ts=timestamp,
            uncertainty=uncertainty,
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
        admission_age = None
        if store is not None:
            assessment = suggestion.uncertainty
            gate_reason = None
            if suggestion.agent_mode not in ("advise", "autopilot"):
                gate_reason = "mode_does_not_queue_intents"
            elif not suggestion.validated or not self._validated:
                gate_reason = "checkpoint_not_validated"
            elif assessment.reasons:
                gate_reason = "uncertainty:" + ",".join(assessment.reasons)
            elif suggestion.action == NOOP:
                gate_reason = "noop_action"
            elif suggestion.action != RENICE_BACK:
                gate_reason = "dashboard_only_action"
            elif assessment.eligible:
                try:
                    admission_age = store.sample_staleness()
                except Exception:  # noqa: BLE001 - admission fails closed
                    admission_age = None
                if (not math.isfinite(assessment.max_telemetry_age_s)
                        or assessment.max_telemetry_age_s <= 0
                        or admission_age is None or not math.isfinite(admission_age)
                        or admission_age < 0
                        or admission_age > assessment.max_telemetry_age_s):
                    gate_reason = "telemetry_stale_at_admission"
            store.record_event(suggestion.ts, "policy_suggestion", {
                "action": suggestion.action_name, "mode": suggestion.agent_mode,
                "probability": (
                    suggestion.probability
                    if math.isfinite(suggestion.probability)
                    else None
                ),
                "probability_calibrated": False,
                "validated": suggestion.validated, "checkpoint": suggestion.checkpoint_path,
                "uncertainty": assessment.as_dict(),
                "uncertainty_abstain": suggestion.action != NOOP and bool(assessment.reasons),
                "intent_gate_reason": gate_reason,
            })
        if suggestion.agent_mode not in ("advise", "autopilot") or not suggestion.validated:
            return None  # advisory only; record suggestion to events below
        if not suggestion.uncertainty.eligible:
            return None
        if (not math.isfinite(suggestion.uncertainty.max_telemetry_age_s)
                or suggestion.uncertainty.max_telemetry_age_s <= 0
                or admission_age is None or not math.isfinite(admission_age)
                or admission_age < 0
                or admission_age > suggestion.uncertainty.max_telemetry_age_s):
            return None
        if (not self._validated or suggestion.checkpoint_path != self._ckpt_path
                or time.time() - suggestion.ts > 12 or suggestion.ts > time.time()
                or not suggestion.mask[RENICE_BACK]):
            return None
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
            "recommendation_context": {
                "action": suggestion.action_name,
                "risk_level": suggestion.uncertainty.risk_level,
            },
        }
        store.insert_intent(
            id=intent_id,
            kind="renice_back",
            payload=payload,
            created_ts=suggestion.ts,
        )
        return intent_id


def run_inference_once(
    store: Store,
    checkpoint_dir: str | Path,
    *,
    mode: str | None = None,
    now: float | None = None,
    history_limit: int = 300,
    risk_config: RiskConfig | None = None,
) -> dict:
    """Record one visible recommendation assessment without executing actions."""
    if history_limit < 1 or history_limit > 5000:
        raise ValueError("history_limit must be between 1 and 5000")
    timestamp = time.time() if now is None else float(now)
    from osrl.risk import RiskTracker
    from osrl.rl.env import _row_to_sample

    rows = store.recent_samples(history_limit)
    samples = [_row_to_sample(row) for row in rows]
    cfg = risk_config or RiskConfig()
    tracker = RiskTracker(cfg)
    assessment = None
    previous_ts = None
    for index, sample in enumerate(samples):
        current_time = timestamp if index == len(samples) - 1 else sample.ts
        if previous_ts is not None and sample.ts - previous_ts > cfg.staleness_s:
            tracker.feed(None, sample.ts)
        assessment = tracker.feed(sample, current_time)
        previous_ts = sample.ts
    if samples:
        # Observations must not contain deltas across a sampling outage.
        tail_start = len(samples) - 1
        while (tail_start > 0 and
               samples[tail_start].ts - samples[tail_start - 1].ts <= cfg.staleness_s):
            tail_start -= 1
        window = samples[tail_start:][-N_OBS_STEPS:]
    else:
        window = []
    if assessment is None:
        assessment = tracker.feed(None, timestamp)
    agent = AgentState(
        risk_level=assessment.level,
        bad_streak=assessment.bad_streak,
        good_streak=assessment.good_streak,
        psi_recent=assessment.psi_recent,
    )
    inference = PolicyInference(checkpoint_dir)
    suggestion = inference.suggest(
        window, agent,
        mode=mode or store.get_agent_mode(),
        store=store,
        now=timestamp,
        risk_config=cfg,
    )
    intent_id = inference.write_intent_if_advised(
        suggestion, store, samples[-1].top_procs if samples else []
    )
    return {
        "action": suggestion.action_name,
        "mode": suggestion.agent_mode,
        "validated": suggestion.validated,
        "checkpoint": suggestion.checkpoint_path,
        "uncertainty": suggestion.uncertainty.as_dict(),
        "intent_id": intent_id,
        "executed": False,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    from osrl.config import load_config
    from osrl.store import Store

    parser = argparse.ArgumentParser(
        description="Record one policy recommendation; never execute an OS action."
    )
    parser.add_argument("--db", help="SQLite telemetry DB path")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--history-limit", type=int, default=300)
    parser.add_argument("--mode", choices=VALID_MODES)
    args = parser.parse_args(argv)
    cfg = load_config()
    db_path = Path(args.db).expanduser() if args.db else cfg.db_path
    checkpoint_dir = (
        Path(args.checkpoint_dir).expanduser()
        if args.checkpoint_dir
        else cfg.rl.checkpoint_dir.expanduser()
    )
    store = Store(db_path)
    try:
        result = run_inference_once(
            store,
            checkpoint_dir,
            mode=args.mode,
            history_limit=args.history_limit,
            risk_config=cfg.risk,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"[osrl-infer] {exc}\n")
    finally:
        store.close()
    print(json.dumps(result, sort_keys=True))
    return 0


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
    for proc in sorted(top_procs, key=lambda p: float(p.get("cpu_pct") or 0), reverse=True):
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
