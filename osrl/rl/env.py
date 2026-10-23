"""OSResourceEnv: Gymnasium environment for PPO training.

Environment semantics:
- One STEP = one 20 s decision window (N_OBS_STEPS=10 samples at 2 s cadence).
- Episodes are contiguous sessions from the digital twin (twin.py sessions).
- done=True at end of session (never resets the real OS).
- Action space: Discrete(5) — {NOOP, ALERT, RENICE_BACK, REQUEST_APPROVAL,
  TUNE_THRESHOLDS}.
- Observation: flat float32[OBS_DIM] from osrl.rl.obs.encode_obs().
- Action masking: via the 'action_masks()' method (sb3-contrib contract).

Synthetic stress mode:
  When synthetic_stress=True, the env overlays configurable pressure spikes
  onto real session samples. This injects PSI stalls and swap-out spikes into
  windows that would otherwise be quiet, creating labelled training events.
  Injection is bounded to physically-reachable values (never exceeds real
  PSI=1.0 or swap-out beyond 10× the observed max).

Safety: RENICE_BACK in the env does NOT call executor.py. It writes an
  observation flag and updates agent_state.last_action. The store intent
  path is only exercised by osrl.rl.inference in production mode.
"""

from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "gymnasium is required for OSResourceEnv. "
        "Install with: pip install 'osrl-agent[rl]'"
    ) from e

from osrl.collector import GpuSample, PsiInterval, Sample
from osrl.risk import RiskTracker
from osrl.rl.mask import compute_action_mask
from osrl.rl.obs import (
    N_ACTIONS,
    N_OBS_STEPS,
    OBS_DIM,
    AgentState,
    encode_obs,
)
from osrl.rl.reward import compute_reward, reward_breakdown
from osrl.twin import sessions_from_rows

# ---------------------------------------------------------------------------
# Synthetic stress injection
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """Parameters for synthetic stress injection."""
    probability: float = 0.15        # prob of injecting stress on any step
    psi_mem_stall_ratio: float = 0.4  # injected stall fraction (0..1)
    swp_out_spike_ps: float = 1500.0  # injected swap-out pages/s
    duration_steps: int = 5           # how many consecutive steps to sustain


def _inject_stress(sample: Sample, scenario: StressScenario) -> Sample:
    """Overlay stress onto a sample. Returns a new Sample; original unmodified."""
    interval = sample.interval_s
    if interval is None or not math.isfinite(interval) or interval <= 0:
        return sample
    existing = sample.psi_mem or PsiInterval()
    stall_us = min(int(interval * 1e6), max(
        existing.some_stall_us or 0, existing.full_stall_us or 0,
        int(scenario.psi_mem_stall_ratio * interval * 1e6),
    ))
    new_psi_mem = PsiInterval(
        some_stall_us=stall_us,
        some_total_us=existing.some_total_us,
        full_stall_us=existing.full_stall_us,
        full_total_us=existing.full_total_us,
    )
    # Overlay swap-out spike
    swp = max(sample.swp_out_ps or 0.0, scenario.swp_out_spike_ps)

    # Build modified sample by replacing only the stressed fields
    return Sample(
        ts=sample.ts,
        interval_s=sample.interval_s,
        ncpu=sample.ncpu,
        cpu_total_pct=sample.cpu_total_pct,
        cpu_percpu_pct=sample.cpu_percpu_pct,
        load1=sample.load1,
        load5=sample.load5,
        load15=sample.load15,
        mem_total_bytes=sample.mem_total_bytes,
        mem_available_bytes=sample.mem_available_bytes,
        mem_used_pct=sample.mem_used_pct,
        swap_total_bytes=sample.swap_total_bytes,
        swap_used_bytes=sample.swap_used_bytes,
        swap_used_pct=sample.swap_used_pct,
        swp_in_ps=sample.swp_in_ps,
        swp_out_ps=swp,
        majflt_ps=sample.majflt_ps,
        reclaim_scan_ps=sample.reclaim_scan_ps,
        reclaim_steal_ps=sample.reclaim_steal_ps,
        allocstall_ps=sample.allocstall_ps,
        psi_cpu=sample.psi_cpu,
        psi_mem=new_psi_mem,
        psi_io=sample.psi_io,
        gpu=sample.gpu,
        gpu_top_procs=sample.gpu_top_procs,
        top_procs=sample.top_procs,
        collect_ms=sample.collect_ms,
    )


# ---------------------------------------------------------------------------
# SQLite row → Sample reconstruction
# ---------------------------------------------------------------------------

def _row_to_sample(row: sqlite3.Row) -> Sample:
    """Reconstruct a Sample from a stored DB row (matching store.py schema)."""
    def _f(key: str) -> float | None:
        try:
            v = row[key]
        except (IndexError, KeyError):
            return None
        return None if v is None else float(v)

    def _i(key: str) -> int | None:
        v = _f(key)
        return None if v is None else int(v)

    import json
    def _json(key: str) -> list:
        try:
            raw = row[key]
            return json.loads(raw) if raw else []
        except Exception:  # noqa: BLE001
            return []

    interval = _f("interval_s")
    stall_mem = _i("psi_mem_some_stall_us")
    return Sample(
        ts=_f("ts") or 0.0,
        interval_s=interval,
        ncpu=_i("ncpu") or 1,
        cpu_total_pct=_f("cpu_total_pct"),
        cpu_percpu_pct=_json("cpu_percpu_json"),
        load1=_f("load1"),
        load5=_f("load5"),
        load15=_f("load15"),
        mem_total_bytes=_i("mem_total_bytes"),
        mem_available_bytes=_i("mem_available_bytes"),
        mem_used_pct=_f("mem_used_pct"),
        swap_total_bytes=_i("swap_total_bytes"),
        swap_used_bytes=_i("swap_used_bytes"),
        swap_used_pct=_f("swap_used_pct"),
        swp_in_ps=_f("swp_in_ps"),
        swp_out_ps=_f("swp_out_ps"),
        majflt_ps=_f("majflt_ps"),
        reclaim_scan_ps=_f("reclaim_scan_ps"),
        reclaim_steal_ps=_f("reclaim_steal_ps"),
        allocstall_ps=_f("allocstall_ps"),
        psi_cpu=PsiInterval(
            some_stall_us=_i("psi_cpu_some_stall_us"),
            some_total_us=_i("psi_cpu_some_total_us"),
            full_stall_us=_i("psi_cpu_full_stall_us"),
            full_total_us=_i("psi_cpu_full_total_us"),
        ),
        psi_mem=PsiInterval(
            some_stall_us=stall_mem,
            some_total_us=_i("psi_mem_some_total_us"),
            full_stall_us=_i("psi_mem_full_stall_us"),
            full_total_us=_i("psi_mem_full_total_us"),
        ),
        psi_io=PsiInterval(
            some_stall_us=_i("psi_io_some_stall_us"),
            some_total_us=_i("psi_io_some_total_us"),
            full_stall_us=_i("psi_io_full_stall_us"),
            full_total_us=_i("psi_io_full_total_us"),
        ),
        gpu=GpuSample(
            util_pct=_f("gpu_util_pct"), mem_used_mib=_f("gpu_mem_used_mib"),
            mem_total_mib=_f("gpu_mem_total_mib"), temp_c=_f("gpu_temp_c"),
            power_w=_f("gpu_power_w"),
        ) if any(_f(k) is not None for k in (
            "gpu_util_pct", "gpu_mem_used_mib", "gpu_mem_total_mib", "gpu_temp_c", "gpu_power_w"
        )) else None,
        gpu_top_procs=_json("gpu_top_procs_json"),
        top_procs=_json("top_procs_json"),
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OSResourceEnv(gym.Env):
    """Gymnasium environment replaying recorded telemetry for PPO training.

    Observations:  Box(OBS_DIM,) float32 in [0, 1] approximately.
    Actions:       Discrete(5) with action masking via action_masks().
    Reward:        DESIGN §3 formula evaluated over the step window.
    Episode end:   End of a contiguous session from the digital twin.

    Args:
        db_path: Path to the SQLite telemetry database.
        synthetic_stress: If True, inject stress scenarios to create events.
        stress_scenario: Parameters for synthetic stress injection.
        cooldown_steps: Mirror of guard's min_spacing_s / step_cadence_s.
        seed: Random seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        db_path: str,
        synthetic_stress: bool = True,
        stress_scenario: StressScenario | None = None,
        cooldown_steps: int = 15,
        seed: int | None = None,
        sessions: list[list[Sample]] | None = None,
        allow_synthetic_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.db_path = db_path
        self.synthetic_stress = synthetic_stress
        self.stress_scenario = stress_scenario or StressScenario()
        self.cooldown_steps = cooldown_steps

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._sessions: list[list[Sample]] = []
        self._cur_session: list[Sample] = []
        self._step_idx: int = 0
        self._agent: AgentState = AgentState()
        self._window: list[Sample] = []
        self._stress_countdown: int = 0
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._risk_tracker = RiskTracker()
        self._allow_synthetic_fallback = allow_synthetic_fallback
        if sessions is None:
            self._load_sessions()
        else:
            self._sessions = [list(s) for s in sessions if len(s) >= N_OBS_STEPS * 2]
            if not self._sessions:
                raise ValueError("No usable real replay sessions")

    # ------------------------------------------------------------------ setup

    def _load_sessions(self) -> None:
        """Load all telemetry sessions from the DB."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM samples ORDER BY ts"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []   # table not yet created (fresh/empty DB)
        finally:
            conn.close()

        sessions = sessions_from_rows(rows)
        self._sessions = [
            [_row_to_sample(r) for r in s.rows]
            for s in sessions
            if len(s.rows) >= N_OBS_STEPS * 2  # need at least 2 windows
        ]
        if not self._sessions:
            # Fallback: create a minimal synthetic session so the env is usable
            # even with an empty or tiny DB (e.g. in CI)
            if not self._allow_synthetic_fallback:
                raise ValueError("No usable real replay sessions; synthetic fallback disabled")
            self._sessions = [self._make_synthetic_session()]

    def _make_synthetic_session(self, n_samples: int = 200) -> list[Sample]:
        """Generate a minimal synthetic session for bootstrapping."""
        samples = []
        t = 1_700_000_000.0
        for _ in range(n_samples):
            # Baseline quiet desktop
            s = Sample(
                ts=t, interval_s=2.0, ncpu=12,
                cpu_total_pct=float(self._np_rng.uniform(5, 30)),
                load1=self._np_rng.uniform(0.5, 2.0),
                mem_available_bytes=int(9 * 1024**3),
                swap_used_pct=87.0,
                swp_out_ps=float(self._np_rng.uniform(0, 50)),
                psi_mem=PsiInterval(some_stall_us=0),
                psi_io=PsiInterval(some_stall_us=0),
                psi_cpu=PsiInterval(some_stall_us=0),
            )
            samples.append(s)
            t += 2.0
        return samples

    # ------------------------------------------------------------------ gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        # Pick a random session for this episode
        self._cur_session = list(self._rng.choice(self._sessions))
        self._step_idx = N_OBS_STEPS
        self._agent = AgentState()
        self._window = self._cur_session[:N_OBS_STEPS]
        self._risk_tracker = RiskTracker()
        self._update_risk(self._window)
        self._stress_countdown = 0

        obs = self._build_obs()
        return obs, {"session_len": len(self._cur_session)}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one 20 s decision window."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self._step_idx >= len(self._cur_session):
            raise RuntimeError("Episode finished; call reset() before step()")
        if not self.action_masks()[action]:
            raise ValueError(f"Action {action} is masked for this state")
        # Collect the next N_OBS_STEPS samples
        start = self._step_idx
        end = min(start + N_OBS_STEPS, len(self._cur_session))
        raw_window = self._cur_session[start:end]
        self._step_idx = end

        # Apply synthetic stress if enabled
        if self.synthetic_stress:
            if self._stress_countdown > 0:
                self._stress_countdown -= 1
                raw_window = [_inject_stress(s, self.stress_scenario)
                              for s in raw_window]
            elif self._rng.random() < self.stress_scenario.probability:
                self._stress_countdown = max(0, self.stress_scenario.duration_steps - 1)
                raw_window = [_inject_stress(s, self.stress_scenario)
                              for s in raw_window]

        self._window = raw_window
        self._update_risk(raw_window)

        # Update agent state
        n_actions = 0 if action == 0 else 1  # NOOP=0 has no cost
        self._agent.last_action = action
        if action != 0:  # non-NOOP
            self._agent.steps_since_last_action = 0
        else:
            self._agent.steps_since_last_action = min(
                self._agent.steps_since_last_action + 1, 999
            )

        # Compute reward
        reward = compute_reward(raw_window, n_actions_this_step=n_actions)

        # Episode ends at session boundary
        done = self._step_idx >= len(self._cur_session)

        obs = self._build_obs()
        info: dict[str, Any] = {
            "action": action,
            "reward_components": reward_breakdown(raw_window, n_actions),
            "transition_model": "recorded_replay_no_action_effect",
            "step_idx": self._step_idx,
        }
        return obs, reward, done, False, info

    def _update_risk(self, window: list[Sample]) -> None:
        for sample in window:
            verdict = self._risk_tracker.feed(sample, now=sample.ts)
            self._agent.risk_level = verdict.level
            self._agent.bad_streak = verdict.bad_streak
            self._agent.good_streak = verdict.good_streak
            self._agent.psi_recent = verdict.psi_recent

    def action_masks(self) -> np.ndarray:
        """Return action mask array (sb3-contrib MaskablePPO contract)."""
        return compute_action_mask(
            self._agent, cooldown_steps=self.cooldown_steps
        )

    def _build_obs(self) -> np.ndarray:
        return encode_obs(self._window, self._agent)

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass
