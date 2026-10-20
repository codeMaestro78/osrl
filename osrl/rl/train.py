"""MaskablePPO training loop for the OS resource management policy.

Training pipeline:
  1. Load sessions from store (real telemetry + synthetic stress injection).
  2. Wrap env in DummyVecEnv + VecNormalize.
  3. Train MaskablePPO for N steps with resource capping (nice +10 on self).
  4. Checkpoint every checkpoint_interval_steps to checkpoints/ dir.
  5. Eval: run held-out sessions, compare reward vs NOOP baseline.
  6. If eval beats threshold: stamp checkpoint 'VALIDATED'; else 'UNVALIDATED'.
  7. Write training metrics to store.events (kind='rl_train_step', etc.).

Structural guarantees:
  - This module is NEVER imported by daemon.py (AST-enforced test).
  - No RL imports bleed into the guard chain.
  - Training runs at nice +10 (E-core priority) — never competes with desktop.
  - GPU is never used (MlpPolicy, CPU-only torch build).

Usage:
  osrl-train --db data/telemetry.db [--steps 100000] [--synthetic]
             [--checkpoint-dir checkpoints/] [--eval-only]
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.evaluation import evaluate_policy as mask_eval
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as e:
    raise ImportError(
        "sb3-contrib is required for training. "
        "Install with: pip install 'osrl-agent[rl]'"
    ) from e

from osrl.rl.env import OSResourceEnv, StressScenario

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_STEPS    = 100_000
DEFAULT_CHECKPOINT_DIR = Path("~/os-rl-agent/checkpoints").expanduser()
DEFAULT_EVAL_EPISODES  = 5
EVAL_GATE_REWARD_RATIO = 0.05   # must beat NOOP by at least this fraction

# PPO hyperparameters (tuned for a small MLP on CPU)
PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=1024,
    batch_size=128,
    n_epochs=15,
    gamma=0.995,
    gae_lambda=0.97,
    clip_range=0.15,
    clip_range_vf=0.15,
    ent_coef=0.005,
    vf_coef=0.6,
    max_grad_norm=0.5,
    learning_rate=2e-4,
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn_kwargs={},
    ),
    verbose=1,
    device="cpu",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_fn(env: OSResourceEnv) -> np.ndarray:
    return env.action_masks()


def _make_env(db_path: str, synthetic: bool, seed: int = 0):
    def _inner():
        env = OSResourceEnv(
            db_path=db_path,
            synthetic_stress=synthetic,
            stress_scenario=StressScenario(),
            seed=seed,
        )
        return ActionMasker(env, _mask_fn)
    return _inner


def _checkpoint_path(checkpoint_dir: Path, step: int, validated: bool) -> Path:
    status = "VALIDATED" if validated else "UNVALIDATED"
    return checkpoint_dir / f"ckpt_step{step:08d}_{status}"


def _write_event(conn: sqlite3.Connection, kind: str, payload: dict) -> None:
    """Write a training event to the store's events table."""
    try:
        conn.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(payload, sort_keys=True)),
        )
        conn.commit()
    except Exception:  # noqa: BLE001
        pass   # metrics writes must never kill training


def _eval_policy(
    model: MaskablePPO,
    db_path: str,
    n_episodes: int,
    seed: int = 42,
) -> tuple[float, float]:
    """Evaluate policy on a separate env instance; return (mean_reward, std)."""
    eval_env = DummyVecEnv([_make_env(db_path, synthetic=False, seed=seed)])
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
    mean_r, std_r = mask_eval(
        model, eval_env,
        n_eval_episodes=n_episodes,
        warn=False,
    )
    eval_env.close()
    return float(mean_r), float(std_r)


def _noop_baseline_reward(db_path: str, n_episodes: int = 3) -> float:
    """Estimate mean reward of always-NOOP policy for the eval gate."""
    from osrl.rl.env import OSResourceEnv  # local import to keep top clear
    rewards = []
    for ep in range(n_episodes):
        env = OSResourceEnv(db_path=db_path, synthetic_stress=False, seed=ep)
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            obs, r, done, _, _ = env.step(0)  # always NOOP
            ep_reward += r
        rewards.append(ep_reward)
        env.close()
    return float(sum(rewards) / len(rewards)) if rewards else 0.0


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(
    db_path: str,
    *,
    n_steps: int = DEFAULT_TRAIN_STEPS,
    synthetic: bool = True,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    checkpoint_interval: int = 10_000,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    resume_path: str | None = None,
) -> Path:
    """Run the full training loop. Returns path to the best checkpoint."""
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Self-renice to E-core priority: training must not compete with desktop
    try:
        os.nice(10)
    except OSError:
        pass   # no permission on some setups; proceed regardless

    # Store connection for metrics
    db_conn = sqlite3.connect(db_path)

    _write_event(db_conn, "rl_train_start", {
        "n_steps": n_steps, "synthetic": synthetic,
        "checkpoint_dir": str(checkpoint_dir),
        "resume_path": resume_path,
    })

    # Build vectorised env
    env = DummyVecEnv([_make_env(db_path, synthetic=synthetic, seed=0)])
    env = VecNormalize(env, training=True, norm_reward=True)

    # Build or load model
    if resume_path and Path(resume_path).exists():
        print(f"[osrl-train] Resuming from {resume_path}")
        model = MaskablePPO.load(resume_path, env=env, device="cpu")
        vec_norm_path = Path(resume_path).parent / "vecnorm.pkl"
        if vec_norm_path.exists():
            env = VecNormalize.load(str(vec_norm_path), env)
    else:
        model = MaskablePPO(env=env, **PPO_KWARGS)

    # Baseline reward (for eval gate)
    print("[osrl-train] Estimating NOOP baseline reward …")
    baseline_reward = _noop_baseline_reward(db_path)
    print(f"[osrl-train] NOOP baseline mean reward: {baseline_reward:.4f}")

    best_ckpt: Path | None = None
    best_reward = -math.inf
    steps_done = 0

    print(f"[osrl-train] Training for {n_steps:,} steps …")
    while steps_done < n_steps:
        chunk = min(checkpoint_interval, n_steps - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=(steps_done == 0))
        steps_done += chunk

        # Checkpoint
        mean_r, std_r = _eval_policy(model, db_path, eval_episodes)
        validated = mean_r > baseline_reward * (1.0 + EVAL_GATE_REWARD_RATIO)
        ckpt = _checkpoint_path(checkpoint_dir, steps_done, validated)
        model.save(str(ckpt))
        env.save(str(ckpt.parent / "vecnorm.pkl"))

        _write_event(db_conn, "rl_train_step", {
            "steps_done": steps_done,
            "eval_mean_reward": mean_r,
            "eval_std_reward": std_r,
            "baseline_reward": baseline_reward,
            "validated": validated,
            "checkpoint": str(ckpt),
        })

        print(
            f"[osrl-train] step={steps_done:,}  eval_reward={mean_r:.4f}±{std_r:.4f}"
            f"  {'✓ VALIDATED' if validated else '✗ UNVALIDATED'}"
        )

        if mean_r > best_reward:
            best_reward = mean_r
            best_ckpt = ckpt
            # Symlink best checkpoint for easy loading
            best_link = checkpoint_dir / "best"
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(ckpt.name)

    _write_event(db_conn, "rl_train_done", {
        "best_reward": best_reward,
        "best_checkpoint": str(best_ckpt),
        "total_steps": steps_done,
    })
    db_conn.close()

    print(f"\n[osrl-train] Done. Best checkpoint: {best_ckpt}")
    return best_ckpt or checkpoint_dir / "none"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="osrl-train",
        description="Train MaskablePPO policy on recorded OS telemetry.",
    )
    p.add_argument("--db", required=True, help="SQLite telemetry DB path")
    p.add_argument(
        "--steps", type=int, default=DEFAULT_TRAIN_STEPS,
        help=f"Total PPO training timesteps (default {DEFAULT_TRAIN_STEPS:,})",
    )
    p.add_argument(
        "--synthetic", action="store_true",
        help="Enable synthetic stress injection for curriculum training",
    )
    p.add_argument(
        "--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR),
        help=f"Directory for checkpoints (default {DEFAULT_CHECKPOINT_DIR})",
    )
    p.add_argument(
        "--resume", default=None,
        help="Path to existing checkpoint to resume from",
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Only run evaluation on the latest checkpoint, skip training",
    )
    args = p.parse_args(argv)

    if args.eval_only:
        best_link = Path(args.checkpoint_dir) / "best"
        if not best_link.exists():
            print("[osrl-train] No best checkpoint found. Run training first.")
            return 1
        mean_r, std_r = _eval_policy(str(best_link), args.db, 5)
        print(f"Eval on {best_link}: mean={mean_r:.4f} ± {std_r:.4f}")
        return 0

    train(
        db_path=args.db,
        n_steps=args.steps,
        synthetic=args.synthetic,
        checkpoint_dir=Path(args.checkpoint_dir),
        resume_path=args.resume,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
