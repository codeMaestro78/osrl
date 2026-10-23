"""Reproducible MaskablePPO replay training with chronological evaluation.

Replay has no action-conditioned transitions. Its reward comparison is a
pipeline diagnostic, never evidence sufficient to promote a live policy.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sqlite3
import time
import uuid
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from osrl.rl.artifacts import normalization_path, publish_link, read_manifest
from osrl.rl.checkpoint_io import save_model
from osrl.rl.dataset import fingerprint, load_rows, quality_summary, split_rows
from osrl.rl.env import OSResourceEnv, StressScenario, _row_to_sample

DEFAULT_TRAIN_STEPS = 100_000
DEFAULT_CHECKPOINT_DIR = Path("~/os-rl-agent/checkpoints").expanduser()
DEFAULT_EVAL_EPISODES = 5
EVAL_GATE_REWARD_RATIO = 0.05
PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=1024,
    batch_size=128,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.97,
    clip_range=0.15,
    ent_coef=0.005,
    vf_coef=0.6,
    max_grad_norm=0.5,
    learning_rate=2e-4,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    verbose=1,
    device="cpu",
)


def _mask_fn(env):
    return env.action_masks()


def _make_env(db_path: str, synthetic: bool, seed: int = 0, sessions=None):
    def inner():
        return ActionMasker(
            OSResourceEnv(
                db_path=db_path,
                synthetic_stress=synthetic,
                stress_scenario=StressScenario(),
                seed=seed,
                sessions=sessions,
                allow_synthetic_fallback=False,
            ),
            _mask_fn,
        )

    return inner


def _write_event(conn, kind: str, payload: dict) -> None:
    try:
        conn.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (time.time(), kind, json.dumps(payload, sort_keys=True, allow_nan=False)),
        )
        conn.commit()
    except sqlite3.Error:
        pass  # A metrics outage must not discard a trained artifact.


def reward_gate(mean_reward: float, baseline_reward: float) -> bool:
    """Sign-correct threshold: a negative baseline must improve toward zero."""
    return (
        math.isfinite(mean_reward)
        and math.isfinite(baseline_reward)
        and mean_reward > baseline_reward + abs(baseline_reward) * EVAL_GATE_REWARD_RATIO
    )


def evaluate_replay(model, sessions, normalizer, n_episodes=5, seed=42) -> dict:
    """Paired deterministic episodes; frozen training stats, raw rewards.

    Each policy and NOOP see exactly the same session in each pair. Reward
    normalization is never used for reporting or the comparison threshold.
    """
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    policy_rewards, baseline_rewards, lengths = [], [], []
    for episode in range(n_episodes):
        # Cycle through all held-out sessions before repeating any of them.
        selected = [sessions[episode % len(sessions)]]
        pair = []
        for use_policy in (True, False):
            env = OSResourceEnv(
                "",
                synthetic_stress=False,
                sessions=selected,
                seed=seed + episode,
                allow_synthetic_fallback=False,
            )
            try:
                obs, _ = env.reset(seed=seed + episode)
                total, steps, done = 0.0, 0, False
                while not done:
                    action = 0
                    if use_policy:
                        obs_in = obs.reshape(1, -1)
                        if normalizer is not None:
                            obs_in = normalizer.normalize_obs(obs_in)
                        action, _ = model.predict(
                            obs_in,
                            deterministic=True,
                            action_masks=env.action_masks().reshape(1, -1),
                        )
                        action = int(np.asarray(action).item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total += reward
                    steps += 1
                pair.append(total)
            finally:
                env.close()
        policy_rewards.append(pair[0])
        baseline_rewards.append(pair[1])
        lengths.append(steps)
    mean = float(np.mean(policy_rewards))
    baseline = float(np.mean(baseline_rewards))
    return {
        "eval_mean_reward": mean,
        "eval_std_reward": float(np.std(policy_rewards)),
        "baseline_reward": baseline,
        "reward_delta": mean - baseline,
        "policy_episode_rewards": policy_rewards,
        "noop_episode_rewards": baseline_rewards,
        "episode_steps": lengths,
        "reward_gate_passed": reward_gate(mean, baseline),
        "validated": False,
        "validation_reason": "replay_has_no_action_conditioned_effects",
        "evaluation_kind": "chronological_heldout_replay",
        "seed": seed,
    }


def _sample_sessions(row_sessions):
    return [[_row_to_sample(row) for row in session] for session in row_sessions]


def _train_locked(
    db_path,
    n_steps,
    synthetic,
    checkpoint_dir,
    checkpoint_interval,
    eval_episodes,
    resume_path,
    seed,
):
    import torch

    # Bound native thread pools so a background training run leaves desktop capacity.
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    rows = load_rows(db_path)  # One immutable snapshot for the entire run.
    split = split_rows(rows)
    split.summary["eval_fingerprint"] = fingerprint([r for s in split.evaluation for r in s])
    if quality_summary(rows)["psi_valid_samples"] == 0:
        raise ValueError("No valid memory PSI readings; collect real pressure telemetry first")
    train_sessions = _sample_sessions(split.train)
    eval_sessions = _sample_sessions(split.evaluation)
    run_id = uuid.uuid4().hex[:12]
    run_dir = checkpoint_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_path, timeout=10)
    env = None
    _write_event(
        conn,
        "rl_train_start",
        {
            "run_id": run_id,
            "n_steps": n_steps,
            "synthetic": synthetic,
            "checkpoint_dir": str(run_dir),
            "resume_path": resume_path,
            "dataset": split.summary,
            "quality": quality_summary(rows),
        },
    )
    try:
        base = DummyVecEnv([_make_env(db_path, synthetic, seed, train_sessions)])
        if resume_path:
            path = Path(resume_path).expanduser().resolve()
            if not path.exists():
                raise ValueError(f"Resume checkpoint does not exist: {path}")
            prior = read_manifest(path).get("dataset", {})
            if prior.get("train_end_ts", float("inf")) >= split.summary["eval_start_ts"]:
                raise ValueError(
                    "Resume checkpoint provenance overlaps held-out data or is missing"
                )
            stats = normalization_path(path)
            # Legacy artifacts had a shared vecnorm.pkl. They cannot establish
            # which running statistics belong to this particular model.
            if not stats.exists():
                raise ValueError(
                    "Checkpoint has no matching normalization file; train from scratch"
                )
            env = VecNormalize.load(str(stats), base)
            env.training, env.norm_reward = True, True
            model = MaskablePPO.load(str(path), env=env, device="cpu")
        else:
            env = VecNormalize(base, training=True, norm_reward=True, gamma=PPO_KWARGS["gamma"])
            model = MaskablePPO(env=env, seed=seed, **PPO_KWARGS)
        start_steps = model.num_timesteps
        best_reward = -math.inf
        best_path = None
        while model.num_timesteps - start_steps < n_steps:
            remaining = n_steps - (model.num_timesteps - start_steps)
            model.learn(
                total_timesteps=min(checkpoint_interval, remaining), reset_num_timesteps=False
            )
            steps_done = model.num_timesteps - start_steps
            report = evaluate_replay(model, eval_sessions, env, eval_episodes, seed + 42)
            report.update(
                run_id=run_id,
                steps_done=steps_done,
                model_total_steps=model.num_timesteps,
                dataset=split.summary,
            )
            checkpoint = run_dir / f"ckpt_step{model.num_timesteps:08d}_UNVALIDATED.zip"
            save_model(model, checkpoint)
            env.save(str(normalization_path(checkpoint)))
            report.update(
                checkpoint=str(checkpoint), training_data_end_ts=split.summary["data_end_ts"]
            )
            checkpoint.with_suffix(".json").write_text(
                json.dumps(report, indent=2, allow_nan=False)
            )
            _write_event(conn, "rl_train_step", report)
            if report["eval_mean_reward"] > best_reward:
                best_reward, best_path = report["eval_mean_reward"], checkpoint
                publish_link(checkpoint_dir, "candidate", checkpoint.relative_to(checkpoint_dir))
            publish_link(checkpoint_dir, "latest", checkpoint.relative_to(checkpoint_dir))
            print(
                f"[osrl-train] step={steps_done} reward={report['eval_mean_reward']:.4f} "
                f"noop={report['baseline_reward']:.4f} UNVALIDATED",
                flush=True,
            )
        _write_event(
            conn,
            "rl_train_done",
            {
                "run_id": run_id,
                "best_reward": best_reward,
                "best_checkpoint": str(best_path),
                "total_steps": model.num_timesteps - start_steps,
                "training_data_end_ts": split.summary["data_end_ts"],
                "dataset": split.summary,
                "validated": False,
            },
        )
        return best_path
    except Exception as exc:
        _write_event(conn, "rl_train_failed", {"run_id": run_id, "error": str(exc)})
        raise
    finally:
        if env is not None:
            env.close()
        conn.close()


def train(
    db_path: str,
    *,
    n_steps=DEFAULT_TRAIN_STEPS,
    synthetic=True,
    checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
    checkpoint_interval=10_000,
    eval_episodes=DEFAULT_EVAL_EPISODES,
    resume_path=None,
    seed=0,
) -> Path:
    """Train one candidate; concurrent launchers share a per-database lock."""
    if min(n_steps, checkpoint_interval, eval_episodes) <= 0:
        raise ValueError("steps, checkpoint interval and evaluation episodes must be positive")
    db_path = str(Path(db_path).expanduser().resolve())
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(f"{db_path}.training.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Training is already running for this database") from exc
        return _train_locked(
            db_path,
            n_steps,
            synthetic,
            checkpoint_dir,
            checkpoint_interval,
            eval_episodes,
            resume_path,
            seed,
        )


def evaluate_checkpoint(path: Path, db_path: str, episodes=5) -> dict:
    path = path.expanduser().resolve()
    manifest = read_manifest(path)
    dataset = manifest.get("dataset", {})
    if "eval_start_ts" not in dataset or "data_end_ts" not in dataset:
        raise ValueError("Checkpoint lacks a recorded held-out evaluation range")
    rows = [
        r
        for r in load_rows(str(Path(db_path).expanduser().resolve()))
        if dataset["eval_start_ts"] <= r["ts"] <= dataset["data_end_ts"]
    ]
    from osrl.twin import sessions_from_rows

    sessions = _sample_sessions([s.rows for s in sessions_from_rows(rows) if len(s.rows) >= 20])
    if sum(map(len, sessions)) != dataset["eval_samples"]:
        raise ValueError(
            "Held-out telemetry has changed or expired; evaluation is not reproducible"
        )
    from osrl.twin import sessions_from_rows

    usable_rows = [r for s in sessions_from_rows(rows) if len(s.rows) >= 20 for r in s.rows]
    if dataset.get("eval_fingerprint") != fingerprint(usable_rows):
        raise ValueError("Held-out telemetry content changed; evaluation is not reproducible")
    base = DummyVecEnv([_make_env(db_path, False, sessions=sessions)])
    env = VecNormalize.load(str(normalization_path(path)), base)
    env.training, env.norm_reward = False, False
    try:
        return evaluate_replay(
            MaskablePPO.load(str(path), device="cpu"),
            sessions,
            env,
            episodes,
            manifest.get("seed", 42),
        )
    finally:
        env.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--resume")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    args = parser.parse_args(argv)
    # Only the dedicated CLI process changes its own scheduling priority.
    # Library callers (including tests and applications) retain theirs.
    try:
        os.nice(10)
    except OSError:
        pass
    try:
        if args.eval_only:
            directory = Path(args.checkpoint_dir).expanduser()
            checkpoint = Path(args.resume) if args.resume else directory / "latest"
            print(
                json.dumps(evaluate_checkpoint(checkpoint, args.db, args.eval_episodes), indent=2)
            )
        else:
            train(
                args.db,
                n_steps=args.steps,
                synthetic=args.synthetic,
                checkpoint_dir=Path(args.checkpoint_dir),
                resume_path=args.resume,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
            )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"[osrl-train] {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
