"""Real tiny PPO runs exercise save, resume, normalization and evaluation."""

import fcntl
import json

import pytest

from osrl.collector import PsiInterval, Sample
from osrl.rl import auto_train
from osrl.rl.artifacts import is_validated, normalization_path, read_manifest
from osrl.rl.dataset import split_rows
from osrl.rl.obs import AgentState
from osrl.store import Store


@pytest.fixture
def replay_db(tmp_path):
    path = tmp_path / "telemetry.db"
    store = Store(path)
    store.insert_samples(
        [
            Sample(
                ts=1000 + i * 2,
                interval_s=2,
                cpu_total_pct=20 + i % 15,
                mem_available_bytes=4 * 1024**3,
                swap_used_pct=20,
                swp_out_ps=10,
                psi_mem=PsiInterval(some_stall_us=1000),
            )
            for i in range(200)
        ]
    )
    store.close()
    return path


def test_split_has_no_overlap_and_embargo(replay_db):
    from osrl.rl.dataset import load_rows

    split = split_rows(load_rows(str(replay_db)))
    assert split.summary["train_samples"] == 160
    assert split.summary["eval_samples"] == 30
    assert split.summary["eval_start_ts"] - split.summary["train_end_ts"] == 22


def test_unvalidated_filename_is_never_validated():
    assert not is_validated("ckpt_step100_UNVALIDATED.zip")
    assert is_validated("ckpt_step100_VALIDATED.zip")


def test_negative_baseline_gate_cannot_promote_worse_reward():
    from osrl.rl.train import reward_gate

    assert not reward_gate(-102, -100)
    assert not reward_gate(-96, -100)
    assert reward_gate(-94, -100)
    assert not reward_gate(float("nan"), 1)


def test_training_save_resume_and_eval(replay_db, tmp_path, monkeypatch):
    from osrl.rl import train as training

    monkeypatch.setattr(training.os, "nice", lambda *_: None)
    monkeypatch.setitem(training.PPO_KWARGS, "n_steps", 16)
    monkeypatch.setitem(training.PPO_KWARGS, "batch_size", 8)
    monkeypatch.setitem(training.PPO_KWARGS, "n_epochs", 1)
    monkeypatch.setitem(training.PPO_KWARGS, "policy_kwargs", {"net_arch": [16]})
    directory = tmp_path / "checkpoints"
    first = training.train(
        str(replay_db),
        n_steps=32,
        checkpoint_interval=16,
        eval_episodes=2,
        checkpoint_dir=directory,
        synthetic=False,
    )
    assert first.exists() and normalization_path(first).exists()
    assert (directory / "candidate").resolve() == first.resolve()
    assert read_manifest(directory / "latest")["steps_done"] == 32
    assert not (directory / "best").exists()
    manifest = read_manifest(first)
    assert manifest["validated"] is False
    assert manifest["dataset"]["eval_start_ts"] > manifest["dataset"]["train_end_ts"]
    result = training.evaluate_checkpoint(first, str(replay_db), episodes=2)
    assert result["policy_episode_rewards"] == pytest.approx(manifest["policy_episode_rewards"])
    assert result["eval_mean_reward"] <= result["baseline_reward"] + 1e-6
    second = training.train(
        str(replay_db),
        n_steps=16,
        checkpoint_interval=16,
        eval_episodes=2,
        checkpoint_dir=directory,
        resume_path=str(first),
        synthetic=False,
    )
    assert second != first and first.exists()
    assert read_manifest(second)["model_total_steps"] > manifest["model_total_steps"]
    from osrl.rl.inference import PolicyInference

    policy = PolicyInference(directory)
    assert policy._model is not None
    assert policy._validated is False
    suggestion = policy.suggest(
        [Sample(ts=1, interval_s=2, cpu_total_pct=20)], AgentState(), "advise"
    )
    assert not suggestion.validated
    assert suggestion.uncertainty.policy_entropy is not None
    assert policy.write_intent_if_advised(suggestion, None, []) is None


def test_training_lock_prevents_duplicate_run(replay_db, tmp_path):
    from osrl.rl.train import train

    with open(f"{replay_db}.training.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            train(str(replay_db), n_steps=1, checkpoint_dir=tmp_path / "ckpt")
        assert auto_train._is_training_running(str(replay_db))
    assert not auto_train._is_training_running(str(replay_db))


def test_empty_data_does_not_train_synthetic_policy(tmp_path):
    from osrl.rl.train import train

    path = tmp_path / "empty.db"
    store = Store(path)
    store.close()
    with pytest.raises(ValueError, match="real telemetry"):
        train(str(path), n_steps=1, checkpoint_dir=tmp_path / "ckpt")


def test_scheduler_logs_to_file_instead_of_unread_pipe(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(auto_train.subprocess, "Popen", lambda *a, **kw: calls.append((a, kw)))
    auto_train._launch_training("db", 100, True, tmp_path, False)
    _, kwargs = calls[0]
    assert kwargs["stdout"] != auto_train.subprocess.PIPE
    assert (tmp_path / "training.log").exists()


def test_scheduler_uses_snapshot_watermark_not_completion_time(replay_db):
    import sqlite3

    with sqlite3.connect(replay_db) as conn:
        conn.execute(
            "INSERT INTO events(ts,kind,payload) VALUES (99999, 'rl_train_done',?)",
            (json.dumps({"training_data_end_ts": 1300}),),
        )
        assert auto_train._last_data_ts(conn) == 1300
        assert auto_train._samples_since(conn, 1300) == 49
        auto_train._save_state(conn, {"last_attempt_ts": 5000, "last_exit_code": 1})
        assert auto_train._scheduler_state(conn)["last_attempt_ts"] == 5000


def test_scheduler_failure_backoff_survives_restart(replay_db, tmp_path, monkeypatch):
    monkeypatch.setattr(auto_train.os, "nice", lambda *_: None)
    monkeypatch.setattr(auto_train.signal, "signal", lambda *a: None)
    monkeypatch.setattr(auto_train.time, "time", lambda: 1400.0)
    monkeypatch.setattr(auto_train, "_is_training_running", lambda *_: False)
    launches = []

    class FailedRun:
        pid = 12345

        def poll(self):
            return 1

    def launch(*args):
        launches.append(args)
        return FailedRun()

    def run_polls():
        count = 0

        def poll_wait(_seconds):
            nonlocal count
            count += 1
            if count >= 3:
                auto_train._STOP = True

        monkeypatch.setattr(auto_train._WAKE, "wait", poll_wait)
        auto_train.run_scheduler(
            str(replay_db),
            steps=16,
            min_samples=50,
            min_interval_s=100,
            poll_s=1,
            checkpoint_dir=tmp_path / "checkpoints",
        )

    monkeypatch.setattr(auto_train, "_launch_training", launch)
    run_polls()
    run_polls()
    assert len(launches) == 1  # failed attempt still cools down after restart


def test_re_evaluation_rejects_modified_data(replay_db, tmp_path, monkeypatch):
    from osrl.rl import train as training

    monkeypatch.setattr(training.os, "nice", lambda *_: None)
    for key, value in {
        "n_steps": 16,
        "batch_size": 8,
        "n_epochs": 1,
        "policy_kwargs": {"net_arch": [16]},
    }.items():
        monkeypatch.setitem(training.PPO_KWARGS, key, value)
    checkpoint = training.train(
        str(replay_db), n_steps=16, eval_episodes=1, checkpoint_dir=tmp_path / "ckpt"
    )
    store = Store(replay_db)
    store._conn.execute("UPDATE samples SET cpu_total_pct=99 WHERE ts=1398")
    store.close()
    with pytest.raises(ValueError, match="content changed"):
        training.evaluate_checkpoint(checkpoint, str(replay_db), 1)
