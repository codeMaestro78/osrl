"""Regression checks for replay, telemetry and evaluation integrity."""

import numpy as np
import pytest

from osrl.collector import Collector, CpuTimes, GpuSample, Sample
from osrl.rl.env import OSResourceEnv, _row_to_sample
from osrl.rl.obs import AgentState, encode_obs
from osrl.rl.reward import _oom_proximity_penalty, _trend_component
from osrl.store import Store
from osrl.worker import build_request_from_intent


def test_encoder_is_repeatable_with_same_agent():
    agent = AgentState()
    sample = Sample(ts=2, interval_s=2, cpu_total_pct=50)
    np.testing.assert_array_equal(encode_obs([sample], agent), encode_obs([sample], agent))


def test_gpu_survives_store_replay(tmp_path):
    store = Store(tmp_path / "gpu.db")
    try:
        store.insert_samples([Sample(ts=2, gpu=GpuSample(util_pct=42))])
        sample = _row_to_sample(store._conn.execute("SELECT * FROM samples").fetchone())
        assert sample.gpu is not None and sample.gpu.util_pct == 42
    finally:
        store.close()


def test_reset_exposes_real_initial_observation(tmp_path):
    store = Store(tmp_path / "replay.db")
    store.insert_samples([Sample(ts=i * 2, interval_s=2, cpu_total_pct=75) for i in range(40)])
    store.close()
    env = OSResourceEnv(str(tmp_path / "replay.db"), synthetic_stress=False)
    obs, _ = env.reset(seed=42)
    assert obs[0] == pytest.approx(0.75)


def test_zero_memory_is_maximal_oom_penalty():
    assert _oom_proximity_penalty([Sample(ts=0, mem_available_bytes=0)]) == 1


def test_swap_trend_includes_zero_endpoint():
    window = [Sample(ts=i, swap_used_pct=p) for i, p in enumerate([50, 50, 0, 0])]
    assert _trend_component(window) > 0


@pytest.mark.parametrize("payload", ["[]", "null", '{"pid":2,"start_time":1,"target_nice":"bad"}'])
def test_malformed_intent_is_rejected_without_exception(payload):
    assert build_request_from_intent({"kind": "restore", "payload": payload}) is None


def test_cpu_outage_does_not_diff_across_gap(machine):
    providers = machine.providers()
    original = providers.cpu_times
    collector = Collector(providers)
    collector.tick(ts=0)
    providers.cpu_times = lambda: (_ for _ in ()).throw(OSError("unavailable"))
    collector.tick(ts=2)
    providers.cpu_times = original
    machine.cpu_total = CpuTimes(user=200, idle=200)
    recovered, _ = collector.tick(ts=4)
    assert recovered.cpu_total_pct is None


def test_new_process_unknown_cpu_cannot_outrank_busy_process(machine):
    machine.set_proc(20, 1, "busy", 0, 20)
    collector = Collector(machine.providers())
    collector.tick(ts=0)
    machine.set_proc(20, 1, "busy", 2, 20)
    machine.set_proc(21, 2, "new", 0, 900)
    sample, _ = collector.tick(ts=2)
    assert sample.top_procs[0]["pid"] == 20


def test_cli_overrides_preserve_config(monkeypatch, tmp_path):
    from osrl import daemon

    captured = []
    monkeypatch.setenv("OSRL_RISK_ENTER_N", "9")
    monkeypatch.setenv("OSRL_GUARD_RENICE_STEP", "4")
    monkeypatch.setenv("OSRL_PSI_TRIGGER_ENABLED", "false")
    monkeypatch.setattr(daemon, "run", lambda cfg, **kw: captured.append(cfg))
    daemon.main(["--db", str(tmp_path / "db"), "--interval", "0.5", "--once"])
    assert captured[0].risk.enter_n == 9
    assert captured[0].guard.renice_step == 4
    assert not captured[0].psi_trigger.enabled
    daemon.main(["--interval", "0.25", "--once"])
    assert captured[1].collector.interval_s == 0.25


@pytest.mark.parametrize("age", [-1, float("nan"), float("inf")])
def test_guard_rejects_future_or_invalid_telemetry(age):
    from osrl.config import GuardConfig
    from osrl.guard import ActionRequest, Guard, GuardDeps

    guard = Guard(
        GuardConfig(),
        GuardDeps(
            self_pid=lambda: 999,
            self_euid=lambda: 1000,
            target_uid=lambda _: 1000,
            start_time_of=lambda _: 1.0,
            read_nice=lambda _: 0,
        ),
        sample_staleness=lambda _: age,
        last_action_ts=lambda *_: None,
        read_action=lambda _: None,
    )
    assert not guard.check(ActionRequest("renice_back", 55, 1), now=10).ok


def test_intent_reconcile_cannot_overwrite_completed_claim(tmp_path, monkeypatch):
    from contextlib import contextmanager

    path = tmp_path / "claims.db"
    store, rival = Store(path), Store(path)
    store.insert_intent("i", "renice_back", {}, 1)
    store.claim_next_pending(None, "worker", 2)
    transaction = store._txn

    @contextmanager
    def complete_before_transaction():
        assert rival.finish_intent("i", "applied", {}, 200)
        with transaction() as conn:
            yield conn

    monkeypatch.setattr(store, "_txn", complete_before_transaction)
    try:
        assert store.reconcile_intents(200) == []
        assert store._conn.execute("SELECT status FROM intents").fetchone()[0] == "applied"
    finally:
        rival.close()
        store.close()


def test_synthetic_overlay_cannot_reduce_real_pressure():
    from osrl.collector import PsiInterval
    from osrl.rl.env import StressScenario, _inject_stress

    sample = Sample(ts=1, interval_s=2, psi_mem=PsiInterval(some_stall_us=1800000))
    stressed = _inject_stress(sample, StressScenario(psi_mem_stall_ratio=0.4))
    assert stressed.psi_mem.some_stall_us == 1800000
    assert sample.psi_mem.some_stall_us == 1800000
