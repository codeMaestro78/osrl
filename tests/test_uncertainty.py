"""Uncertainty assessment and fail-closed recommendation behavior."""

import json
import time

from osrl.collector import PsiInterval
from osrl.rl.inference import (
    ActionSuggestion,
    PolicyInference,
    RecommendationAssessment,
    assess_recommendation,
    run_inference_once,
)
from osrl.rl.mask import RENICE_BACK, action_name
from osrl.rl.obs import AgentState
from osrl.store import Store
from tests.conftest import make_sample


class OutcomeCounts:
    def __init__(self, count, age=1.0):
        self.count = count
        self.age = age

    def sample_staleness(self, now=None):
        return self.age

    def relevant_outcome_count(self, action, risk_level):
        assert action == "RENICE_BACK"
        assert risk_level in {"none", "critical"}
        return self.count


def pressure_sample(ts=100.0):
    return make_sample(
        ts=ts,
        interval_s=2.0,
        psi_mem=PsiInterval(some_stall_us=1_400_000),
        mem_available_bytes=512 * 1024**2,
    )


def pressure_window(last_ts):
    return [pressure_sample(last_ts - 18 + index * 2) for index in range(10)]


def test_uncertainty_explains_stale_telemetry_signal_disagreement_and_sparse_outcomes():
    window = pressure_window(80.0)
    assessment = assess_recommendation(
        window,
        AgentState(risk_level="none"),
        RENICE_BACK,
        probability=0.55,
        entropy=0.9,
        store=OutcomeCounts(1, age=20.0),
        now=100.0,
    )

    assert assessment.eligible is False
    assert set(assessment.reasons) == {
        "stale_telemetry",
        "stale_persisted_telemetry",
        "risk_signals_disagree",
        "insufficient_relevant_outcomes",
        "low_policy_probability",
        "ambiguous_policy_distribution",
    }
    assert assessment.policy_probability_calibrated is False


def test_fresh_agreeing_state_with_relevant_outcomes_can_be_assessed_eligible():
    assessment = assess_recommendation(
        pressure_window(100.0),
        AgentState(risk_level="critical"),
        RENICE_BACK,
        probability=0.82,
        entropy=0.25,
        store=OutcomeCounts(5, age=1.0),
        now=101.0,
    )

    assert assessment.eligible is True
    assert assessment.reasons == ()
    assert assessment.relevant_outcomes == 5
    assert assessment.policy_probability_calibrated is False


def test_uncertain_suggestion_is_logged_and_does_not_create_intent(tmp_path):
    store = Store(tmp_path / "uncertainty.db")
    try:
        checkpoint = tmp_path / "candidate.zip"
        inference = PolicyInference(tmp_path / "no-checkpoint")
        # Isolate the admission gate: these flags represent a loaded validated
        # artifact; the test still exercises the public intent-writing method.
        inference._validated = True
        inference._ckpt_path = str(checkpoint)
        suggestion = ActionSuggestion(
            action=RENICE_BACK,
            action_name=action_name(RENICE_BACK),
            probability=0.9,
            mask=[True, True, True, True, True],
            obs_vector=[],
            agent_mode="advise",
            validated=True,
            checkpoint_path=str(checkpoint),
            ts=100.0,
            uncertainty=RecommendationAssessment(
                reasons=("insufficient_relevant_outcomes",),
                telemetry_age_s=0.5,
                relevant_outcomes=0,
                risk_level="critical",
            ),
        )

        assert inference.write_intent_if_advised(
            suggestion, store, [{"pid": 42, "start_time": 1.0,
                                 "name": "renderer", "cpu_pct": 80.0}]
        ) is None
        assert store.pending_intent_count() == 0
        [event] = store._conn.execute(
            "SELECT payload FROM events WHERE kind='policy_suggestion'"
        ).fetchall()
        payload = json.loads(event["payload"])
        assert payload["uncertainty_abstain"] is True
        assert payload["uncertainty"]["reasons"] == ["insufficient_relevant_outcomes"]
    finally:
        store.close()


def test_one_shot_inference_records_recommendation_without_execution(tmp_path):
    store = Store(tmp_path / "inference.db")
    now = 1_800_000_000.0
    try:
        store.insert_samples([
            make_sample(
                ts=now - 18 + index * 2,
                psi_mem=PsiInterval(some_stall_us=0),
                mem_available_bytes=8 * 1024**3,
            )
            for index in range(10)
        ])
        result = run_inference_once(
            store, tmp_path / "no-checkpoint", mode="advise", now=now
        )

        assert result["action"] == "NOOP"
        assert result["intent_id"] is None
        assert result["executed"] is False
        assert store.pending_intent_count() == 0
        assert store._conn.execute(
            "SELECT COUNT(*) FROM events WHERE kind='policy_suggestion'"
        ).fetchone()[0] == 1
    finally:
        store.close()


def test_fresh_recommendation_is_rechecked_before_queuing(tmp_path):
    class StoreWithStaleLatestSample:
        def __init__(self):
            self.events = []
            self.inserted = False

        def sample_staleness(self, now=None):
            return 20.0

        def relevant_outcome_count(self, action, risk_level):
            return 5

        def record_event(self, ts, kind, payload):
            self.events.append((kind, payload))

        def insert_intent(self, **kwargs):
            self.inserted = True

    checkpoint = tmp_path / "validated.zip"
    inference = PolicyInference(tmp_path / "no-checkpoint")
    inference._validated = True
    inference._ckpt_path = str(checkpoint)
    store = StoreWithStaleLatestSample()
    suggestion = ActionSuggestion(
        action=RENICE_BACK,
        action_name=action_name(RENICE_BACK),
        probability=0.95,
        mask=[True, True, True, True, True],
        obs_vector=[],
        agent_mode="advise",
        validated=True,
        checkpoint_path=str(checkpoint),
        ts=time.time(),
        uncertainty=RecommendationAssessment(
            reasons=(),
            telemetry_age_s=1.0,
            relevant_outcomes=5,
            risk_level="critical",
        ),
    )

    assert inference.write_intent_if_advised(
        suggestion, store,
        [{"pid": 42, "start_time": 1.0, "name": "renderer", "cpu_pct": 80.0}],
    ) is None
    assert store.inserted is False
    assert store.events[0][1]["intent_gate_reason"] == "telemetry_stale_at_admission"
