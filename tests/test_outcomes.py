"""Public Store outcome tracking behavior for guarded intervention attempts."""

import json

from osrl.collector import PsiInterval
from osrl.store import Store
from tests.conftest import make_sample


def test_rejected_attempt_is_recorded_without_claiming_it_changed_telemetry(tmp_path):
    store = Store(tmp_path / "outcomes.db")
    try:
        assert store.record_action_outcome(
            intent_id="intent-rejected",
            action_id=None,
            status="rejected",
            ts=100.0,
            details={"reason": "stale telemetry"},
        )

        [outcome] = store.recent_action_outcomes()
        assert outcome["intent_id"] == "intent-rejected"
        assert outcome["status"] == "not_applied"
        assert outcome["measured"] is False
        assert outcome["details"]["reason"] == "stale telemetry"
    finally:
        store.close()


def test_applied_action_is_measured_after_a_complete_pre_and_post_window(tmp_path):
    store = Store(tmp_path / "outcomes.db")
    try:
        store.insert_action(
            id="action-1",
            created_ts=100.0,
            proposed_by="operator",
            kind="renice_back",
            target_pid=123,
            target_start_time=1.0,
            target_nice=10,
            before_nice=0,
            applied_nice=10,
            intent_id="intent-1",
            status="applied",
            ts_applied=100.0,
        )
        assert store.record_action_outcome(
            "intent-1", "action-1", "applied", 100.0,
            {"recommendation_context": {"risk_level": "critical", "action": "RENICE_BACK"}},
        )

        samples = []
        for ts in range(40, 221, 2):
            # Sustained pressure before/during the intervention, a real recovery,
            # then renewed pressure inside the outcome window.
            pressured = ts < 130 or ts >= 202
            samples.append(make_sample(
                ts=float(ts),
                interval_s=2.0,
                cpu_total_pct=70.0 if pressured else 25.0,
                mem_available_bytes=(512 if pressured else 4096) * 1024**2,
                psi_mem=PsiInterval(some_stall_us=1_200_000 if pressured else 0),
                top_procs=[{"pid": 123, "cpu_pct": 30.0 if pressured else 10.0}],
            ))
        store.insert_samples(samples)

        # The pipeline waits for the post-action window and a collector flush grace.
        assert store.finalize_action_outcomes(now=234.0) == []
        finalized = store.finalize_action_outcomes(now=235.0)
        assert len(finalized) == 1
        [outcome] = store.recent_action_outcomes()
        assert outcome["status"] == "measured"
        assert outcome["measured"] is True
        assert outcome["pressure_before"] in ("elevated", "critical")
        assert outcome["recovered_seconds"] is not None
        assert outcome["pressure_returned"] is True
        assert outcome["foreground_impact"] is None
        assert "not collected" in outcome["foreground_impact_note"]
        assert outcome["target_cpu_pct_delta"] is not None
        details = json.loads(outcome["details_json"])
        assert details["recommendation_context"]["risk_level"] == "critical"

        # Finalization is idempotent and only measured applied outcomes count
        # toward action/risk-specific experience.
        assert store.finalize_action_outcomes(now=300.0) == []
        assert store.relevant_outcome_count("RENICE_BACK", "critical") == 1
    finally:
        store.close()


def test_sparse_telemetry_is_not_reported_as_a_measured_outcome(tmp_path):
    store = Store(tmp_path / "outcomes.db")
    try:
        store.insert_action(
            id="action-gap",
            created_ts=100.0,
            proposed_by="operator",
            kind="renice_back",
            target_pid=123,
            target_start_time=1.0,
            before_nice=0,
            applied_nice=10,
            intent_id="intent-gap",
            status="applied",
            ts_applied=100.0,
        )
        assert store.record_action_outcome("intent-gap", "action-gap", "applied", 100.0, {})
        store.insert_samples([
            make_sample(ts=70.0),
            make_sample(ts=72.0),
            make_sample(ts=210.0),
            make_sample(ts=212.0),
        ])

        assert store.finalize_action_outcomes(now=235.0) == [
            {"intent_id": "intent-gap", "status": "insufficient_telemetry"}
        ]
        [outcome] = store.recent_action_outcomes()
        assert outcome["measured"] is False
        assert outcome["reason"] == "incomplete_pre_or_post_window"
        assert outcome["pressure_returned"] is None
        assert store.relevant_outcome_count("RENICE_BACK", "critical") == 0
    finally:
        store.close()
