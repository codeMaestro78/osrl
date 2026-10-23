"""Streamlit flow tests against the real schema and partial telemetry."""

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from osrl.collector import PsiInterval, Sample
from osrl.store import Store


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    db = tmp_path / "dashboard.db"
    store = Store(db)
    now = time.time()
    store.insert_samples(
        [
            Sample(
                ts=now - 400 + 2 * i,
                interval_s=2,
                cpu_total_pct=25,
                mem_available_bytes=8 * 1024**3,
                swap_used_pct=20,
                psi_mem=PsiInterval(some_stall_us=0),
                cpu_percpu_pct=[None, 20],
                top_procs=[
                    {
                        "pid": 55,
                        "name": "worker <sample>",
                        "start_time": 1,
                        "cpu_pct": None,
                        "rss_mib": 25,
                    }
                ],
            )
            for i in range(200)
        ]
    )
    store.heartbeat(now, 123, {"risk_level": "none"})
    store.insert_action(
        id="attempt",
        created_ts=now,
        proposed_by="operator",
        kind="renice_back",
        target_pid=55,
        target_start_time=1,
        status="rejected",
        error="identity drift",
    )
    store.record_action_outcome(
        "dashboard-attempt", None, "rejected", now,
        {"reason": "identity drift"},
    )
    store.close()
    monkeypatch.setenv("OSRL_DB", str(db))
    monkeypatch.setenv("OSRL_DEMO", "1")
    monkeypatch.setenv("OSRL_RL_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    from osrl.dashboard.shared import get_conn

    get_conn.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "osrl/dashboard/app.py"))
    yield app.run(timeout=30)
    get_conn.clear()


def test_all_dashboard_pages_render_partial_telemetry(dashboard):
    for page in dashboard.radio[0].options:
        dashboard.radio[0].set_value(page).run(timeout=30)
        assert not dashboard.exception, (page, dashboard.exception)
    dashboard.radio[0].set_value("⚙️ Processes").run(timeout=30)
    assert any("Reason" in frame.value.columns for frame in dashboard.dataframe)
    dashboard.radio[0].set_value("🤖 Agent").run(timeout=30)
    assert any("Reason" in frame.value.columns for frame in dashboard.dataframe)
    assert "Recommendation Evidence" in [item.value for item in dashboard.subheader]
    assert "Intervention Outcomes" in [item.value for item in dashboard.subheader]


def test_evaluation_generates_report_and_is_clear_about_evidence(dashboard):
    dashboard.radio[0].set_value("🧪 Evaluation").run(timeout=30)
    dashboard.button[0].click().run(timeout=30)
    assert not dashboard.exception
    assert len(dashboard.dataframe) == 2
    assert dashboard.dataframe[0].value["recall"].isna().all()
