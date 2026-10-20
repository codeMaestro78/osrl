"""Per-process VRAM sampling (risk #4 fix): optional provider, sorted top-5,
failure degrades to empty, store roundtrip includes it."""

import json

from osrl.collector import Collector
from osrl.store import Store
from tests.conftest import MachineFixture, make_sample


def test_gpu_procs_sorted_top5(tmp_path):
    machine = MachineFixture(tmp_path)
    rows = [
        {"pid": 1, "mem_bytes": 10},
        {"pid": 2, "mem_bytes": 300},
        {"pid": 3, "mem_bytes": 200},
        {"pid": 4, "mem_bytes": 400},
        {"pid": 5, "mem_bytes": 500},
        {"pid": 6, "mem_bytes": 250},
        {"pid": 7, "mem_bytes": 150},
    ]
    p = machine.providers()
    p.gpu_procs = lambda: rows
    c = Collector(p, heavy_cpu_pct=50.0, heavy_rss_mib=512.0)
    sample, _ = c.tick(1000.0)
    tops = sample.gpu_top_procs
    assert [r["mem_bytes"] for r in tops] == [500, 400, 300, 250, 200]
    assert [r["pid"] for r in tops] == [5, 4, 2, 6, 3]


def test_gpu_procs_none_provider_leaves_empty(tmp_path):
    machine = MachineFixture(tmp_path)
    p = machine.providers()
    assert p.gpu_procs is None
    c = Collector(p, heavy_cpu_pct=50.0, heavy_rss_mib=512.0)
    sample, _ = c.tick(1000.0)
    assert sample.gpu_top_procs == []


def test_gpu_procs_error_degrades_to_empty(tmp_path):
    machine = MachineFixture(tmp_path)
    p = machine.providers()
    p.gpu_procs = lambda: (_ for _ in ()).throw(RuntimeError("nvml boom"))
    c = Collector(p, heavy_cpu_pct=50.0, heavy_rss_mib=512.0)
    sample, _ = c.tick(1000.0)
    assert sample.gpu_top_procs == []


def test_gpu_top_procs_persist_roundtrip(tmp_path):
    store = Store(tmp_path / "t.db")
    try:
        s = make_sample(ts=1.0, gpu_top_procs=[{"pid": 5, "mem_bytes": 123}])
        store.insert_samples([s])
        row = store._conn.execute(
            "SELECT gpu_top_procs_json FROM samples WHERE ts=1.0"
        ).fetchone()
        assert json.loads(row["gpu_top_procs_json"]) == [
            {"pid": 5, "mem_bytes": 123}
        ]
    finally:
        store.close()
