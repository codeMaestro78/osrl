"""Collector behavior: diff math, event logic, resilience to missing inputs."""

import pytest

import osrl.collector as col


def make_collector(machine, heavy_cpu_pct=25.0, heavy_rss_mib=200.0):
    return col.Collector(
        machine.providers(),
        heavy_cpu_pct=heavy_cpu_pct,
        heavy_rss_mib=heavy_rss_mib,
    )


def test_first_tick_rates_are_none_not_zero(machine):
    machine.write_psi("cpu", 48_508_075)
    machine.write_psi("memory", 1_182_582)
    machine.write_psi("io", 13_384_149)
    machine.write_vmstat({"pswpin": 157141, "pswpout": 1026156})
    machine.set_proc(1, 100.0, "idle-proc", cpu_time_s=0.0, rss_mib=10.0)

    c = make_collector(machine)
    sample, events = c.tick(ts=1000.0)

    assert sample.interval_s is None  # first tick: interval unknown
    assert sample.cpu_total_pct is None
    assert sample.swp_out_ps is None
    assert sample.swp_in_ps is None
    assert sample.psi_cpu.some_stall_us is None
    assert sample.psi_cpu.some_total_us == 48_508_075  # raw totals still recorded
    assert sample.top_procs and sample.top_procs[0]["cpu_pct"] is None
    assert events == []


def test_second_tick_derives_rates(machine):
    machine.write_psi("memory", 1_000)
    machine.write_vmstat({"pswpin": 50, "pswpout": 100})
    machine.set_proc(42, 7.0, "chrome", cpu_time_s=10.0, rss_mib=100.0)
    c = make_collector(machine)
    c.tick(ts=1000.0)

    machine.write_psi("memory", 3_000)                      # +2000 us stall
    machine.write_vmstat({"pswpin": 50, "pswpout": 130})    # +30 pages swapped out
    machine.cpu_total = col.CpuTimes(user=140.0, system=30.0, idle=110.0)
    machine.set_proc(42, 7.0, "chrome", cpu_time_s=10.3, rss_mib=100.0)  # +0.3 s
    sample2, events2 = c.tick(ts=1004.0)                    # dt = 4.0 s

    assert events2 == []
    assert sample2.interval_s == 4.0
    assert sample2.swp_out_ps == 30 / 4
    assert sample2.swp_in_ps == 0.0
    assert sample2.psi_mem.some_stall_us == 2000
    # CPU: busy delta (140-100)+(30-20)=50 over total delta 80 (idle 80->110)
    assert sample2.cpu_total_pct == (50 / 80) * 100
    # proc: 0.3 s / (4 s * 4 cpus) = 1.875%
    assert sample2.top_procs[0]["cpu_pct"] == pytest.approx(1.875)


def test_counter_reset_yields_none(machine):
    machine.write_psi("memory", 5_000)
    machine.write_vmstat({"pswpout": 1_000})
    c = make_collector(machine)
    c.tick(ts=0.0)
    machine.write_psi("memory", 4_000)     # decrease: reboot happened
    machine.write_vmstat({"pswpout": 0})
    s, _ = c.tick(ts=2.0)
    assert s.psi_mem.some_stall_us is None
    assert s.swp_out_ps is None


def test_unreadable_psi_resets_history_no_diff_across_gap(machine):
    machine.write_psi("cpu", 1_000)
    c = make_collector(machine)
    c.tick(ts=10.0)
    (machine.psi_dir / "cpu").unlink()     # outage
    s2, _ = c.tick(ts=12.0)
    assert s2.psi_cpu.some_total_us is None
    assert s2.psi_cpu.some_stall_us is None
    machine.write_psi("cpu", 4_000)        # restored, grew relative to tick 1
    s3, _ = c.tick(ts=14.0)
    # must NOT diff against the tick-1 total across the outage gap
    assert s3.psi_cpu.some_stall_us is None


def test_missing_psi_files_entirely(machine):
    c = make_collector(machine)            # no psi files written at all
    s, _ = c.tick(ts=0.0)
    assert s.psi_cpu.some_total_us is None
    assert s.psi_io.some_total_us is None


def test_gpu_provider_error_yields_none(machine):
    c = make_collector(machine)
    machine.gpu_error = True
    s, _ = c.tick(ts=0.0)
    assert s.gpu is None
    machine.gpu_error = False
    machine.gpu_sample = col.GpuSample(util_pct=59.0, mem_used_mib=800.0,
                                       mem_total_mib=6144.0, temp_c=61.0,
                                       power_w=35.0)
    s2, _ = c.tick(ts=2.0)
    assert s2.gpu.util_pct == 59.0


def test_heavyweight_birth_death_events(machine):
    machine.set_proc(1, 100.0, "light-proc", cpu_time_s=0.0, rss_mib=10.0)
    c = make_collector(machine, heavy_rss_mib=200.0, heavy_cpu_pct=25.0)
    _, events = c.tick(ts=0.0)             # seed history; process is light
    assert events == []

    machine.set_proc(1, 100.0, "heavy-proc", cpu_time_s=0.0, rss_mib=350.0)
    _, events = c.tick(ts=2.0)             # same identity crossed the threshold
    births = [e for e in events if e.kind == "birth"]
    assert [e.pid for e in births] == [1]
    assert births[0].detail["transition"] == "crossed"
    assert births[0].detail["rss_mib"] == pytest.approx(350.0)

    machine.procs = []
    _, events = c.tick(ts=4.0)             # process exits -> exactly one death
    deaths = [e for e in events if e.kind == "death"]
    assert [e.pid for e in deaths] == [1]


def test_pid_reuse_detected(machine):
    c = make_collector(machine)
    machine.set_proc(77, 100.0, "flake-worker", cpu_time_s=0.0, rss_mib=300.0)
    _, events1 = c.tick(ts=0.0)            # heavy on arrival -> spawn birth
    assert [e.kind for e in events1 if e.pid == 77] == ["birth"]

    machine.set_proc(77, 999.0, "flake-worker", cpu_time_s=0.0, rss_mib=300.0)
    _, events2 = c.tick(ts=2.0)
    kinds = [e.kind for e in events2 if e.pid == 77]
    assert "pid_reuse" in kinds  # old identity reported, never treated as same
    assert "birth" in kinds      # and the new identity is a fresh heavyweight


def test_top_procs_ranked_by_cpu_second_tick(machine):
    c = make_collector(machine)
    machine.set_proc(1, 1.0, "a", 0.0, 5.0)
    machine.set_proc(2, 1.0, "b", 0.0, 500.0)
    c.tick(ts=0.0)
    machine.set_proc(1, 1.0, "a", 1.0, 5.0)   # a burned 1.0 s CPU
    machine.set_proc(2, 1.0, "b", 0.1, 500.0)
    s, _ = c.tick(ts=2.0)
    assert s.top_procs[0]["pid"] == 1         # cpu wins over rss once history exists


def test_per_core_cpu_percentages(machine):
    machine.per_cpu = [col.CpuTimes(user=10.0, idle=10.0)] * 4
    c = make_collector(machine)
    s1, _ = c.tick(ts=0.0)
    assert s1.cpu_percpu_pct == []
    machine.per_cpu = [col.CpuTimes(user=14.0, idle=16.0)] * 4
    s2, _ = c.tick(ts=2.0)
    assert len(s2.cpu_percpu_pct) == 4
    assert all(v == (4.0 / 10.0) * 100 for v in s2.cpu_percpu_pct)
