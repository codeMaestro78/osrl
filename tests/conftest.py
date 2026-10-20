"""Shared fixtures: a scriptable synthetic machine standing in for the real one.

Unit tests inject `MachineFixture.providers()` into the collector/daemon, so
they exercise OUR logic (diff math, event logic, store semantics) without
depending on psutil/NVML/real /proc. A few smoke tests intentionally hit the
real machine and skip when /proc/pressure is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from osrl import collector as col  # noqa: E402


def make_sample(ts: float, **over) -> col.Sample:
    """Sample factory with sane non-toxic defaults for store/daemon tests."""
    fields = {
        "interval_s": 2.0,
        "ncpu": 4,
        "cpu_total_pct": None,
        "mem_total_bytes": 16 * 1024**3,
        "mem_available_bytes": 8 * 1024**3,
        "mem_used_pct": 50.0,
        "swap_total_bytes": 4 * 1024**3,
        "swap_used_bytes": 1 * 1024**3,
        "swap_used_pct": 25.0,
        "swp_out_ps": None,
        "psi_mem": col.PsiInterval(),
        "psi_io": col.PsiInterval(),
        "psi_cpu": col.PsiInterval(),
        "top_procs": [],
    }
    fields.update(over)
    return col.Sample(ts=ts, **fields)


class MachineFixture:
    """Mutable synthetic machine; providers() builds collector.Providers that
    read this state at call time."""

    def __init__(self, tmp_path: Path, ncpu: int = 4) -> None:
        self.tmp_path = tmp_path
        self.ncpu = ncpu
        self.mem = col.MemStats(
            total_bytes=16 * 1024**3,
            available_bytes=8 * 1024**3,
            used_pct=50.0,
            swap_total_bytes=4 * 1024**3,
            swap_used_bytes=1 * 1024**3,
            swap_pct=25.0,
        )
        self.cpu_total = col.CpuTimes(user=100.0, system=20.0, idle=80.0)
        self.per_cpu = [col.CpuTimes(user=25.0, system=5.0, idle=60.0)] * ncpu
        self.load = (1.0, 0.8, 0.6)
        self.procs: list[col.ProcRow] = []
        self.gpu_sample: col.GpuSample | None = None
        self.gpu_error = False
        self.psi_dir = tmp_path / "pressure"
        self.psi_dir.mkdir()
        self.vmstat_file = tmp_path / "vmstat"

    # ---- writers -------------------------------------------------------

    def set_proc(
        self, pid: int, start_time: float, name: str, cpu_time_s: float, rss_mib: float
    ) -> None:
        self.procs = [r for r in self.procs if r.pid != pid]
        self.procs.append(
            col.ProcRow(
                pid=pid,
                start_time=start_time,
                name=name,
                cpu_time_s=cpu_time_s,
                rss_bytes=int(rss_mib * 1024 * 1024),
            )
        )

    def write_psi(
        self,
        resource: str,
        some_total_us: int,
        full_total_us: int = 0,
        avg10: float = 0.0,
    ) -> None:
        text = (
            f"some avg10={avg10:.2f} avg60=0.00 avg300=0.00 "
            f"total={some_total_us}\n"
            f"full avg10=0.00 avg60=0.00 avg300=0.00 total={full_total_us}\n"
        )
        (self.psi_dir / resource).write_text(text)

    def write_vmstat(self, counters: dict[str, int]) -> None:
        lines = [f"{k} {v}" for k, v in counters.items()]
        self.vmstat_file.write_text("\n".join(lines) + "\n")

    # ---- providers -----------------------------------------------------

    def _gpu(self) -> col.GpuSample | None:
        if self.gpu_error:
            raise RuntimeError("simulated NVML failure")
        return self.gpu_sample

    def providers(self) -> col.Providers:
        return col.Providers(
            virtual_memory=lambda: self.mem,
            cpu_times=lambda: self.cpu_total,
            per_cpu_times=lambda: list(self.per_cpu),
            load_average=lambda: self.load,
            process_iter=lambda: list(self.procs),
            gpu=self._gpu,
            ncpu=self.ncpu,
            psi_paths={
                "cpu": self.psi_dir / "cpu",
                "memory": self.psi_dir / "memory",
                "io": self.psi_dir / "io",
            },
            vmstat_path=self.vmstat_file,
        )


@pytest.fixture
def machine(tmp_path) -> MachineFixture:
    return MachineFixture(tmp_path)
