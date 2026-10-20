"""Wiring of real machine providers (psutil, NVML) into collector.Providers.

This module is only imported by the daemon -- never by unit tests, which
inject their own synthetic Providers. Every provider here is defensive:
machine calls degrade to None / empty instead of raising, because missing
telemetry must be representable ("unknown") rather than fatal.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from osrl.collector import CpuTimes, GpuSample, MemStats, ProcRow, Providers


def _mem() -> MemStats:
    import psutil

    v = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return MemStats(
        total_bytes=v.total,
        available_bytes=v.available,
        used_pct=v.percent,
        swap_total_bytes=sw.total,
        swap_used_bytes=sw.used,
        swap_pct=sw.percent,
    )


def _cpu_times() -> CpuTimes:
    import psutil

    t = psutil.cpu_times()
    return CpuTimes(
        user=getattr(t, "user", 0.0),
        nice=getattr(t, "nice", 0.0),
        system=getattr(t, "system", 0.0),
        idle=getattr(t, "idle", 0.0),
        iowait=getattr(t, "iowait", 0.0),
        irq=getattr(t, "irq", 0.0),
        softirq=getattr(t, "softirq", 0.0),
        steal=getattr(t, "steal", 0.0),
    )


def _per_cpu_times() -> list[CpuTimes]:
    import psutil

    out = []
    for t in psutil.cpu_times(percpu=True):
        out.append(
            CpuTimes(
                user=getattr(t, "user", 0.0),
                nice=getattr(t, "nice", 0.0),
                system=getattr(t, "system", 0.0),
                idle=getattr(t, "idle", 0.0),
                iowait=getattr(t, "iowait", 0.0),
                irq=getattr(t, "irq", 0.0),
                softirq=getattr(t, "softirq", 0.0),
                steal=getattr(t, "steal", 0.0),
            )
        )
    return out


def _load() -> tuple[float, float, float]:
    import psutil

    return psutil.getloadavg()


def _process_iter() -> list[ProcRow]:
    import psutil

    rows: list[ProcRow] = []
    for p in psutil.process_iter(
        attrs=["pid", "create_time", "name", "memory_info", "cpu_times"],
        ad_value=None,
    ):
        info = getattr(p, "info", None)
        if not info:
            continue
        create_time = info.get("create_time")
        mem = info.get("memory_info")
        cput = info.get("cpu_times")
        rows.append(
            ProcRow(
                pid=info.get("pid") or p.pid,
                start_time=float(create_time) if create_time else 0.0,
                name=info.get("name") or "",
                cpu_time_s=(cput.user + cput.system) if cput else 0.0,
                rss_bytes=mem.rss if mem else 0,
            )
        )
    return rows


class _Nvml:
    """Lazy NVML singleton; device metrics return None when unsupported."""

    def __init__(self) -> None:
        self._handle = None
        self._failed = False

    def _init(self) -> bool:
        if self._handle is not None:
            return True
        if self._failed:
            return False
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._h = pynvml  # noqa: module ref for later calls
            return True
        except Exception:  # noqa: BLE001 - no GPU / no driver / no module
            self._failed = True
            return False

    def sample(self) -> GpuSample | None:
        if not self._init():
            return None
        h, nvml = self._handle, self._h
        try:
            util = nvml.nvmlDeviceGetUtilizationRates(h).gpu
            info = nvml.nvmlDeviceGetMemoryInfo(h)
            temp = nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
            power = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            return GpuSample(
                util_pct=float(util),
                mem_used_mib=info.used / (1024 * 1024),
                mem_total_mib=info.total / (1024 * 1024),
                temp_c=float(temp),
                power_w=float(power),
            )
        except Exception:  # noqa: BLE001 - device vanished / NotSupported
            return None

    def gpu_processes(self) -> list[dict]:
        """Per-client VRAM (verified readable on this driver for desktop
        graphics clients 2026-10-20). Graphics + compute lists are merged;
        duplicate pids keep the max. Errors degrade to an empty list."""
        if not self._init():
            return []
        h, nvml = self._handle, self._h
        per_pid: dict[int, int] = {}
        getters = [
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
            getattr(nvml, "nvmlDeviceGetComputeRunningProcesses_v2", None),
            getattr(nvml, "nvmlDeviceGetGraphicsRunningProcesses", None),
        ]
        for getter in getters:
            if getter is None:
                continue
            try:
                for proc in getter(h):
                    pid = int(getattr(proc, "pid", 0) or 0)
                    if pid <= 0:
                        continue
                    mem = getattr(proc, "usedGpuMemory", None)
                    mem = 0 if mem is None else int(mem)
                    per_pid[pid] = max(per_pid.get(pid, 0), mem)
            except Exception:  # noqa: BLE001 - one list failing is survivable
                continue
        return [{"pid": pid, "mem_bytes": mem} for pid, mem in per_pid.items()]


def default_providers(psi_dir: str | Path = "/proc/pressure") -> Providers:
    psi_dir = Path(psi_dir)
    nvml = _Nvml()
    return Providers(
        virtual_memory=_mem,
        cpu_times=_cpu_times,
        per_cpu_times=_per_cpu_times,
        load_average=_load,
        process_iter=_process_iter,
        gpu=nvml.sample,
        gpu_procs=nvml.gpu_processes,
        ncpu=multiprocessing.cpu_count(),
        psi_paths={
            "cpu": psi_dir / "cpu",
            "memory": psi_dir / "memory",
            "io": psi_dir / "io",
        },
        vmstat_path=Path("/proc/vmstat"),
    )
