"""Telemetry collector: one Sample per tick plus heavyweight process events.

Design constraints carried from DESIGN.md and the audit brief:

- Rates/pressure stalls come ONLY from diffs of cumulative counters. The
  first tick yields None (never 0) for every derived rate: "unknown" must
  stay distinguishable from "zero".
- Counter resets (reboot) yield None for that interval, not a fake value.
- A temporarily unreadable input resets its stored history so a diff NEVER
  spans an outage (stale telemetry still means None, not a cumulative jump).
- Every machine interaction is injectable through ``Providers`` so unit tests
  script synthetic machines; the real wiring lives in osrl.adapters.
- tick() returns (sample, events). Events are keyed by (pid, start_time); a
  same-pid process with a different start_time is reported as a pid-reuse
  death plus a fresh birth, never silently treated as the same process.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from osrl.psi import read_psi
from osrl.vmstat import VmSnapshot, metric_deltas, read_vmstat

# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MemStats:
    total_bytes: int
    available_bytes: int
    used_pct: float
    swap_total_bytes: int
    swap_used_bytes: int
    swap_pct: float


@dataclass(frozen=True)
class CpuTimes:
    """Attribute-compatible subset of psutil's cpu_times."""

    user: float = 0.0
    nice: float = 0.0
    system: float = 0.0
    idle: float = 0.0
    iowait: float = 0.0
    irq: float = 0.0
    softirq: float = 0.0
    steal: float = 0.0

    def busy_ratio(self) -> tuple[float, float]:
        """(busy_seconds, total_seconds) over the counter lifetime."""
        idle_total = self.idle + self.iowait
        total = (
            self.user + self.nice + self.system + self.idle
            + self.iowait + self.irq + self.softirq + self.steal
        )
        return (total - idle_total), total


@dataclass(frozen=True)
class ProcRow:
    pid: int
    start_time: float
    name: str
    cpu_time_s: float  # cumulative user+system seconds of that process
    rss_bytes: int

    @property
    def key(self) -> tuple[int, float]:
        return (self.pid, self.start_time)


@dataclass(frozen=True)
class GpuSample:
    util_pct: float | None = None
    mem_used_mib: float | None = None
    mem_total_mib: float | None = None
    temp_c: float | None = None
    power_w: float | None = None


@dataclass(frozen=True)
class PsiInterval:
    """Raw cumulative totals plus the interval stall diff (microseconds)."""

    some_total_us: int | None = None
    some_stall_us: int | None = None
    full_total_us: int | None = None
    full_stall_us: int | None = None


@dataclass
class ProcEvent:
    ts: float
    kind: str  # 'birth' | 'death' | 'pid_reuse'
    pid: int
    start_time: float
    name: str
    detail: dict = field(default_factory=dict)


@dataclass
class Sample:
    ts: float
    interval_s: float | None = None
    ncpu: int = 1
    cpu_total_pct: float | None = None
    cpu_percpu_pct: list = field(default_factory=list)
    load1: float | None = None
    load5: float | None = None
    load15: float | None = None
    mem_total_bytes: int | None = None
    mem_available_bytes: int | None = None
    mem_used_pct: float | None = None
    swap_total_bytes: int | None = None
    swap_used_bytes: int | None = None
    swap_used_pct: float | None = None
    swp_in_ps: float | None = None
    swp_out_ps: float | None = None
    majflt_ps: float | None = None
    reclaim_scan_ps: float | None = None
    reclaim_steal_ps: float | None = None
    allocstall_ps: float | None = None
    psi_cpu: PsiInterval = field(default_factory=PsiInterval)
    psi_mem: PsiInterval = field(default_factory=PsiInterval)
    psi_io: PsiInterval = field(default_factory=PsiInterval)
    gpu: GpuSample | None = None
    gpu_top_procs: list = field(default_factory=list)  # [{pid, mem_bytes}]
    top_procs: list = field(default_factory=list)
    collect_ms: float = 0.0


@dataclass
class Providers:
    """Everything the collector touches from the machine. Unit tests replace
    these callables with synthetics; osrl.adapters builds the real ones."""

    virtual_memory: Callable[[], MemStats]
    cpu_times: Callable[[], CpuTimes]
    per_cpu_times: Callable[[], list[CpuTimes]]
    load_average: Callable[[], tuple[float, float, float]]
    process_iter: Callable[[], list[ProcRow]]
    gpu: Callable[[], GpuSample | None]
    ncpu: int
    psi_paths: dict[str, Path] = field(default_factory=dict)
    vmstat_path: Path = Path("/proc/vmstat")
    gpu_procs: Callable[[], list[dict]] | None = None


# Internals: per resource -> (some_total_us, full_total_us) or (None, None)
_PsiPrev = dict[str, tuple[int | None, int | None]]

_TOP_K = 15


class Collector:
    def __init__(
        self,
        providers: Providers,
        heavy_cpu_pct: float = 25.0,
        heavy_rss_mib: float = 200.0,
        top_k_procs: int = 15,
    ) -> None:
        self._p = providers
        self._top_k = max(1, int(top_k_procs))
        self._heavy_cpu = heavy_cpu_pct
        self._heavy_rss = heavy_rss_mib * 1024 * 1024
        self._last_ts: float | None = None
        self._last_cpu: CpuTimes | None = None
        self._last_percpu: list[CpuTimes] | None = None
        self._last_psi: _PsiPrev = {}
        self._last_vm: VmSnapshot | None = None
        # key=(pid, start_time) -> (cpu_time_s, name, rss_bytes)
        self._last_procs: dict[tuple[int, float], tuple[float, str, int]] = {}
        # keys currently meeting a heavy threshold (drives birth/death events)
        self._last_heavy: set[tuple[int, float]] = set()

    # ------------------------------------------------------------------ tick

    def tick(self, ts: float | None = None) -> tuple[Sample, list[ProcEvent]]:
        t0 = time.perf_counter()
        now = ts if ts is not None else time.time()
        dt: float | None = None
        if self._last_ts is not None:
            candidate = now - self._last_ts
            dt = candidate if candidate > 0.0 else None  # clock glitch -> unknown

        sample = Sample(ts=now, interval_s=dt, ncpu=self._p.ncpu)

        self._collect_cpu(sample, dt)
        self._collect_mem(sample)
        self._collect_psi(sample)
        self._collect_vmstat(sample, dt)
        events = self._collect_procs(sample, dt)
        self._collect_gpu(sample)

        sample.collect_ms = (time.perf_counter() - t0) * 1000.0
        self._last_ts = now
        return sample, events

    # -------------------------------------------------------------- internals

    def _collect_cpu(self, s: Sample, dt: float | None) -> None:
        try:
            prev = self._last_cpu
            cur = self._p.cpu_times()
            if prev is not None and dt:
                pb, pt = prev.busy_ratio()
                cb, ct = cur.busy_ratio()
                if ct > pt and 0 <= cb - pb <= ct - pt:
                    s.cpu_total_pct = (cb - pb) / (ct - pt) * 100.0
            self._last_cpu = cur

            prev_list = self._last_percpu
            cur_list = self._p.per_cpu_times()
            if prev_list is not None and dt and len(prev_list) == len(cur_list):
                per = []
                for pcpu, ccpu in zip(prev_list, cur_list, strict=False):
                    pb, pt = pcpu.busy_ratio()
                    cb, ct = ccpu.busy_ratio()
                    per.append((cb - pb) / (ct - pt) * 100.0
                               if ct > pt and 0 <= cb - pb <= ct - pt else None)
                s.cpu_percpu_pct = per
            self._last_percpu = cur_list

            l1, l5, l15 = self._p.load_average()
            s.load1, s.load5, s.load15 = l1, l5, l15
        except OSError:
            s.cpu_total_pct = None
            s.cpu_percpu_pct = []
            self._last_cpu = None
            self._last_percpu = None

    def _collect_mem(self, s: Sample) -> None:
        m = self._p.virtual_memory()
        s.mem_total_bytes = m.total_bytes
        s.mem_available_bytes = m.available_bytes
        s.mem_used_pct = m.used_pct
        s.swap_total_bytes = m.swap_total_bytes
        s.swap_used_bytes = m.swap_used_bytes
        s.swap_used_pct = m.swap_pct

    def _collect_psi(self, s: Sample) -> None:
        """Diff cumulative PSI totals; an unreadable file resets its history.

        System-level CPU full is captured but documented as undefined/zero
        at system level (kernel docs); downstream consumers must not use it.
        """
        prev = self._last_psi
        block: dict[str, PsiInterval] = {}
        for resource in ("cpu", "memory", "io"):
            stats = read_psi(self._p.psi_paths[resource])
            if stats is None:
                block[resource] = PsiInterval()  # resets prev below: no gaps
                continue
            some, full = stats["some"], stats.get("full")
            prev_some, prev_full = prev.get(resource, (None, None))
            block[resource] = PsiInterval(
                some_total_us=some.total_us,
                some_stall_us=_stall_diff(prev_some, some.total_us),
                full_total_us=full.total_us if full else None,
                full_stall_us=(
                    _stall_diff(prev_full, full.total_us) if full else None
                ),
            )
        s.psi_cpu = block["cpu"]
        s.psi_mem = block["memory"]
        s.psi_io = block["io"]
        self._last_psi = {r: (v.some_total_us, v.full_total_us) for r, v in block.items()}

    def _collect_vmstat(self, s: Sample, dt: float | None) -> None:
        cur = read_vmstat(self._p.vmstat_path)
        if cur is None:
            self._last_vm = None  # reset: never diff across an outage
            return
        d = metric_deltas(self._last_vm, cur)
        if dt:
            s.swp_in_ps = None if d["swp_in"] is None else d["swp_in"] / dt
            s.swp_out_ps = None if d["swp_out"] is None else d["swp_out"] / dt
            s.majflt_ps = None if d["majflt"] is None else d["majflt"] / dt
            s.reclaim_scan_ps = (
                None if d["reclaim_scan"] is None else d["reclaim_scan"] / dt
            )
            s.reclaim_steal_ps = (
                None if d["reclaim_steal"] is None else d["reclaim_steal"] / dt
            )
            s.allocstall_ps = (
                None if d["allocstall"] is None else d["allocstall"] / dt
            )
        self._last_vm = cur

    def _proc_cpu_pct(
        self, row: ProcRow, prior: tuple[float, str, int] | None, dt: float | None
    ) -> float | None:
        """Process CPU% over the tick; None until it has history."""
        if prior is None or not dt or row.cpu_time_s < prior[0]:
            return None
        return (row.cpu_time_s - prior[0]) / (dt * self._p.ncpu) * 100.0

    def _collect_procs(
        self, s: Sample, dt: float | None
    ) -> list[ProcEvent]:
        rows = self._p.process_iter()
        events: list[ProcEvent] = []
        prev = self._last_procs
        prev_heavy = self._last_heavy
        cur_rows: dict[tuple[int, float], ProcRow] = {r.key: r for r in rows}
        heavy_now: set[tuple[int, float]] = set()

        # 1) vanishings among previously heavy identities (death / pid reuse)
        for (pid, start), (_cpu_s, name, _rss) in prev.items():
            if (pid, start) in cur_rows or (pid, start) not in prev_heavy:
                continue
            pid_still_alive = any(r.pid == pid for r in rows)
            kind = "pid_reuse" if pid_still_alive else "death"
            detail = (
                {"note": "same pid reappeared with a different start_time"}
                if pid_still_alive
                else {}
            )
            events.append(
                ProcEvent(ts=s.ts, kind=kind, pid=pid, start_time=start,
                          name=name, detail=detail)
            )

        # 2) heaviness classification: births for new heavy identities and
        #    for tracked processes that just crossed a heavy threshold
        enriched: list[tuple[tuple, ProcRow]] = []
        for row in rows:
            prior = prev.get(row.key)
            cpu_pct = self._proc_cpu_pct(row, prior, dt)
            rss_heavy = row.rss_bytes >= self._heavy_rss
            cpu_heavy = cpu_pct is not None and cpu_pct >= self._heavy_cpu
            heavy = rss_heavy or cpu_heavy

            # First tick has no CPU history -> rank by RSS instead of inventing %
            weight = ((cpu_pct is not None, cpu_pct or 0.0, row.rss_bytes)
                      if dt else (False, 0.0, row.rss_bytes))
            enriched.append((weight, row))

            if not heavy:
                continue
            key = row.key
            heavy_now.add(key)
            if key not in prev_heavy:
                events.append(
                    ProcEvent(
                        ts=s.ts,
                        kind="birth",
                        pid=row.pid,
                        start_time=row.start_time,
                        name=row.name,
                        detail={
                            "cpu_pct": cpu_pct,
                            "rss_mib": row.rss_bytes / (1024 * 1024),
                            "transition": "spawn" if key not in prev else "crossed",
                        },
                    )
                )

        # 3) top-K ranking (CPU after history exists, RSS on the first tick)
        enriched.sort(key=lambda item: item[0], reverse=True)
        top: list[dict] = []
        for _, row in enriched[:self._top_k]:
            top.append(
                {"pid": row.pid, "start_time": row.start_time,
                 "name": row.name,
                 "cpu_pct": self._proc_cpu_pct(row, prev.get(row.key), dt),
                 "rss_mib": row.rss_bytes / (1024 * 1024)}
            )
        s.top_procs = top

        self._last_procs = {
            key: (row.cpu_time_s, row.name, row.rss_bytes)
            for key, row in cur_rows.items()
        }
        self._last_heavy = heavy_now
        return events

    def _collect_gpu(self, s: Sample) -> None:
        try:
            s.gpu = self._p.gpu()
        except Exception:  # noqa: BLE001 - GPU telemetry must never kill a tick
            s.gpu = None
        # Per-process VRAM (desired because watchdog decisions may need to
        # name the VRAM-heavy process later). Optional provider; the driver
        # exposes graphics+compute clients with per-pid bytes (verified
        # 2026-10-20 on this RTX 3050: 5 desktop clients readable).
        gp = getattr(self._p, "gpu_procs", None)
        if gp is not None:
            try:
                rows = gp() or []
                s.gpu_top_procs = sorted(
                    rows, key=lambda r: -(r.get("mem_bytes") or 0)
                )[:5]
            except Exception:  # noqa: BLE001
                s.gpu_top_procs = []


def _stall_diff(prev_us: int | None, cur_us: int | None) -> int | None:
    """Interval stall microseconds from cumulative totals; None = unknown.

    None on: missing endpoint (first tick / unreadable file) and on counter
    reset (cur < prev, e.g. reboot). Never fabricates a zero stall.
    """
    if prev_us is None or cur_us is None:
        return None
    if cur_us < prev_us:
        return None
    return cur_us - prev_us
