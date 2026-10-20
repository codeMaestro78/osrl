"""Process identity and revalidation primitives.

On Linux a bare pid does NOT identify a process across samples: pids are
recycled. The stable identity is the pair (pid, start_time), tracked from the
psutil create_time. The same helpers later feed the guard's identity
revalidation (Phase 2) and pidfd-based signaling (research brief D).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessId:
    pid: int
    start_time: float  # epoch-like seconds (psutil create_time)

    @property
    def key(self) -> tuple[int, float]:
        return (self.pid, self.start_time)


StarttimeFn = Callable[[int], float | None]


def get_psutil_starttime(pid: int) -> float | None:
    """Real revalidation source: psutil create_time of the live pid.

    Returns None for dead pids and for pid-reuse canaries where the reader
    itself races the exit (raises no exceptions).
    """
    try:
        import psutil

        return psutil.Process(pid).create_time()
    except Exception:  # noqa: BLE001 - any psutil/OSError means "unknown"
        return None


def revalidate(
    pid: int,
    expected_start_time: float,
    starttime_fn: StarttimeFn = get_psutil_starttime,
    epsilon_s: float = 1e-3,
) -> bool:
    """True only if the live pid is the SAME process we recorded.

    Requirements: pid alive AND start_time (nearly) equal. This is the
    contract every live action will have to satisfy before firing
    (Phase 2 guard), kept strict and dependency-free of RL.
    """
    current = starttime_fn(pid)
    if current is None:
        return False
    return abs(current - expected_start_time) <= epsilon_s
