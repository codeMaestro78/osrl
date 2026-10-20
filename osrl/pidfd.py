"""pidfd: stable process handles that prevent pid recycling while open.

Why this exists (brief D / DESIGN §6): between "revalidate (pid, start_time)"
and "apply the syscall" there is a window where the target could exit and its
pid could be recycled by a NEW process. os.pidfd_open(pid) pins the struct pid
in the kernel: while the pidfd is open, that pid cannot be reused by anyone,
even if the original process exits. The executor therefore:

1. pidfd_open(pid)  -> pin (raises if process already gone)
2. poll(0) the fd   -> POLLIN means the process EXITED (handle now useless)
3. revalidate start_time -> still the process we recorded
4. perform the mutation -> ESRCH surfaces as a typed failure otherwise
5. close()          -> release the pin

This is part of the action path, never of the risk/RL layers. Handles errors
as typed exceptions so the ledger can record exact causes.
"""

from __future__ import annotations

import os
import select
from collections.abc import Callable

from osrl.procid import revalidate

POLL_MASK = select.POLLIN | select.POLLERR | select.POLLHUP


class PidfdError(Exception):
    """pidfd unavailable on this kernel/platform."""


class TargetGone(PidfdError):
    """Process not found at open, or exited while pinned."""


class Pidfd:
    """Owns one pidfd for its lifetime; close() releases the kernel pin."""

    def __init__(self, pid: int) -> None:
        if getattr(os, "pidfd_open", None) is None:
            raise PidfdError("kernel/python lacks pidfd_open; refusing unsafe path")
        try:
            self.pid = pid
            self.fd: int = os.pidfd_open(pid)
        except ProcessLookupError as err:
            raise TargetGone(f"pid {pid} not found") from err
        except OSError as err:
            raise PidfdError(f"pidfd_open({pid}) failed: {err}") from err

    def exited(self) -> bool:
        """True when the pinned process has exited (POLLIN on pidfd)."""
        p = select.poll()
        p.register(self.fd, POLL_MASK)
        events = dict(p.poll(0))
        p.unregister(self.fd)
        return bool(events.get(self.fd, 0))

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def assert_same_task(
    pid: int,
    expected_start_time: float,
    starttime_fn: Callable[[int], float | None],
) -> None:
    """Revalidate while a pidfd holds the pid against recycling.

    Raises TargetGone for a dead/mismatched target; returns otherwise.
    This guards the tiny residual race after pinning: start_time must still
    match exactly (same uid process could have raced exits before open).
    """
    if not revalidate(pid, expected_start_time, starttime_fn=starttime_fn):
        raise TargetGone(f"pid {pid} is no longer the recorded task")
