"""PSI trigger registration and event-driven polling.

Kernel contract verified live on this machine (2026-10-20, kernel 7.0):

- Open /proc/pressure/<resource> with O_RDWR and write the spec
  "<some|full> <stall_us> <window_us>" to arm a monitor (one spec per fd;
  a second write to the same fd fails EBUSY -- matches the docs example).
- Unprivileged windows must be multiples of 2 s (kernel rejects others
  with EINVAL; docs constrain windows to 500 ms..10 s).
- Closing the fd deregisters the monitor; POLLPRI wakes the poller and
  POLLERR means the event source is gone.
- IMPORTANT (empirically observed): a trigger may fire immediately after
  registration when the resource just exited a stall window. A fired
  event is therefore *evidence of recent stall*, never an error.

Design choice: events are advisory "recent stall" witnesses; numeric
sample evidence in osrl.risk remains authoritative. Events are recorded
to the store by the daemon's callback (own thread-local connection).
"""

from __future__ import annotations

import os
import select
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_MASK = select.POLLPRI | select.POLLERR | select.POLLHUP
_DEAD_MASK = select.POLLERR | select.POLLHUP


@dataclass(frozen=True)
class TriggerSpec:
    """Arms '<kind> <threshold_us> <window_us>' on one resource file."""

    resource: str  # 'cpu' | 'memory' | 'io'
    kind: str  # 'some' | 'full'
    threshold_us: int
    window_us: int  # unprivileged: multiple of 2 s

    @property
    def text(self) -> str:
        return f"{self.kind} {self.threshold_us} {self.window_us}"

    @property
    def min_spacing_s(self) -> float:
        """Local duplicate-suppression floor. The kernel already rate-limits
        to one notification per tracking window; half-window here is a cheap
        belt-and-braces so our own bookkeeping cannot spam the ledger."""
        return max(0.25, (self.window_us / 1e6) * 0.5)

    def describe(self) -> dict:
        return {
            "resource": self.resource,
            "kind": self.kind,
            "threshold_us": self.threshold_us,
            "window_us": self.window_us,
        }


class PsiRegistrationError(Exception):
    def __init__(self, spec: TriggerSpec, path: Path, errno_: int) -> None:
        import errno as errno_mod

        code = errno_mod.errorcode.get(errno_, errno_)
        super().__init__(
            f"psi trigger {spec.text!r} on {path} failed: {code}"
        )
        self.errno_value = errno_


class PsiTrigger:
    """One armed monitor; holds its fd open for its whole lifetime."""

    def __init__(self, spec: TriggerSpec, path: str | Path) -> None:
        self.spec = spec
        self.path = Path(path)
        self.fd = -1
        try:
            self.fd = os.open(str(self.path), os.O_RDWR | os.O_NONBLOCK)
            os.write(self.fd, (spec.text + "\x00").encode())
        except OSError as err:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
            raise PsiRegistrationError(spec, self.path, err.errno) from err

    def fileno(self) -> int:
        return self.fd

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1  # kernel side is now deregistered


class PsiPoller:
    """select/poll loop over armed triggers -> callbacks; stop via event.

    Callbacks: on_event(trigger, ts) for POLLPRI fires (local spacing
    applied per trigger spec), on_dead(trigger, ts) for POLLERR/POLLHUP
    (source gone; the trigger stops being waited on afterwards).
    Overrides _wait() in tests to simulate kernel events without them.
    """

    def __init__(
        self,
        triggers: list[PsiTrigger],
        on_event: Callable[[PsiTrigger, float], None] | None = None,
        on_dead: Callable[[PsiTrigger, float], None] | None = None,
        probe_s: float = 0.5,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.triggers = triggers
        self._fdmap: dict[int, PsiTrigger] = {t.fd: t for t in triggers}
        self._alive = list(triggers)
        self._last_fired: dict[TriggerSpec, float] = {}
        self.on_event = on_event or (lambda t, ts: None)
        self.on_dead = on_dead or (lambda t, ts: None)
        self.probe_s = probe_s
        self._clock = clock

    # -- overridable wait --------------------------------------------------

    def _wait(self, fds: list[int], timeout_ms: int) -> dict[int, int]:
        if not fds:
            return {}
        p = select.poll()
        for fd in fds:
            p.register(fd, _MASK)
        return dict(p.poll(timeout_ms))

    # -- lifecycle ---------------------------------------------------------

    def run(self, stop_event=None, max_iters: int | None = None) -> None:
        it = 0
        while True:
            if stop_event is not None:
                if stop_event.is_set():
                    break
            if max_iters is not None and it >= max_iters:
                break
            it += 1
            if self._alive:
                fds = [t.fd for t in self._alive]
                try:
                    events = self._wait(fds, int(self.probe_s * 1000))
                except OSError:
                    events = {}
                self._handle(events)
            else:
                # no live arms: idle-wait on stop_event for responsiveness
                if stop_event is not None and stop_event.wait(self.probe_s):
                    break
                if stop_event is None:
                    time.sleep(self.probe_s)

    def _handle(self, events: dict[int, int]) -> None:
        now = self._clock()
        for fd, mask in events.items():
            trig = self._fdmap.get(fd)
            if trig is None or trig not in self._alive:
                continue
            if mask & _DEAD_MASK:
                self._alive.remove(trig)
                self.on_dead(trig, now)
                continue
            if mask & select.POLLPRI:
                last = self._last_fired.get(trig.spec)
                if last is None or (now - last) >= trig.spec.min_spacing_s:
                    self._last_fired[trig.spec] = now
                    self.on_event(trig, now)

    def close(self) -> None:
        """Closes all fds; kernel deregisters each monitor."""
        for trig in list(self.triggers):
            try:
                trig.close()
            except OSError:
                pass
        self._alive = []

    @property
    def alive_triggers(self) -> list[PsiTrigger]:
        return list(self._alive)
