"""PSI trigger + poller: registration contracts, event handling, shutdown."""

import errno
import threading
from select import POLLERR, POLLPRI  # real Linux event values

import pytest

from osrl.psi_trigger import (
    PsiPoller,
    PsiRegistrationError,
    PsiTrigger,
    TriggerSpec,
)


class FakeClock:
    def __init__(self, step: float = 0.2) -> None:
        self.t = 1000.0
        self.step = step

    def __call__(self) -> float:
        return self.t

    def advance(self) -> None:
        self.t += self.step


class ScriptedPoller(PsiPoller):
    """Overrides kernel waiting with a scripted event sequence; advances its
    fake clock once per wait so spacing math behaves like real time."""

    def __init__(self, script, **kwargs):
        super().__init__(**kwargs)
        self._script = list(script)

    def _wait(self, fds, timeout_ms):
        if hasattr(self._clock, "advance"):
            self._clock.advance()
        if not self._script:
            return {}
        fd, mask = self._script.pop(0)
        return {fd: mask}


def make_triggers(tmp_path, n=1, window_us=1_000_000):
    specs = [TriggerSpec("memory", "some", 150_000 + i, window_us)
             for i in range(n)]
    trig_path = tmp_path / "memory"
    trig_path.write_text("placeholder\n")  # content is irrelevant to fakes
    return [PsiTrigger(s, trig_path) for s in specs]


def test_scripted_fires_respect_local_spacing(tmp_path):
    clk = FakeClock(step=0.3)
    triggers = make_triggers(tmp_path)
    events, dead = [], []

    p = ScriptedPoller(
        script=[(triggers[0].fd, POLLPRI),   # fire -> recorded
                (triggers[0].fd, POLLPRI),   # within half-window -> suppressed
                (triggers[0].fd, POLLPRI)],  # after spacing -> recorded
        triggers=triggers,
        on_event=lambda t, ts: events.append((t, ts)),
        on_dead=lambda t, ts: dead.append(t),
        clock=clk,
    )
    p.run(max_iters=3)
    assert len(events) == 2  # middle fire suppressed by min_spacing(0.5 s)
    assert events[0][1] == pytest.approx(1000.3)
    assert events[1][1] == pytest.approx(1000.9)
    assert not dead
    p.close()
    assert triggers[0].fd == -1


def test_dead_source_removes_trigger(tmp_path):
    clk = FakeClock()
    triggers = make_triggers(tmp_path)
    events, dead = [], []

    p = ScriptedPoller(
        script=[(triggers[0].fd, POLLERR),            # source gone
                (triggers[0].fd, POLLPRI)],           # must be ignored now
        triggers=triggers,
        on_event=lambda t, ts: events.append((t, ts)),
        on_dead=lambda t, ts: dead.append((t, ts)),
        clock=clk,
    )
    p.run(max_iters=2)
    assert len(dead) == 1
    assert events == []
    assert p.alive_triggers == []
    p.close()


def test_unknown_fd_events_ignored(tmp_path):
    triggers = make_triggers(tmp_path)
    events, dead = [], []
    p = ScriptedPoller(
        script=[(9999, POLLPRI)],  # fd we never armed
        triggers=triggers,
        on_event=lambda t, ts: events.append(t),
        on_dead=lambda t, ts: dead.append(t),
        clock=FakeClock(),
    )
    p.run(max_iters=1)
    assert events == [] and dead == []
    p.close()


def test_stop_event_terminates_run(tmp_path):
    clk = FakeClock()
    triggers = make_triggers(tmp_path)
    stop = threading.Event()
    p = ScriptedPoller(script=[], triggers=triggers, clock=clk)
    t = threading.Thread(target=p.run, kwargs={"stop_event": stop}, daemon=True)
    t.start()
    stop.set()
    t.join(timeout=3.0)
    assert not t.is_alive()
    p.close()


def test_immediate_fire_is_recorded_not_an_error(tmp_path):
    """Live probe behavior: an arm may fire at once (stall just ended).
    The poller must record it as evidence, regardless of time since arm."""
    clk = FakeClock()
    triggers = make_triggers(tmp_path)
    events = []
    p = ScriptedPoller(
        script=[(triggers[0].fd, POLLPRI)],
        triggers=triggers,
        on_event=lambda t, ts: events.append((t, ts)),
        clock=clk,
    )
    p.run(max_iters=1)
    assert len(events) == 1
    assert events[0][1] == pytest.approx(1000.2)  # default step advances once
    p.close()


# ---------------------------- real-kernel contracts (skip if no PSI files)

@pytest.mark.skipif(not __import__("pathlib").Path("/proc/pressure/memory").exists(),
                    reason="requires real /proc/pressure")
def test_real_registration_and_deregistration_cycle():
    spec = TriggerSpec("memory", "some", 150_000, 2_000_000)
    t1 = PsiTrigger(spec, "/proc/pressure/memory")
    t1.close()  # kernel deregisters on close
    t2 = PsiTrigger(spec, "/proc/pressure/memory")  # must not linger EBUSY
    t2.close()


@pytest.mark.skipif(not __import__("pathlib").Path("/proc/pressure/memory").exists(),
                    reason="requires real /proc/pressure")
def test_real_double_arm_returns_ebusy():
    spec = TriggerSpec("memory", "some", 150_000, 2_000_000)
    t = PsiTrigger(spec, "/proc/pressure/memory")
    try:
        with pytest.raises(OSError) as ei:
            import os as _os

            _os.write(t.fd, (spec.text + "\x00").encode())
        assert ei.value.errno == errno.EBUSY
    finally:
        t.close()


@pytest.mark.skipif(not __import__("pathlib").Path("/proc/pressure/memory").exists(),
                    reason="requires real /proc/pressure")
def test_real_unprivileged_window_constraint():
    """Docs: window must be a multiple of 2 s for unprivileged monitors.
    Live probe (2026-10-20): 1.5 s -> EINVAL."""
    spec = TriggerSpec("memory", "some", 150_000, 1_500_000)
    with pytest.raises(PsiRegistrationError) as ei:
        PsiTrigger(spec, "/proc/pressure/memory")
    assert ei.value.errno_value == errno.EINVAL


@pytest.mark.skipif(not __import__("pathlib").Path("/proc/pressure/memory").exists(),
                    reason="requires real /proc/pressure")
def test_real_poller_idle_roundtrip():
    """Registration + poll loop + clean shutdown + NO resource leak on a
    live system. Events may fire legitimately (swap-heavy box); format is
    checked instead of emptiness."""
    fired, dead = [], []
    spec = TriggerSpec("memory", "some", 150_000, 2_000_000)
    trig = PsiTrigger(spec, "/proc/pressure/memory")
    p = PsiPoller(
        [trig],
        on_event=lambda t, ts: fired.append(t.spec.describe()),
        on_dead=lambda t, ts: dead.append(t),
        probe_s=0.2,
    )
    stop = threading.Event()
    th = threading.Thread(target=p.run, kwargs={"stop_event": stop}, daemon=True)
    th.start()
    stop.set()
    th.join(timeout=3.0)
    p.close()
    assert not th.is_alive()
    for e in fired:
        assert e["resource"] == "memory" and e["window_us"] == 2_000_000
    assert dead == []
