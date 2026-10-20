"""Executor: real renice_back on our own children + typed failure modes.

Restore (nice-lowering) is EPERM unprivileged on this box (probed twice);
its capable=True path is exercised with an injected setter so the CODE is
verified without lying about the kernel.
"""

import subprocess
import time

import psutil
import pytest

from osrl.executor import (
    ElevationRequired,
    ExecutorError,
    apply_renice_back,
    apply_restore,
)
from osrl.pidfd import TargetGone


def spawn_child() -> subprocess.Popen:
    return subprocess.Popen(["sleep", "20"])


def test_renice_back_real_and_verified():
    p = spawn_child()
    try:
        info = psutil.Process(p.pid)
        before = info.nice()
        applied = apply_renice_back(p.pid, before + 5, info.create_time(), before)
        assert applied == before + 5
        assert psutil.Process(p.pid).nice() == before + 5
    finally:
        p.kill()
        p.wait()


def test_renice_back_rejects_priority_raising():
    p = spawn_child()
    try:
        info = psutil.Process(p.pid)
        with pytest.raises(ExecutorError):
            apply_renice_back(
                p.pid, info.nice() - 1, info.create_time(), info.nice()
            )
    finally:
        p.kill()
        p.wait()


def test_renice_back_rejects_out_of_range():
    p = spawn_child()
    try:
        info = psutil.Process(p.pid)
        with pytest.raises(ExecutorError):
            apply_renice_back(p.pid, 25, info.create_time(), info.nice())
    finally:
        p.kill()
        p.wait()


def test_renice_back_identity_mismatch_refuses_to_touch():
    p = spawn_child()
    try:
        info = psutil.Process(p.pid)
        calls = []
        with pytest.raises(TargetGone):
            apply_renice_back(
                p.pid, 10, info.create_time() + 5.0, info.nice(),
                nice_setter=lambda pid, n: calls.append((pid, n)),
            )
        assert calls == []  # identity failed BEFORE any mutation
    finally:
        p.kill()
        p.wait()


def test_renice_back_dead_pid_typed():
    with pytest.raises(TargetGone):
        apply_renice_back(999_999_999, 10, time.time(), 0)


def test_postcondition_mismatch_is_typed_failure():
    """Setter runs, read-back disagrees -> ExecutorError (never trust a
    syscall that may have silently failed)."""
    p = spawn_child()
    try:
        info = psutil.Process(p.pid)
        writes = []
        with pytest.raises(ExecutorError, match="postcondition"):
            apply_renice_back(
                p.pid, 10, info.create_time(), info.nice(),
                nice_reader=lambda _pid: 3,  # lies
                nice_setter=lambda pid, n: writes.append((pid, n)),
            )
        assert writes == [(p.pid, 10)]
    finally:
        p.kill()
        p.wait()


def test_restore_capability_gate_is_hard():
    p = spawn_child()
    try:
        info = psutil.Process(p.pid)
        calls = []
        with pytest.raises(ElevationRequired):
            apply_restore(
                p.pid, 0, info.create_time(), info.nice(), capable=False,
                nice_setter=lambda pid, n: calls.append((pid, n)),
            )
        assert calls == []  # never even attempts the known-denied syscall
    finally:
        p.kill()
        p.wait()


def test_restore_capable_path_with_fake_syscall():
    p = spawn_child()
    try:
        info = psutil.Process(p.pid)
        writes = []
        applied = apply_restore(
            p.pid, 2, info.create_time(), 12, capable=True,
            nice_reader=lambda _pid: 2,
            nice_setter=lambda pid, n: writes.append(n),
        )
        assert applied == 2
        assert writes == [2]
    finally:
        p.kill()
        p.wait()


def test_restore_rejects_priority_raising():
    p = spawn_child()
    try:
        info = psutil.Process(p.pid)
        with pytest.raises(ExecutorError):
            apply_restore(
                p.pid, 15, info.create_time(), 10, capable=True,
                nice_reader=lambda _pid: 10,
                nice_setter=lambda pid, n: None,
            )
    finally:
        p.kill()
        p.wait()
