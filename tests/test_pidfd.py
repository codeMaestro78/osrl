"""pidfd primitives: pinning, exit detection, identity re-hold."""

import subprocess
import time

import pytest

from osrl.pidfd import Pidfd, PidfdError, TargetGone, assert_same_task


def wait_exit_poll(pidfd: Pidfd, timeout: float = 3.0) -> bool:
    """poll(0) is instantaneous; give a just-killed child a moment."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pidfd.exited():
            return True
        time.sleep(0.05)
    return False


def test_open_live_chid_not_exited(tmp_path):
    p = subprocess.Popen(["sleep", "5"])
    try:
        fd = Pidfd(p.pid)
        assert fd.exited() is False
        fd.close()
    finally:
        p.kill()
        p.wait()


def test_open_missing_pid_raises_typed():
    with pytest.raises(TargetGone):
        Pidfd(999_999_999)


def test_exited_reports_after_death():
    """Open the pidfd BEFORE the child exits, then let it die: poll must
    report POLLIN. (Opening after reap correctly raises TargetGone instead.)
    """
    p = subprocess.Popen(["sleep", "0.3"])
    fd = Pidfd(p.pid)
    try:
        p.kill()
        p.wait()
        assert wait_exit_poll(fd)
    finally:
        fd.close()


def test_reuse_window_contract_documents_itself(tmp_path):
    """While the pidfd is open the kernel keeps the pid pinned; after our
    child exits we must see exited()==True, i.e. revalidation would refuse
    rather than mutate a recycled pid."""
    p = subprocess.Popen(["sleep", "30"])
    fd = Pidfd(p.pid)
    try:
        p.kill()
        p.wait()
        assert wait_exit_poll(fd)
    finally:
        fd.close()


def test_assert_same_task_mismatch_raises():
    with pytest.raises(TargetGone):
        assert_same_task(1234, 111.0, starttime_fn=lambda _pid: 999.0)


def test_close_releases_fd():
    p = subprocess.Popen(["sleep", "5"])
    try:
        fd = Pidfd(p.pid)
        fd.close()
        assert fd.fd == -1
    finally:
        p.kill()
        p.wait()


def test_pidfd_unavailable(monkeypatch):
    """Missing kernel hook must fail typed, never fall back to raw pids."""
    import os as _os

    from osrl import pidfd as pidfd_mod

    monkeypatch.setattr(_os, "pidfd_open", None, raising=False)
    with pytest.raises(PidfdError):
        pidfd_mod.Pidfd(1)
