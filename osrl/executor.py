"""Real process mutations: the ONLY module permitted to touch setpriority.

Empirical contract probed on this machine (2026-10-20, kernel 7.0,
RLIMIT_NICE = (0, 0)):

- Raising a same-uid target's nice (de-prioritize, ``renice_back``): always
  succeeds unprivileged (verified +10 on a child).
- LOWERING nice again (``restore``): EPERM for cross-process AND self-lower
  (verified twice a second after). Restore therefore REQUIRES elevation and
  is capability-gated by config (default False on this box); the guard
  rejects restore attempts with 'restore_requires_elevation', and the
  future RL action mask must mask restore off when the capability is False.
- No SIGKILL/SIGSTOP/termination primitive exists here at all.

Application safety contract (belt + suspenders):
1. pidfd_open pins the target pid against recycling for the whole window.
2. exit-poll + (pid, start_time) revalidation INSIDE the hold.
3. mutation (psutil setpriority wrapper: same syscall os.setpriority uses).
4. postcondition read-back: applied nice must equal requested; otherwise a
   typed ExecutorError marks the attempt failed (never assume success).

Errors are typed and always surfaced; nothing is swallowed silently.
"""

from __future__ import annotations

from collections.abc import Callable

from osrl.pidfd import Pidfd, PidfdError, TargetGone, assert_same_task

REVERSIBLE_KINDS = ("renice_back", "restore")


class ExecutorError(Exception):
    """Typed failure of a real mutation attempt."""


class ElevationRequired(ExecutorError):
    """The requested direction needs privileges we refuse to use."""


def _default_nice_reader(pid: int) -> int | None:
    try:
        import psutil

        return int(psutil.Process(pid).nice())
    except Exception:  # noqa: BLE001 - unreadable = unknown, never assumed
        return None


def _default_nice_setter(pid: int, nice: int) -> None:
    try:
        import psutil

        psutil.Process(pid).nice(nice)
    except psutil.AccessDenied as err:
        raise ElevationRequired(
            f"setpriority({pid}, {nice}) denied: {err}"
        ) from err
    except psutil.NoSuchProcess as err:
        raise TargetGone(f"pid {pid} vanished mid-apply") from err


def apply_renice_back(
    pid: int,
    target_nice: int,
    expected_start_time: float,
    current_nice: int,
    starttime_fn: Callable[[int], float | None] | None = None,
    nice_reader: Callable[[int], int | None] = _default_nice_reader,
    nice_setter: Callable[[int, int], None] = _default_nice_setter,
) -> int:
    """Raise the target's nice value (lower priority) and verify it stuck.

    precondition: target_nice >= current_nice >= 0 and target_nice <= 19.
    Returns the verified applied nice. Raises TargetGone / ExecutorError /
    ElevationRequired otherwise; the caller records exact causes.
    """
    if target_nice < current_nice:
        raise ExecutorError("renice_back must not raise priority")
    if not 0 <= target_nice <= 19:
        raise ExecutorError(f"target nice {target_nice} out of [0, 19]")
    try:
        holder = Pidfd(pid)
    except TargetGone:
        raise
    except PidfdError as err:
        raise ExecutorError(f"pin failed: {err}") from err
    try:
        if holder.exited():
            raise TargetGone(f"pid {pid} exited while pinned")
        assert_same_task(
            pid, expected_start_time, starttime_fn or _default_starttime
        )
        nice_setter(pid, target_nice)
        got = nice_reader(pid)
        if got is None:
            raise ExecutorError("postcondition unreadable after apply")
        if got != target_nice:
            raise ExecutorError(
                f"postcondition failed: nice={got}, wanted {target_nice}"
            )
        return got
    finally:
        holder.close()


def _default_starttime(pid: int) -> float | None:
    from osrl.procid import get_psutil_starttime

    return get_psutil_starttime(pid)


def apply_restore(
    pid: int,
    target_nice: int,
    expected_start_time: float,
    current_nice: int,
    capable: bool,
    starttime_fn: Callable[[int], float | None] | None = None,
    nice_reader: Callable[[int], int | None] = _default_nice_reader,
    nice_setter: Callable[[int, int], None] = _default_nice_setter,
) -> int:
    """Lower the target's nice back toward its recorded original value.

    Capability-gated: this machine CANNOT do it unprivileged (empirical
    EPERM), so the guard defaults restore_capable=False and this function
    raises ElevationRequired rather than attempting a known-denied syscall.
    """
    if not capable:
        raise ElevationRequired(
            "restore requires privileges; unallowed on this deployment"
        )
    if target_nice > current_nice:
        raise ExecutorError("restore must not lower priority")
    if not (-20 <= target_nice <= 19):
        raise ExecutorError(f"target nice {target_nice} out of [-20, 19]")
    try:
        holder = Pidfd(pid)
    except TargetGone:
        raise
    except PidfdError as err:
        raise ExecutorError(f"pin failed: {err}") from err
    try:
        if holder.exited():
            raise TargetGone(f"pid {pid} exited while pinned")
        assert_same_task(
            pid, expected_start_time, starttime_fn or _default_starttime
        )
        nice_setter(pid, target_nice)
        got = nice_reader(pid)
        if got is None:
            raise ExecutorError("postcondition unreadable after apply")
        if abs(got - target_nice) > 1e-9:
            raise ExecutorError(
                f"postcondition failed: nice={got}, wanted {target_nice}"
            )
        return got
    finally:
        holder.close()
