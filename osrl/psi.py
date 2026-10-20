"""Pressure Stall Information parsing and interval arithmetic.

Kernel contract (https://docs.kernel.org/accounting/psi.html):

- /proc/pressure/{cpu,memory,io} expose two lines, ``some`` and ``full``.
- ``avg10/60/300`` are percentages over those second-windows (trends).
- ``total`` is the *cumulative* absolute stall time in microseconds since
  boot -- therefore interval stall time is always the DIFFERENCE of two
  consecutive ``total`` readings, never the raw value itself.

Caveats implemented here:

- Counter reset (``cur < prev``, e.g. after reboot) yields ``None`` --
  explicitly NOT zero, because a zero stall claim would be fabricated.
- The kernel docs state: "CPU full is undefined at the system level, but has
  been reported since 5.13, so it is set to zero for backward compatibility."
  System-level CPU ``full`` (and it will read zero) is therefore captured for
  auditability but MUST NOT be used as a contention signal. memory/io ``full``
  counters are meaningful.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PsiStats:
    avg10: float
    avg60: float
    avg300: float
    total_us: int  # cumulative microseconds; meaningless in isolation


_LINE_RE = re.compile(
    r"^(?P<kind>some|full)\s+"
    r"avg10=(?P<a10>[\d.]+)\s+"
    r"avg60=(?P<a60>[\d.]+)\s+"
    r"avg300=(?P<a300>[\d.]+)\s+"
    r"total=(?P<total>\d+)\s*$"
)


def parse_psi_text(text: str) -> dict[str, PsiStats]:
    """Parse PSI file content. Raises ValueError on any unparseable line."""
    out: dict[str, PsiStats] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            raise ValueError(f"unparseable PSI line: {line!r}")
        out[m["kind"]] = PsiStats(
            avg10=float(m["a10"]),
            avg60=float(m["a60"]),
            avg300=float(m["a300"]),
            total_us=int(m["total"]),
        )
    if "some" not in out:
        raise ValueError("PSI file without a 'some' line")
    return out


def read_psi(path: str | Path) -> dict[str, PsiStats] | None:
    """Read one PSI file; returns None when the file is missing/unreadable or
    malformed. The collector treats None as 'unknown pressure' and must not
    fabricate zeros from it."""
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    try:
        return parse_psi_text(text)
    except ValueError:
        return None


def interval_stall_us(prev_total_us: int | None, cur_total_us: int | None) -> int | None:
    """Stall microseconds inside the measurement interval, or None if unknown."""
    if prev_total_us is None or cur_total_us is None:
        return None
    if cur_total_us < prev_total_us:
        return None  # counter reset (reboot): cannot infer this interval
    return cur_total_us - prev_total_us
