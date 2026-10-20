"""/proc/vmstat deltas as secondary memory-pressure evidence.

Rationale (research brief A): high swap *occupancy* does not prove active
thrashing. What does discriminate is swap *traffic* and reclaim work:

- pswpin / pswpout -- pages swapped in/out since boot (cumulative)
- pgmajfault       -- major page faults (cumulative)
- pgscan_kswapd / pgsteal_kswapd / pgscan_direct / pgsteal_direct --
  reclaim scanning / reclaim progress (cumulative)
- allocstall_*     -- direct-reclaim entry events, the sharpest pressure
  signature (cumulative; kernel exports one line per zone, so we sum a PREFIX)

Counters missing on a given kernel are reported as None -- "unknown", never a
fabricated zero. Counter reset between snapshots (cur < prev) also yields None.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_LINE_RE = re.compile(r"^(\S+)\s+(\d+)$")

# metric -> exact counter names, summed when present, None if none present
_EXACT: dict[str, tuple[str, ...]] = {
    "swp_in": ("pswpin",),
    "swp_out": ("pswpout",),
    "majflt": ("pgmajfault",),
    "reclaim_scan": ("pgscan_kswapd", "pgscan_direct"),
    "reclaim_steal": ("pgsteal_kswapd", "pgsteal_direct"),
}

# metric -> name prefix (sums allocstall_dma, allocstall_normal, ...)
_PREFIX: dict[str, str] = {
    "allocstall": "allocstall",
}


@dataclass(frozen=True)
class VmSnapshot:
    counters: Mapping[str, int]


def read_vmstat(path: str | Path = "/proc/vmstat") -> VmSnapshot | None:
    """Parse /proc/vmstat. None when the file is unreadable or empty."""
    try:
        text = Path(path).read_text()
    except OSError:
        return None
    counters: dict[str, int] = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            counters[m.group(1)] = int(m.group(2))
    return VmSnapshot(counters) if counters else None


def metric_values(snap: VmSnapshot) -> dict[str, int | None]:
    """Snapshot values per metric; None = counter absent on this kernel."""
    out: dict[str, int | None] = {}
    for metric, names in _EXACT.items():
        found = [snap.counters[n] for n in names if n in snap.counters]
        out[metric] = sum(found) if found else None
    for metric, prefix in _PREFIX.items():
        found = [v for k, v in snap.counters.items() if k.startswith(prefix)]
        out[metric] = sum(found) if found else None
    return out


def metric_deltas(
    prev: VmSnapshot | None, cur: VmSnapshot | None
) -> dict[str, float | None]:
    """Per-metric event-count deltas over the interval between two snapshots.

    None: first tick, unreadable file, missing counter, or counter reset.
    Never returns a fabricated zero for a hidden/reset counter.
    """
    if prev is None or cur is None:
        return {m: None for m in (*_EXACT, *_PREFIX)}
    cur_vals = metric_values(cur)
    prev_vals = metric_values(prev)
    out: dict[str, float | None] = {}
    for metric in (*_EXACT, *_PREFIX):
        c, p = cur_vals[metric], prev_vals[metric]
        if c is None or p is None or c < p:
            out[metric] = None
        else:
            out[metric] = float(c - p)
    return out
