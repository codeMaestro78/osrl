"""Deterministic thrash-risk tracker (observe-only; no policy, no actions).

Consumes the Phase 0 sample stream plus PSI-trigger firings and classifies
the machine's memory-pressure posture. Rules encoded here follow the audit
brief (§A, target architecture item 3) and DESIGN §11:

- Evidence comes from SAMPLES; a missing metric casts no vote and is never
  reported as zero. Reading zero stall where a counter is absent would be
  fabrication.
- Multi-condition escalation: an ELEVATED state requires persistent
  concurrency of at least two warn signatures OR one strong signature.
  Single blips are explicitly insufficient (the same brief's noise rule).
- CRITICAL additionally requires PRIMARY PSI evidence to be present
  (memory PSI readable in the relevant window). Swap traffic or direct
  reclaim alone can raise the level only to ELEVATED -- an honest cap,
  because our freeze-prevention thesis is PSI-first.
- Persistence/hysteresis: enter a higher level after enter_n consecutive
  bad samples; demote only after exit_n consecutive good samples; any bad
  sample resets the exit streak. This trades ~10 s entry lag for the
  elimination of flapping.
- Staleness gate: stale or missing telemetry forces level 'unknown' and
  clears streaks. 'unknown' is NOT 'elevated': acceptance criterion 2
  (missing/stale telemetry can never autonomously intervene) is enforced
  at this gate, and it must hold when the policy layer arrives (Phase 4):
  nothing downstream may treat 'unknown' as actionable.
- PSI trigger firings (event-driven witnesses) add one extra warn vote while
  fresh (trigger_freshness_s, default 4 s = two tracking windows): the
  kernel rate-limits notifications to one per window, so the absence of a
  second notification is NOT proof of recovery. Thread-safe via note_trigger.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from osrl.collector import Sample
from osrl.config import RiskConfig

LEVELS = ("none", "elevated", "critical", "unknown")

_STRONG = "strong"
_WARN = "warn"


@dataclass
class Assessment:
    level: str
    changed: bool
    stale: bool
    bad_streak: int
    good_streak: int
    strong: list[str] = field(default_factory=list)
    warn: list[str] = field(default_factory=list)
    psi_present: bool = False       # memory PSI was readable in this sample
    psi_recent: bool = False        # readable within the staleness horizon


def sample_evidence(
    s: Sample, cfg: RiskConfig
) -> tuple[list[str], list[str], bool]:
    """Classify one sample's signatures. Returns (strong, warn, psi_present).

    memory-PSI ratio = interval-stall-us / interval-us; only meaningful
    when BOTH the stall diff and a positive interval exist (first tick ->
    None -> no vote, no "evidence").
    """
    strong: list[str] = []
    warn: list[str] = []

    ratio: float | None = None
    if (
        s.interval_s
        and s.interval_s > 0
        and s.psi_mem is not None
        and s.psi_mem.some_stall_us is not None
    ):
        ratio = s.psi_mem.some_stall_us / (s.interval_s * 1e6)
        if ratio >= cfg.psi_mem_stall_crit:
            strong.append("psi_mem_stall")
        elif ratio >= cfg.psi_mem_stall_warn:
            warn.append("psi_mem_stall")
    psi_present = ratio is not None

    if s.swp_out_ps is not None:
        if s.swp_out_ps >= cfg.swp_out_crit_ps:
            strong.append("swap_traffic")
        elif s.swp_out_ps >= cfg.swp_out_warn_ps:
            warn.append("swap_traffic")

    if s.allocstall_ps is not None:
        if s.allocstall_ps >= cfg.allocstall_crit_ps:
            strong.append("direct_reclaim")
        elif s.allocstall_ps >= cfg.allocstall_warn_ps:
            warn.append("direct_reclaim")

    if s.mem_available_bytes is not None:
        mib = s.mem_available_bytes / (1024 * 1024)
        if mib <= cfg.memavail_crit_mib:
            strong.append("low_mem_available")
        elif mib <= cfg.memavail_warn_mib:
            warn.append("low_mem_available")

    return strong, warn, psi_present


class RiskTracker:
    """Stateful but tiny: level + streaks + trigger recency."""

    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or RiskConfig()
        self.level = "none"
        self._bad = 0
        self._good = 0
        self._psi_last_ts: float | None = None
        self._mem_trigger_ts: float | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------- trigger witness feed

    def note_trigger(self, resource: str, ts: float) -> None:
        """Called from the PSI poller thread. Kernel rate-limits firings to
        one per tracking window; see module docstring for the vote math."""
        with self._lock:
            if resource == "memory":
                self._mem_trigger_ts = max(ts, self._mem_trigger_ts or ts)

    def _mem_trigger_recent(self, now: float) -> bool:
        return (
            self._mem_trigger_ts is not None
            and now >= self._mem_trigger_ts
            and now - self._mem_trigger_ts <= self.cfg.trigger_freshness_s
        )

    # ---------------------------------------------------------- sample feed

    def feed(self, sample: Sample | None, now: float) -> Assessment:
        """Feed one tick (or None when no sample exists yet), get verdict."""
        with self._lock:
            return self._feed_locked(sample, now)

    def _feed_locked(self, sample: Sample | None, now: float) -> Assessment:
        prev_level = self.level

        if sample is None:
            self.level = "unknown"
            self._bad = self._good = 0
            return Assessment(
                level="unknown", changed=True, stale=True,
                bad_streak=0, good_streak=0,
                psi_recent=self._psi_recent_locked(now),
            )

        stale = (now - sample.ts) > self.cfg.staleness_s or now < sample.ts
        strong, warn, psi_present = sample_evidence(sample, self.cfg)
        if psi_present:
            self._psi_last_ts = sample.ts

        if stale:
            self.level = "unknown"
            self._bad = self._good = 0
            return Assessment(
                level="unknown", changed=prev_level != "unknown", stale=True,
                bad_streak=0, good_streak=0, strong=strong, warn=[],
                psi_present=psi_present,
                psi_recent=self._psi_recent_locked(now),
            )

        #  PSI trigger firing acts as ONE extra warn vote (bounded, rate-
        #  limited by the kernel). Recorded as its own evidence name so
        #  later audits can distinguish steady evidence from event replies.
        if self._mem_trigger_recent(now):
            warn = warn + ["psi_memory_trigger"]

        bad = bool(strong) or len(warn) >= 2

        if bad:
            self._good = 0
            self._bad += 1
            if (
                self.level in ("none", "unknown")
                and self._bad >= self.cfg.enter_n
            ):
                self.level = "elevated"
            elif self.level == "elevated":
                # PSI is the primary signal (brief A); secondary-only strong
                # evidence stays capped at ELEVATED.
                if self._bad >= self.cfg.enter_n * 2 and psi_present:
                    self.level = "critical"
        else:
            self._bad = 0
            if self.level in ("elevated", "critical"):
                self._good += 1
                if self._good >= self.cfg.exit_n:
                    # two-step demotion: critical walks down one level
                    self.level = (
                        "elevated" if self.level == "critical" else "none"
                    )
                    self._good = 0
            elif self.level == "unknown":
                # fresh clean evidence releases the wait-state (never jumps
                # straight to a high level from a stale-blind state)
                self.level = "none"

        return Assessment(
            level=self.level,
            changed=self.level != prev_level,
            stale=False,
            bad_streak=self._bad,
            good_streak=self._good,
            strong=strong,
            warn=warn,
            psi_present=psi_present,
            psi_recent=self._psi_recent_locked(now),
        )

    def _psi_recent_locked(self, now: float) -> bool:
        return (
            self._psi_last_ts is not None
            and now - self._psi_last_ts <= self.cfg.staleness_s
        )
