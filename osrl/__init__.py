"""OS-RL-Agent: local freeze-risk watchdog for Ubuntu.

Phase 0 modules (collector foundations):
- osrl.psi / osrl.vmstat: /proc readers with correct cumulative-counter math.
- osrl.procid: (pid, start_time) process identity primitives.
- osrl.collector: injectable telemetry sampler producing Sample rows.
- osrl.adapters: psutil/NVML wiring for the real machine (not used by tests).
- osrl.store: SQLite WAL telemetry store + intent channel.
- osrl.daemon: single-instance collector daemon.

Policy, guard, twin, and RL arrive in Phases 1-4; see DESIGN.md section 11
for the research-amended roadmap.
"""

__version__ = "0.1.0"
