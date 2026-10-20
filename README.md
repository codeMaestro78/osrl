# OS-RL-Agent

Local-first freeze-risk watchdog for Ubuntu: PSI-based pressure detection,
guarded (reversible) interventions, RL as a candidate policy that must earn
its autonomy. See `DESIGN.md` for the architecture and `DESIGN.md` §11 for
the research-audit amendments.

## Status (Phase 4.0 complete — policy evaluation + evidence pipeline)

Implemented and tested:

- **Collector core** (`osrl/collector.py`): 2 s sampling of CPU (total +
  per-core from `cpu_times` diffs), RAM (`MemAvailable`), swap occupancy,
  GPU via NVML when present; PSI diffing (`some`/`full`, cumulative
  microseconds -> interval stalls); `/proc/vmstat` deltas (pswpin/pswpout,
  pgmajfault, reclaim scan/steal, allocstall); per-process CPU/RSS for top-15
  and heavyweight events (birth/death/pid-reuse, keyed `(pid, start_time)`).
  Derived values are `None` when unknown -- never fabricated zeros. A
  counter reset or an unreadable input yields `None` and resets history so
  diffs never span outages.
- **Store** (`osrl/store.py`): SQLite WAL (`synchronous=NORMAL`),
  explicit-schema v1, batched short-transaction writes, ts-indexed, 15-min
  rollups (PSI stall seconds, never raw occupancy as evidence), retention
  (raw 7 d / events 30 d / rollups per DESIGN §7), durable heartbeat,
  **intent channel** with `pending -> claimed -> applied/rejected/failed ->
  expired` lifecycle, atomic single-owner claims (compare-and-set inside
  `BEGIN IMMEDIATE`), lease reconciliation for dead claimants, poison
  expiry at `max_tries`. The `actions_ledger` table intentionally does not
  exist yet: it is Phase 2's schema, added with its guard semantics, not
  pre-created.
- **Daemon** (`osrl/daemon.py`): single-instance via flock on
  `<db-dir>/collector.lock`, interval-paced sampling (first sample
  immediate), batched flushes, 30 s heartbeat (with MEASURED own-CPU% via
  `os.times` delta, **and now the current `risk_level`**), intent-lease
  reconciliation, hourly retention prune + WAL passive checkpoint, 15-min
  rollup windows (grid-anchored, backlog bounded to 96 windows/tick and
  tested against multi-day outages: missing windows become dense
  `sample_count=0` rows), graceful SIGTERM/SIGINT flush, `--once` smoke
  mode.
- **Identity primitives** (`osrl/procid.py`): `(pid, start_time)` equality +
  revalidation function consumed later by the Phase 2 guard.
- **Risk estimator** (`osrl/risk.py`, Phase 1): deterministic, sample-fed
  classifier (`none/elevated/critical/unknown`) with a multi-condition
  escalation rule (>=2 concurrent warn signatures OR >=1 strong signature,
  sustained `enter_n` ticks), hysteresis on exit (`exit_n` calm ticks,
  stepwise demotion), **missing metrics cast no vote and are never reported
  as zero**, and a **staleness gate**: stale or future-dated telemetry
  forces `unknown` and clears streaks. `unknown` is never conflated with
  `elevated`; a fresh clean sample releases the wait-state. CRITICAL
  additionally requires primary memory-PSI evidence to be present in the
  window (secondary-only strong evidence is honestly capped at ELEVATED).
  Level changes are recorded as `risk_change` events with the evidence
  lists attached.
- **PSI event triggers** (`osrl/psi_trigger.py`, Phase 1): arms real kernel
  monitors (memory `some 150ms/2s`, io `full 50ms/2s` — windows are the
  smallest unprivileged-legal multiple of 2 s), waits with `poll()` on
  POLLPRI, records `psi_trigger_fired` / `psi_trigger_registered` /
  `psi_trigger_failed` / `psi_trigger_dead` events through its **own
  thread-local store connection**; poller failures land in the ledger as
  `psi_poller_error` and can never take sampling down. Kernel contract was
  probed and VERIFIED live on this machine (EBUSY on double-arm, EINVAL
  for non-2s-multiple windows, fd close deregisters; a monitor may fire
  immediately at arm time — fired events are treated as evidence, never
  as errors). Memory firings add ONE bounded warn vote to the risk tracker
  (`note_trigger`, fresh for 4 s = two windows; the kernel rate-limits to
  one notification per window, so absence of a second notification is not
  proof of recovery).

Not implemented yet (by plan): autonomous policy promotion / the remaining
Phase 5 hardening. The Streamlit control plane now ships in
`osrl/dashboard/`, with a Community Cloud entrypoint (see Deployment below).

### Phase 2 — guarded action infrastructure (implemented, autonomy OFF)

- **pidfd** (`osrl/pidfd.py`): `os.pidfd_open`-based pinning — while held,
  the kernel cannot recycle the pid; poll(0) reports exit; identity is
  re-validated INSIDE the hold. The reuse window between revalidation and
  syscall is closed by construction.
- **Guard** (`osrl/guard.py`): deterministic first-failure-wins checker
  (kind whitelist, pid sanity incl. never-self, (pid,start_time) identity,
  same-uid ownership, TELEMETRY STALENESS GATE — no samples in store or
  staleness > `staleness_max_s` hard-denies —, capability/state (see below),
  per-(pid,kind) rate limiting where even FAILED attempts count, and the
  intent lease: live execution consumes a claimed intent we own). The
  guard never mutates anything and never retries; the ledger records the
  exact one reason for a denial. Structurally RL-free (AST-import scan
  test in `tests/test_safety_invariants.py`).
- **Executor** (`osrl/executor.py`): the ONLY `setpriority` surface.
  `renice_back` raises nice on same-uid targets (verified live, 0→10) and
  verifies the postcondition by read-back. **RESTORE IS ELEVATION-ONLY BY
  EVIDENCE**: unprivileged nice lowering returns EPERM cross-process AND
  self (probed 2026-10-20, RLIMIT_NICE=(0,0)) — so `restore_capable=False`
  gates it off and the guard rejects restore with
  `restore_requires_elevation`; the Phase 4 RL action mask must mask
  restore off whenever that capability is false. No SIGKILL/SIGSTOP
  primitive exists anywhere in the repo.
- **Ledger** (`osrl/store.py`, schema v2): actions_ledger gained
  `target_nice/before_nice/applied_nice/parent_action_id/intent_id/
  guard_checks/error/ts_applied/ts_restored` + indexes; state machine
  proposed→approved→applied|failed (CAS updates), strict-column inserts,
  `last_action_ts` rate-limiter query, prune with events_days. v1→v2
  migration tested on a legacy-shaped db (rows preserved).
- **Worker** (`osrl/worker.py`): the ONLY intent→action path. Dry mode
  (default) peeks and adjudicates nothing claimed; execution requires the
  operator's explicit `--execute` flag per invocation — that token is the
  human approval gate for this phase (the daemon never imports/starts the
  worker; enforced by test). Malformed intents are rejected with reasons;
  invocation-limit releases claims back to pending instead of consuming
  them (`release_claim`).
- **Store/collector upgrades**: schema v2 also adds
  `samples.gpu_top_procs_json` and per-process VRAM collection (NVML
  graphics+compute clients merged per pid; verified readable live:
  5 desktop clients with exact bytes), retained top-5 by VRAM.
- **Heavyweight thresholds retuned from recorded evidence** (88 births /
  0.7 h): (50% CPU, 512 MiB RSS) replaces (25, 200) — the CPU side never
  fired for newborn processes (~0 accumulated time) and the old RSS floor
  was pure Chrome-utility noise (median 273 MiB); the new pair keeps 13%
  of the noise while retaining meaningful births. Both live behind
  `OSRL_HEAVY_*` env overrides; risk thresholds behind `OSRL_RISK_*`,
  guard behind `OSRL_GUARD_*` (tuning now needs no code edits).

### Phase 3 — digital twin + baselines (implemented, observe-only)

The twin is the **measurement layer** any RL policy must be able to beat;
it is NOT a world-simulator and invents nothing. Everything scores
(predictions from data ≤ t) against RECORDED rows at t+h.

- **Session slicing** (`osrl/twin.py`): telemetry is cut into sessions
  wherever a sample gap exceeds `max(3× median interval, 10 s)` — no
  prediction is ever asked to bridge a daemon outage; every baseline
  restarts fresh per session. Missing metrics stay missing (None), and
  every score publishes its own coverage (`n_pairs`, `n_abstained`).
- **Baseline registry**: `noop` (always abstains — the floor),
  `hold` (persistence: last observed value), `linear` (OLS trailing-window
  trend with per-feature abstention until ≥3 points spanning the window).
- **Causal evaluation harness**: targets matched nearest-first within 1%
  of the sampling interval, never interpolated, never across sessions;
  horizons 30/60/120 s; mae/bias computed only over (predicted, recorded)
  pairs both sides real values.
- **Renice forward model** (spec `SPEC_RENICE_TWIN`): action-conditioned
  difference-in-differences over applied ledger rows vs matched pre/post
  windows; **needs 8 real pairs, has 0, so it abstains on every
  prediction** (`insufficient-real-evidence(0<8)`) — the same honest
  refusal as the staleness gate, no fabricated effect sizes.
- **Report CLI**: `osrl-twin --db PATH [--horizons 30,60,120]` prints a
  JSON report; both console scripts (`osrl-collector`, `osrl-twin`) are
  now exercised after a fresh `pip install -e .`.

**Measured on the real hold-test bank** (1.78 h continuous, 223 samples,
single session; timestamp jitter ±0.0003 s): hold beats linear at
essentially every horizon/feature (details in DESIGN.md §14.4) — e.g.
cpu MAE 1.66% vs 1.82% at 30 s, 2.14% vs 2.50% at 60 s, 2.59% vs 3.45%
at 120 s; psi stall-ratio MAE ~2e-4 (hold). On a mean-reverting,
mostly-idle desktop, persistence is the number to beat, so the Phase 4
acceptance bar is *beat-hold on this harness*, not beat-noop.

### Phase 4.0 — policy evaluation + evidence pipeline (implemented)

- **Candidate forecasters** (`ewma`, `rolling-median`) added to the
  registry; beat-hold verdict is now MEASURED, not asserted (DESIGN
  §15.1): rolling-median wins the bursty stress features (psi stall
  ratio, swap-out rate, allocstall) by ~2-3x; hold wins the level
  features (mem available, swap occupancy, cpu total). The Phase 4 bar
  is now the per-feature blend — and one binding caveat: the median is
  burst-blind, so alarm inputs must stay hold/max-shaped.
- **Policy-level evaluation** (`osrl/policy_eval.py`): alarm policies
  scored against a frozen mechanical event labeler (≥3 consecutive
  critical samples; outages never bridged). Shipped comparison:
  deployed risk tracker (replayed unchanged, real `Sample`
  reconstruction) vs a naive single-sample threshold strawman.
  Hysteresis demonstrably buys 1 fewer false-alarm run on a labeled
  scenario at a documented latency cost (18 s at defaults). Zero-event
  banks report recall=None (unmeasurable), never 0/0 coerced.
- **Forward-model evidence pipeline**: applied ledger rows are matched
  with ±60 s telemetry (target CPU share via top-15 (pid, start_time)
  identity, per-side tick floor, unusable windows REPORTED not
  interpolated); `ReniceForwardModel` now counts usable pairs, fits a
  median shift, activates at ≥8 pairs, refuses horizon mismatch /
  non-positive deltas, stamps every forecast UNVALIDATED.
- **Live validation** on a dedicated 2 s experiment bank (shipped
  default thresholds only — no env overrides): 2 operator-owned busy
  children pinned to one core; both intents dry-peeked first, then run
  through the standard worker `--execute` gate. Result (median shares
  of machine total): reniced child 4.17%→0.79% while its contender
  rose 4.17%→7.52% (one-core conservation), then both reniced →
  4.17/4.17 again. The renice mechanism is live-verified in both
  directions.
- **Defect found by the pipeline** (not by review): the worker never
  recorded `before_nice`, so ledger `nice_delta` was always NULL.
  Fixed and test-locked; the two actions taken before the fix stay
  delta-NULL in history (not retro-fitted).
- Model state: 1 usable pair < 8 → still abstains everywhere (fit
  summary publishes n=1 shift −3.35 points with the caveat). PPO
  remains correctly deferred: no policy has beaten the blend, and the
  recorded bank contains zero events to learn from.

## Quickstart (this repo)

```bash
python3 -m venv .venv
./.venv/bin/pip install -q psutil nvidia-ml-py pytest ruff  # dev install
./.venv/bin/pip install -e .                                 # optional: CLI entry
./.venv/bin/pip install -e ".[dashboard]"                    # optional: Streamlit UI

./.venv/bin/python -m pytest -q          # full test suite (169 tests)
./.venv/bin/python -m osrl.daemon --db data/telemetry.db --once   # smoke
./.venv/bin/python -m osrl.daemon --db data/telemetry.db         # run forever

# Intents -> actions worker: DRY by default; --execute is the explicit
# per-invocation human approval gate (the daemon NEVER runs this)
./.venv/bin/python -m osrl.worker --db data/telemetry.db          # dry peek
./.venv/bin/python -m osrl.worker --db data/telemetry.db --execute

# Digital twin: evaluate baseline forecasters over recorded telemetry
./.venv/bin/osrl-twin --db data/telemetry.db --horizons 30,60,120

# Dashboard: live local collector data, then open http://localhost:8501
./.venv/bin/osrl-dashboard --db data/telemetry.db
# (or the Cloud entrypoint, which auto-selects live/demo DB)
./.venv/bin/streamlit run streamlit_app.py
```

Configuration via environment (see `osrl/config.py`): `OSRL_DB`,
`OSRL_INTERVAL`, `OSRL_RAW_DAYS`, `OSRL_ROLLUP_SECONDS`, `OSRL_HEARTBEAT_S`,
`OSRL_BATCH_SIZE`, `OSRL_BATCH_INTERVAL_S`, `OSRL_WAL_CHECKPOINT_S`.
Defaults follow DESIGN §7 (2 s cadence, 7 d raw, 15-min rollups kept).
Risk/trigger thresholds are code-level defaults in
`osrl.config.{RiskConfig,PsiTriggerConfig}` (hypotheses to retune on data;
PSI-trigger windows must stay multiples of 2 s for unprivileged monitors).

## Deploy the dashboard on Streamlit Community Cloud

Streamlit Community Cloud runs only the Streamlit process — it cannot read
your laptop's `/proc`, run the collector daemon, or keep a SQLite file across
restarts. The Cloud deployment therefore replays a **compact recording of real
telemetry** (`data/demo_telemetry.db`, 7 h / 12k samples from this workstation)
and labels itself as a snapshot in the UI.

**Files that make it work**

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | Cloud entrypoint (repo root); resolves the DB and imports the dashboard |
| `requirements.txt` | Dashboard-only deps (no torch / stable-baselines3 — keeps the build ~1 GB smaller) |
| `osrl/dashboard/bootstrap.py` | DB resolution: `OSRL_DB` → local live DB → bundled snapshot → empty store |
| `data/demo_telemetry.db` | Recorded snapshot, copied to `/tmp` at startup so writes never touch the repo |
| `.streamlit/config.toml` | Dark theme + headless server defaults |

**Deploy**

1. Push the repo to GitHub (public, or private with Streamlit's GitHub app
   granted access). The snapshot must be committed — `.gitignore` already has
   an exception for `data/demo_telemetry.db`.
2. Go to <https://share.streamlit.io> → **New app** → pick the repo/branch.
3. Set **Main file path** to `streamlit_app.py`.
4. Optional: choose Python 3.12 in **Advanced settings**. No secrets are
   required.
5. Deploy. Startup copies the snapshot to `/tmp/osrl-dashboard/demo.db` and
   opens on the Overview page.

**Regenerate the snapshot** after collecting more telemetry:

```bash
python deploy/build_demo_db.py \
    --src data/telemetry.db \
    --dst data/demo_telemetry.db \
    --hours 24 --max-procs 8
```

The builder preserves schema version, rollups, events, and ledger rows; it
only trims old samples and compacts per-process JSON.

**Cloud limitations (by design, not bugs)**

- Live metrics are from the recording, not the Cloud container. The banner
  says so; the Overview card reads **Dataset → Recorded**.
- Agent mode changes, rollups, and intents persist only for the life of the
  container (ephemeral filesystem).
- Training launches are disabled: the Cloud build intentionally omits torch.
- For genuine live monitoring use the local dashboard (`osrl-dashboard`), the
  systemd units in `deploy/`, or the Docker image (`Dockerfile`) on Fly.io /
  Render / Railway, where the collector and dashboard run together under
  supervisord.

## Measured on the live workstation (2026-10-20)

### Phase 2 live verification (guarded execution)

- Full intent→claim→guard→pidfd→apply→ledger chain against a real child
  process owned by the operator account: renice_back 0→10 applied and
  read-back verified; the store staleness gate saw 0.2 s fresh telemetry.
- Negative paths held live: a second renice_back 0 s later was
  rate-limit-rejected (300 s spacing; the FAILED rows count too), and a
  restore intent citing the real applied parent row was rejected with
  `restore_requires_elevation` — target untouched.
- CLI path: `python -m osrl.worker --db ... --execute` exited 0 with an
  applied JSON summary; dry mode consumed nothing.
- Ledger: exactly one row per intent (`approved`→`applied` or
  `rejected`), guard_checks JSON recorded per row, intents consumed
  exactly once (3 pending → 0 pending).

### Collector measurements (unchanged from Phase 1)
- 2.000 s cadence ± 0.2 ms; collector's own CPU 0.15% (heartbeat payload,
  `os.times` delta) -- under the 1% budget. Worst single-tick collection
  80 ms wall on ~450 processes (burst, not sustained CPU). Phase 2 smoke:
  own CPU 0.25%, heartbeat carries `pending_intents` and `risk_level`.
- ~7 KB/sample with WAL truncation: one day of raw 2 s data ~= 300 KB, so
  the 7-day raw retention window is a few MB -- measured, not assumed.
- Non-toxic data: first tick rates are NULL; swap traffic (pages/s) only
  populated from the second tick; a reboot or dead /proc read yields NULL
  rather than a negative/fabricated value.
- Phase 1 triggers under real desktop load: both monitors armed and fired
  naturally (io `full` 3-4x, memory `some` 0-1x per ~55 s run) and each
  firing produced a ledger event; the risk level stayed `none` (single
  warn votes do not escalate -- no false alarms). Swap 98% occupied with
  ZERO swap traffic and zero PSI stalls correctly classified `none`:
  occupancy is not thrashing.
- Per-process VRAM (Phase 2): NVML graphics-class list exposed 5 desktop
  clients with exact per-pid bytes; merged compute/graphics top-5 persisted.

## Safety posture now

- Guarded but gated: the ONLY live mutation path is the worker CLI in
  explicit `--execute` mode, consuming intents you (or a future dashboard)
  insert. The autonomous path does not exist: the daemon never starts the
  worker (enforced by test), the guard's staleness/rate/ownership/
  capability checks all widen the denial set, and RL never touches
  enforcement (enforced by AST-import test).
- restore is evidence-capped: raising nice is unprivileged-safe and was
  verified live; lowering it back is EPERM unprivileged (cross-process AND
  self, probed twice) so `restore_capable=False` until an elevated helper
  is deliberately configured — never silently.
- No SIGKILL/SIGSTOP/termination primitive exists anywhere; renice_back is
  verified monotonic (never raises priority), postcondition-read-back
  enforced, and pidfd-pinned against pid reuse.
- Observe mode remains the default; nothing ran autonomously in any smoke
  — all executions in this repo's history targeted test children owned by
  the operator account, with every intent→ledger row auditable in the
  store.
- The intent channel's executable semantics now exist (via the worker);
  the dashboard (Phase 5) will write intents through the same audited
  channel, behind the same explicit gates.
