# OS-RL-Agent — System Design

A continuously-running Reinforcement Learning watchdog for this workstation:
**sensors → memory → PPO policy → guarded actions → Streamlit control plane.**

Decisions locked (with owner): goal = **prevent freezes**; autonomy = **observe 3 days → advise**;
algorithm = **PPO (MaskablePPO, sb3-contrib)**.

---

## 1. Hardware reality this design is built for

| Component | This machine | Design consequence |
|---|---|---|
| OS | Ubuntu 24.04, kernel 7.0.0 | `/proc/pressure/*` (PSI) available → honest "user is suffering" signal |
| CPU | i5-12450HX, 8 cores / 12 threads, hybrid P+E cores | Niceness + affinity are real levers; pin agent itself to E-cores |
| RAM | 15 GiB, ~9.4 GiB used | Monitor `mem.available` directly, not swap alone |
| Swap | **3.5/4 GiB used (87%)** | Primary threat: swap thrash → freeze. Thrash score is a core feature |
| GPU | RTX 3050 Laptop 6 GB, driver 595.99.02, CUDA 13.2 | Agent runs **CPU-only** (tiny MLP) so it never competes for VRAM |
| Disk | 468 GB NVMe, 48% used | PSI io-pressure tracked; disk pressure actions out of scope v1 |
| Workload | Chrome, VS Code, GNOME, dev CLIs — all user-owned | Whitelist covers ~90% of CPU hogs **without root** (renice/SIGSTOP on own pids) |

## 2. Agent mission

> Keep the machine responsive: predict and prevent swap-thrash freezes, protect
> interactive foreground work, surgically deprioritize background hogs.

The agent does not "add resources". Its value is **timing and restraint**:
detect pre-freeze signatures minutes early, act gently and reversibly, and learn
that most of the time the correct action is *nothing*.

## 3. RL formulation

- **Framework**: gymnasium Env + `MaskablePPO` (sb3-contrib) with action masking.
- **Agent network**: MLP (2×128), ~100 KB, trains and infers on CPU.
- **Environment**: live OS when allowed; a **digital twin** (replay + learned
  dynamics over recorded telemetry) for offline pretraining and nightly eval.
- **Decision cadence**: one decision per 20 s, built from a window of 2 s samples
  (last 10 samples → ~100-dim observation, running-normalized via VecNormalize).
- **Episodic structure**: continuing task (`done=False`), no resets of the real OS;
  episodes are bookkeeping windows for learning curves only.

### Observation (~100 dims)

| Group | Features |
|---|---|
| System scalars | per-core CPU% (12), load-avg/threads, iowait, ctx-switches/s |
| Memory | RAM avail/free, cache, **swap-in/swap-out rate (thrash score)**, swap% used |
| Pressure | PSI: cpu / memory / io — avg10, avg60, avg300 + total stall seconds delta |
| GPU | util%, VRAM used/total, temp, power draw, # GPU procs |
| Disk | read/write IOPS, throughput, PSI io avg10 |
| Top-K processes (K=15) | CPU%, RSS share, age (log), current nice bin, suspended flag, is-new (first 60 s), GPU memory |
| Events | newest heavyweight process appeared flag; agent's own recent action flags |

Process identity is stable across pollers: keyed by `(pid, start_time)` with a
tracked-list that follows the top-K + protected/whitelist members.

### Action space (MultiDiscrete, masked)

For each tracked process: `{NOOP, RENICE_BACK(+10 polite ancestry...), ..., FREEZE (SIGSTOP), UNFREEZE (SIGCONT), FLAG_FOR_REVIEW}` → 4 decision values × K=15, plus 2 global actions:

- **RENICE_BACK**: lower priority (`nice +10..+19`) — reversible, no signal sent
- **FREEZE / UNFREEZE**: `SIGSTOP` / `SIGCONT` — reversible, safer than kill
- **FLAG_FOR_REVIEW**: promotion to UI notification, zero risk
- Global: `{NOOP, TUNE_THRESHOLDS}` (agent adjusts its own alert/sampling knobs)
- **No killing in v1.** Termination is a human-confirmed action from the dashboard.

The **safety masker** removes illegal moves before the policy sees them
(non-owned pids, protected pids: gnome-shell/Xorg/our own daemon, cooldown-bound
targets, freeze of an already-frozen process, etc.).

### Reward (freeze-prevention weighting)

All terms evaluated over the 20 s decision window:

```
r = −3.0 * psi_memory_stall_sec        # seconds tasks stalled for memory
  − 2.0 * psi_io_stall_sec
  − 2.5 * norm(swap_out_rate)          # thrash detector, biggest weight
  − 1.0 * psi_cpu_stall_sec
  + 1.0 * headroom(ram_avail)          # saturating bonus ≥ 3 GiB available
  + 0.5 * (1 − swap_used_pct)
  + 0.8 * foreground_share             # gnome-shell/Xorg keep their CPU share
  − 0.05 * n_actions_this_step         # cost of acting → learns restraint
  − 1.5 * remorse                      # human reopens / undoes agent action
```

Reward credit is naturally delayed (actions now → fewer stalls 1–5 min later);
PPO's return estimation handles this, and we additionally log **60 s post-action
PSI/stall deltas** as per-action effectiveness records (returned to the twin).

## 4. Architecture

```
┌──────────────────────────── Streaming layer ────────────────────────────┐
│ collector daemon (asyncio, 2 s tick, runs 24/7, independent process)     │
│ psutil • nvidia-ml-py (NVML, never spawns nvidia-smi) • /proc/pressure   │
└───────────────┬─────────────────────────────────────────────────────────┘
                ▼
        SQLite (WAL) ring buffer  ◄── single source of truth / message broker
                │      tables: samples(2s, 7d) · rollups(15m, forever)
                │              actions · rewards · psi · events · intents
      ┌─────────┴──────────────────┐
      ▼                            ▼
┌───────────────────┐     ┌─────────────────────────────────────┐
│ RL Engine (proc)  │◄───►│ Safety Guard                         │
│ MaskablePPO       │     │ ownership whitelist · action masking │
│ live env / twin   │     │ cooldowns · mode gates · undo ledger │
│ ckpt every 30 min │     │ every action reversible + logged     │
└───────────────────┘     └─────────────────────────────────────┘
                │
                ▼
   Streamlit multipage app  = viewer + control plane (read SQLite,
   write `intents`; never trains, never samples; app crashes ≠ learning stops)
```

**Key decision**: the RL loop never lives inside Streamlit. The daemon is the
only process that samples and trains; Streamlit reads the store and issues
commands (mode switch, approvals, manual freeze) via an `intents` table the
daemon consumes. This makes the system survive app restarts and keeps telemetry
running 24/7.

**Process identity for actions**: every executor call passes
`(pid, start_time, depth=0, depth_limit=2)` — if the process already exited and
PID reuse is detected at start_time mismatch, the action is dropped and logged.

### Digital twin (two uses)

1. **Pre-training**: the policy learns from recorded day-1 telemetry with a small
   learned dynamics model (residual MLP over replay trajectories), so day one is
   competent instead of randomly exploring a work machine.
2. **Nightly evaluation**: candidate checkpoint vs. (a) frozen baseline policy,
   (b) do-nothing, (c) static rule heuristic. Advisory mode unlocks only when the
   candidate beats all three on replayed stress windows.

## 5. Autonomy ladder (owner-approved)

| Mode | Behavior | Unlock condition |
|---|---|---|
| Observe (default) | Collects, predicts, trains on twin nightly; never acts | — |
| Advise | Posts ranked suggestions in dashboard; human applies | ≥3 days telemetry + twin eval gate |
| Autopilot | Acts within guard: owned pids only, reversible only, ≤1 action/30 s, auto-undo ledger | Manual switch + explicit confirmation |

## 6. Streamlit platform

| Page | Contents |
|---|---|
| **Overview** | Live gauges: per-core CPU (P vs E visual), RAM/swap pressure bar, GPU, PSI stall sparklines. `st.fragment(run_every=2)` for the live strip; page itself static |
| **Processes** | Top-K table sorted by CPU/RSS with agent opinion per row ("deprioritize: chrome renderer at 25%"); manual freeze/renice/undo buttons → intents |
| **GPU** | Util, VRAM per process (NVML), temp/power curves |
| **Agent** | Mode selector, live reward chart, action feed with approve/undo, most recent observation vector |
| **Training** | PPO curves (ep_reward, loss, entropy, clip-frac), checkpoint list, "retrain from replay" |
| **Anomalies** | Pre-freeze predictions, drift-from-baseline, incident timeline with telemetry replay |
| **History** | 7-day rollups, hour-of-day heatmaps ("when does this laptop thrive") |

## 7. Data store design

- `samples`: 2 s rows, partitioned by day, **7-day retention**, WAL sqlite.
- `rollups`: 15-min aggregates, kept forever (the "profile" the agent learns from).
- `actions`: intent → guard verdict → executor result → 60 s post-stall delta.
- `events`: process birth/death of heavyweights, anomaly detections, OOM-adjacent signatures.
- `intents`: Streamlit→daemon control channel (mode switches, approvals, manual actions).
- Sizes: ~30 MB/day for samples; sqlite fine. No server needed.

## 8. Continuous operation

- Daemon: `systemd --user` unit (auto-restart, WAN-free, local only) — alternative `nohup` profile for v0 testing.
- Checkpoints every 30 min + best eval checkpoint; crash resume = restart from last good checkpoint + replay the missed samples.
- VecNormalize stats are saved with each checkpoint (no drift surprises).
- The daemon self-reports into `events` (heartbeat every 30 s); a missing heartbeat shows a red banner on the dashboard.
- The agent's own footprint: collector < 1% CPU, trainer bursty ≤ 15% CPU on E-cores, RAM ≤ 300 MB, **GPU never touched**.

## 9. Stack (to install)

`psutil`, `nvidia-ml-py`, `gymnasium`, `torch` (CPU build), `stable-baselines3`, `sb3-contrib`, `streamlit`, `plotly`, standard-library `sqlite3/asyncio/apscheduler` otherwise. Python 3.12 compatible with all.

## 10. Build phases

| Phase | Deliverable | Exit criteria |
|---|---|---|
| 0 | Collector + SQLite ring buffer + minimal Streamlit observe dashboard | 24 h clean telemetry, <1% CPU overhead |
| 1 | Twin env + reward implemented; offline batch eval harness works | Twin replays recorded day accurately; reward curve sanity-checked |
| 2 | MaskablePPO training pipeline + checkpoints | Beats do-nothing baseline on twin stress windows |
| 3 | Guard + executors (renice/SIGSTOP) + intent channel | Manual kill-switch verified; undo ledger proven |
| 4 | Advise mode (suggestions in dashboard) | ≥3 days data + eval gate passed |
| 5 | Autopilot (guarded) + polish (history/anomalies pages) | One week stable, remorse rate ≈ 0 |

---

## 11. Amendments (v0.2 — research audit)

Verified against primary sources on 2026-10-20:
https://docs.kernel.org/accounting/psi.html (kernel docs), the cgroup-v2
admin guide, `pidfd_send_signal(2)`, the SQLite WAL paper, PPO/MaskablePPO
docs (SB3/sb3-contrib), and the linked papers (Decima; model-based RL
compounding error; safe action masking arXiv:2006.14171).

1. **PSI units (verified)**: `total` counters are cumulative microseconds
   since boot; interval stall time must be a DIFF between consecutive
   readings. Raw avg10/60/300 are trend percentages, not stall amounts.
   Counter reset (reboot) yields "unknown" (None), not zero. Implemented
   in `osrl.psi` / `osrl.collector`.
2. **CPU full caveat (verified)**: kernel docs state "CPU full is undefined
   at the system level, but has been reported since 5.13, so it is set to
   zero for backward compatibility." System CPU `full` is therefore
   recorded for audit only and excluded from any future risk/reward logic.
3. **Swap occupancy vs. swap traffic**: resident swap (e.g., 99.9% used on
   this machine) is NOT proof of thrashing. Thrash evidence = pswpin/pswpout
   rates, major faults, reclaim scan/steal, allocstall, PSI memory stalls.
   The +0.5 reward bonus on low swap occupancy in §3 is demoted to a
   hypothesis to validate (brief E); it must never substitute stall time.
4. **Intervention reality checks**: renice changes CPU scheduling, not RAM;
   SIGSTOP does not reliably free resident memory. Advisory responses are
   the default response for unclassified memory pressure. cgroup-v2 limits
   apply ONLY to explicitly managed workloads (never desktop apps as
   arrived-as-is). No autonomous process termination, ever.
5. **Action space narrowed for v1 RL**: one candidate intervention per
   decision — {NOOP, ALERT, RENICE_BACK, RESTORE, REQUEST_APPROVAL} —
   replacing per-process multi-decision arrays (maskable actions, arXiv
   2006.14171). Dashboard manual controls remain a separate, richer surface.
6. **Process identity hardening**: `(pid, start_time)` everywhere; pidfd
   (`pidfd_send_signal(2)`) replaces raw signals once the guard ships.
   The guard is a separate component and never consults RL inference;
   stale/missing telemetry forces abstention.
7. **Reward hygiene**: coefficients in §3 are hypotheses. Evaluation must
   measure delayed outcomes at 30/60/120 s horizons, attribute per-action
   effects independently, and report false-intervention and restoration
   rates. Higher training reward alone is NOT validation of freeze prevention.
8. **Digital twin discipline**: must model action-conditioned behavior, not
   replay; §F requirements (held-out sessions, horizon-wise prediction
   error, uncertainty-gated abstention, no-op + heuristic baselines).
   PPO is on-policy: historical telemetry calibrates the twin and provides
   starting states; PPO training rollouts must be freshly generated in the
   simulated env (Decima; compounding-error paper considerations).
9. **Storage/ops verified in code**: SQLite WAL with `synchronous=NORMAL`
   and short transactions; explicit intent states + unique ids + idempotent
   claim; heartbeat; single-instance lock; atomic checkpoint writes planned
   for Phase 4 (policy weights + normalization state, preserve last
   known-good); training runs resource-limited and yields under pressure.
10. **Overhead is measured, not assumed**: live numbers live in README
    ("Measured on the live workstation"); heartbeat publishes own CPU%
    continuously so regressions are visible on the dashboard later.

## 12. Phase 1 implementation record (risk estimator + PSI triggers, 2026-10-20)

### 12.1 PSI monitor contract — verified live on this kernel

Probed unprivileged on kernel 7.0 (results now BINDING for all later phases):

- Arming = open `/proc/pressure/<resource>` with `O_RDWR`, then write
  `"<some|full> <stall_us> <window_us>"`. One spec per fd; a second write
  to the SAME fd returns `EBUSY`. Distinct fds may each hold a spec.
- Unprivileged windows must be multiples of 2 s; others rejected `EINVAL`
  (the 500 ms floor is therefore unprivileged-unreachable; effective legal
  windows: 2, 4, 6, 8, 10 s).
- Closing the fd deregisters the monitor — our poller never closes before
  shutdown, and `PsiTrigger.close()` is the semantic "unregister".
- `poll(POLLERR | POLLPRI | POLLHUP)`: POLLPRI = trigger fired; POLLERR/
  POLLHUP = event source gone (mark trigger dead, stop polling it).
- **Falsified assumption**: a freshly armed monitor MAY fire immediately
  when the resource just exited a stall window (observed). Consequence:
  firings are "recent stall" evidence with `max()`-recency bookkeeping,
  never treated as registration errors.

Implemented in `osrl/psi_trigger.py`: `TriggerSpec` (validated spec text +
half-window local duplicate suppression), `PsiTrigger` (ownership-managed
fd), `PsiPoller` (bounded probe loop, injectable wait for tests, `max_iters`
for deterministic tests, close() = deregister-all). Runs in a daemon
thread with its OWN sqlite connection; any exception is recorded as
`psi_poller_error` and sampling continues unaffected.

### 12.2 Risk estimator rules (deterministic; Phase 1 observe-only)

`osrl/risk.py` — the exact rules a future policy/guard must consume:

1. Evidence per sample: `psi_mem` interval-stall ratio (stall_us/(interval_
   s*1e6); warn >= 0.20, crit >= 0.50), swap-out pages/s (512 / 2048),
   allocstall per second (5 / 25), MemAvailable MiB (<= 768 / <= 256).
   A missing metric casts NO vote; it is never zero.
2. A "bad" sample = any strong signature, OR >= 2 concurrent warn votes.
   Level raises none->elevated after `enter_n=5` consecutive bad samples
   (10 s). elevated->critical requires `enter_n*2` bad samples AND memory
   PSI readable in the window; strong secondary-only evidence stays at
   ELEVATED (PSI-first honesty about the mechanism we are predicting).
3. Exit hysteresis: `exit_n=15` calm samples per demotion step, two-step
   (critical->elevated->none); any bad sample resets recovery progress.
4. Staleness gate: fresh means `now - sample.ts <= staleness_s=6.0` (> 3x
   interval) AND no future-dated clock warp. Stale/missing forces
   `unknown`, clears streaks; `unknown` is never `elevated` and must never
   be consumed as actionable signal downstream (acceptance #2 lands here
   when the policy layer arrives).
5. Trigger witnesses: a memory-PSI firing adds ONE warn vote, fresh for
   `trigger_freshness_s=4` (two windows of grace for the kernel's one-
   notification-per-window rate limiting). Thread-safe (`note_trigger`)
   because the poller thread calls it; numeric sample evidence stays
   primary and authoritative.
6. Recovery from `unknown` requires fresh evidence: a clean fresh sample
   releases it to `none`; bad fresh streaks may escalate normally (the
   blip-blinded state never wedges the tracker, but never escalates on
   stale data either).

Initial thresholds are HYPOTHESES (§11.7 discipline) to be retuned from
recorded sessions before any action-consumption phase; changing them must
stay description-only (config), never weakening the staleness gate or the
"missing = no vote" rule.

### 12.3 Phase 1 live results (this workstation, 2026-10-20)

Two ~55 s daemon smokes with PSI monitors armed:
- `psi_trigger_registered` (memory, io) on every start; `psi_trigger_fired`
  naturally under desktop load (io full 3-4x, memory some 0-1x per run);
  each firing a ledger event; zero `risk_change` events (single warn votes
  did not escalate — no false alarms).
- Heartbeat now carries `risk_level`; own CPU 0.147% — poller adds no
  measurable sampler overhead.
- Decisive negative case: swap 98% occupied with swp_out = 0 pages/s and
  PSI stalls = 0 classified `none` — occupancy is not thrashing (§11.3).

Tests: 73 passing (was 47), ruff clean. New suites: `test_risk.py`
(escalation persistence, two-vote rule, missing-metric honesty, critical-
requires-PSI cap, stepwise hysteresis, stale/future gates, release-from-
unknown, thread-safety), `test_psi_trigger.py` (scripted POLLPRI/dead/stop
semantics, spacing suppression, real-kernel arm/double-arm/window-rule/
deregister/live-poll completeness), daemon tests now cover risk_change
event recording and the multi-day rollup backlog (dense zero-sample
windows — closes the "rollup over multi-day outages" open item).

## 13. Phase 2 implementation record (guarded action infrastructure, 2026-10-20)

### 13.1 The restore-capability finding (empirical, project-defining)

Probed with real children on this kernel (RLIMIT_NICE=(0,0)):
- Renice RAISE on same-uid target: always permitted (0→10 verified).
- Renice LOWER back (restore): EPERM **cross-process** AND **even the
  target lowering itself**. There is NO unprivileged undo path.

Binding consequences:
1. `restore_capable=False` in GuardConfig is evidence, not convention. The
   guard rejects restore with `restore_requires_elevation` before any
   parent-row lookup; the executor refuses to attempt a known-denied
   syscall; the RL action mask (Phase 4) must mask restore whenever the
   capability is false. Flipping it requires a deliberately configured,
   PROVEN elevated helper — never silent.
2. RENICE_BACK must be treated as potentially-non-reversible-without-
   elevation: the ledger persists `before_nice`/`applied_nice` so an
   operator (or future elevated helper) can audit and undo manually. This
   strengthens the case for ALERT/REQUEST_APPROVAL as default responses.

### 13.2 Guard contract (deterministic, first failure wins)

`osrl/guard.py` check order, each denial recorded verbatim in
`actions_ledger.guard_checks`: kind whitelist {renice_back, restore} →
pid sanity (never self/init) → (pid,start_time) identity → same-euid
ownership → telemetry staleness (None or > staleness_max_s=12 s ⇒ deny)
→ capability/state → per-(pid,kind) min spacing 300 s (FAILED attempts
count: `last_action_ts` scans all states) → intent lease (claim must be
ours). renice_back target = min(current+step, 19) with nice < 0 refused
(we never touch already-prioritized tasks). Restore requires an APPLIED
parent renice_back row for the SAME (pid, start_time) and an un-drifted
nice state (`current == parent.applied_nice`), else deny state_drifted.

Structural RL-freedom: `tests/test_safety_invariants.py` AST-parses
guard/worker/executor/pidfd and forbids torch/sb3/gym/sklearn/rl imports;
daemon import scan forbids importing the worker; executor is the repo's
only `setpriority` reference.

### 13.3 Worker gates (autonomy OFF by construction)

`osrl/worker.py`: the daemon never imports it (test-enforced). Dry mode
(default) claims nothing, mutates nothing, prints verdicts. Execution
requires explicit `--execute` per invocation — this invocation-scoped
token IS the Phase 2 approval gate; the dashboard's per-intent approval
arrives in Phase 5 and must flow through the same channel. Malformed
payloads reject with `malformed_payload`; an invocation limit RELEASES
claims back to pending (`release_claim`, never consumes). Reconcile-lease
recovery from dead claimants was already tested in Phase 0 and composes
with the worker (tries++ per claim; poison expiry at max_tries).

### 13.4 pidfd semantics verified

`os.pidfd_open` pins the struct pid: no recycling while open (kernel
guarantee). poll(0) reports POLLIN on process exit — the executor treats
that + an inside-the-hold (pid,start_time) revalidation as preconditions,
then mutates, then READS BACK the applied nice as a postcondition
(mismatch ⇒ typed `ExecutorError`, never assumed success). Opening a
pidfd after the pid was reaped raises ESRCH → `TargetGone` (test-fixed:
our first test naively opened after `wait()`).

### 13.5 Schema v2 + retune (evidence table carried forward)

Migration `user_version 1→2` (tested against a legacy-shaped db): ledger
gains target_nice/before_nice/applied_nice/parent_action_id/intent_id/
guard_checks/error/ts_applied/ts_restored + intent-column index created
post-ALTER only; samples gain gpu_top_procs_json (NVML graphics+compute
clients merged per pid; live-verified 5 clients with exact bytes).

Heavyweight birth thresholds: 88 recorded births over 0.7 h — at
(25%, 200 MiB) the CPU criterion NEVER fired (newborns have ~0 accumulated
CPU time; max observed 1.88%) and the RSS floor was Chrome-utility noise
(median 273 MiB). Evidence-chosen (50%, 512 MiB) keeps 13% of that noise
while retaining the meaningful names; exposed via OSRL_HEAVY_* and
OSRL_RISK_* env overrides so future tuning is config, not code.

### 13.6 Tests: 140 passing (73 before), all suites real where safe

pidfd (5), executor (9: real children, typed failures, no-touch-before-
identity), guard (23: full first-failure ordering matrix), worker (8:
end-to-end on live children incl. dry-run immutability, rate-limit
rejection, elevation rejection, malformed payload, limit release,
identity refusal), ledger+invariants (10: strict columns, CAS chains,
lease release, prune, legacy migration, RL-free scan). Live smoke: dry
peek honest, execute applied 0→10 verified, spam rejected, restore
rejected, ledger 1:1 with intents (3→0 pending), worker CLI rc=0.

## 14. Phase 3 implementation record (digital twin + baselines, 2026-10-20)

### 14.1 What the twin is — and is not

The twin is the MEASUREMENT layer RL must beat, not a script that dreams
up a world. It loads recorded samples, slices them into sessions, and
scores simple causal forecasters at 30/60/120 s horizons. No synthesis of
future states; every scored pair is (prediction made from data ≤ t) vs
(a RECORDED row at t+h).

### 14.2 Session slicing (`osrl/twin.py`)

Sessions = contiguous runs split where a sample gap exceeds
max(3× median observed interval, 10 s). The hold-test bank currently
yields ONE session (223 samples / 1.78 h, zero gaps) — which makes
"held-out folds" degenerate for now; the harness still isolates state
per session (fresh clone), so multi-session data later is automatically
clean. Timestamp jitter on the real bank measured ±0.0003 s (p99), well
inside the 1%-of-interval match tolerance.

### 14.3 Baseline registry

- `noop`: abstains always. The floor: 0 pairs, full abstention column.
- `hold` (persistence): next value = last observed. THE bar to beat.
- `linear` (OLS trailing-window trend): extrapolation with per-feature
  abstention until ≥3 points spanning ≥60 s in-window.
Features (all None-preserving): psi_mem_stall_ratio (diff/interval),
swp_out_ps, allocstall_ps, mem_available_mib, swap_used_pct,
cpu_total_pct.

### 14.4 Measured on the real bank (measured numbers)

n≈220 pairs/horizon; lookahead targets matched nearest-first within
1% of interval, never interpolated, never across sessions.

| feature           | h=30 MAE (hold vs linear) | h=60 | h=120 |
|-------------------|---------------------------|------|-------|
| cpu_total_pct     | 1.66 vs 1.82              | 2.14 vs 2.50 | 2.59 vs 3.45 |
| psi_mem_ratio     | 1.87e-4 vs 1.96e-4        | 2.28e-4 vs 2.73e-4 | 2.23e-4 vs 3.56e-4 |
| swp_out_ps        | 62.2 vs 66.7              | 71.3 vs 97.9 | 72.0 vs 149.4 |
| mem_available_mib | 140 vs 165                | 182 vs 247 | 278 vs 415 |
| swap_used_pct     | 0.21 vs 0.35              | 0.40 vs 0.69 | 0.67 vs 1.33 |
| allocstall_ps     | 0.197 vs 0.148            | 0.164 vs 0.207 | 0.165 vs 0.303 |

Read: persistence dominates trend extrapolation on a mean-reverting,
mostly-idle desktop (linear loses more at longer horizons everywhere
except allocstall at 30 s). Therefore the honest acceptance bar for any
Phase 4 policy is BEAT-HOLD on this harness, not beat-noop.

### 14.5 Renice forward model — abstains by design

The action-conditioned estimator (`ReniceForwardModel`; spec in
`SPEC_RENICE_TWIN` inside twin.py) runs difference-in-differences over
applied-ledger rows against matched pre/post windows. Real applied
renice rows on this deployment: 0 (the Phase 2 live smoke used
`--execute` once, but that DB was a smoke fixture, not the hold-test
bank). So the model reports `insufficient-real-evidence(0<8)` for every
requested prediction. This is the SAME rejection rule as stale
telemetry: no numbers can be manufactured. It activates only when 8+
real (target, before, after, ledger row) quadruples exist with
interpolable sample windows.

### 14.6 Report surface

`osrl-twin --db PATH [--horizons 30,60,120]` (console script wired in
pyproject; exercised live on the hold-test bank) prints a JSON report —
per baseline × horizon × feature — with n_pairs / mae / bias /
n_abstained, so "good MAE over 2 pairs" can never pass as evidence.

### 14.7 Console entries now both exercised

`osrl-collector --help` (risk item #3 earlier flagged it unexercised)
and `osrl-twin` (fresh `pip install -e .` re-install verified both
launch). Remaining known limits: single-session evaluation is
within-sample until more hold-out data accumulates; risk-level
escalation (Phase 1) still only bench-verified; no real ledger
evidence for the forward model yet (see 14.5).

### 14.8 Tests: 155 passing (140 before)

New suite tests/test_twin.py (15): slicing (outage split, cadence
respect, None preservation), baselines (noop floor, hold persistence +
per-key abstention, trend extrapolation exact-OLS + span/points rules),
harness (in-session-only target matching with exact pair counts, mae /
bias / coverage honesty, causality via spike-persistence mae
arithmetic, session isolation), renice model (abstention on real
schema, spec honesty), and the real-bank smoke (hold forms pairs at all
horizons; noop floor exact). Full suite 155; ruff clean.

## 15. Phase 4.0 implementation record (policy eval + evidence pipeline,
2026-10-20)

### 15.1 The beat-the-baseline verdict is now measured, mechanical

Candidate baseline forecasters added to the registry (`ewma`,
`rolling-median`), scored on the grown hold-test bank (2.49 h / 300
samples, still ONE session — within-sample caveat stands):

| feature           | h=30 (hold/ewma/med) | h=60 | h=120 | winner |
|-------------------|----------------------|------|-------|--------|
| psi_mem_ratio     | 1.49e-4/1.48e-4/0.95e-4 | 1.79e-4/1.58e-4/0.92e-4 | 1.75e-4/1.60e-4/0.95e-4 | median 3/3 |
| swp_out_ps        | 45.4/41.3/19.5          | 52.0/46.1/19.6          | 52.4/46.5/18.9          | median 3/3 |
| allocstall_ps     | 0.144/0.099/0.042       | 0.120/0.102/0.042       | 0.120/0.102/0.042       | median 3/3 |
| mem_available_mib | 126/161/276             | 158/192/304             | 228/252/356             | hold 3/3 |
| swap_used_pct     | 0.16/0.39/0.74          | 0.30/0.49/0.83          | 0.50/0.67/0.99          | hold 3/3 |
| cpu_total_pct     | 1.49/1.66/2.40          | 1.91/1.96/2.60          | 2.32/2.30/2.86          | hold 2/3, ewma 1/3 |

Reading (honest): bursty stress features (stall ratio, swap traffic,
direct reclaim) are best predicted by the rolling MEDIAN (they are
spikes around zero — persistence chases spikes and pays); slowly
drifting level features (mem available, swap occupancy, CPU total) are
best predicted by persisting (hold). So the next bar for a Phase-4
policy is the per-feature BLEND, not any single baseline. ONE
downstream caveat is binding: the median is burst-blind, so an alarm
policy must NOT read its alarm inputs from the median-forecaster —
alarm inputs stay hold/max-shaped. MAE-optimality x safety-optimality
are different objectives; recorded deliberately.

### 15.2 Policy-level evaluation (state-forecaster gap closed)

`osrl/policy_eval.py` scores ALARM policies against a FROZEN mechanical
labeler (event = >= 3 consecutive samples at risk-config critical
thresholds, outage gaps never bridged), keeping the same integrity
pattern as the forecast harness: causality, coverage honesty,
zero-positives rules (recall None when no events exist — never 0/0
coerced, never a fabricated perfect score), false alarms as maximal
consecutive runs outside events normalized per hour.

Shipped policies: `risk_tracker` (the deployed estimator replayed,
unchanged, via real reconstructed `Sample` objects) vs
`naive_threshold` (single-sample critical firing, no hysteresis). On a
synthetic labeled scenario the hysteresis discipline is demonstrably
worth real value: identical recall 1.0, but the tracker has 0 false
alarm runs vs the naive strawman's 1, at a fully documented latency
cost (detection delay = enter_n x cadence, e.g. 18 s at defaults).
On the REAL bank: 0 events in 2.49 h -> recall unmeasurable, 0 false
alarms from both policies — the only measurable column on a quiet
desktop, honestly 0.

### 15.3 Forward-model evidence pipeline + first live validation

`extract_renice_pairs(conn)`: every APPLIED renice_back ledger row is
matched with +-60 s telemetry (per-side tick floor 2, target CPU share
via top-15 `(pid, start_time)` identity, psi/swap medians attached) and
unusable windows are REPORTED (e.g. `window-spans-session-break`) not
dropped or interpolated. `ReniceForwardModel` now counts USABLE pairs
(not raw ledger rows), computes a median-shift fit summary, activates
prediction at >= 8 usable pairs, refuses horizon mismatches and
non-positive deltas, and stamps every non-abstaining forecast
`UNVALIDATED`.

Live validation on a DEDICATED 2-second-cadence bank (shipped-default
guard thresholds passed unmodified — no env overrides, so
staleness/rate/capability checks are the production values):

- 2 real operator-owned targets (busy children pinned to one core,
  contending), both intents dry-peeked FIRST (allow), then run with the
  standard worker `--execute` gate: renice 0 -> 10 applied, read-back
  verified on both.
- Direct share series (medians): A 4.17 -> 0.79, contender B 4.17 ->
  7.52 (~10%/90% of the pinned core; conservation held within one
  core), and after B's later action both returned to 4.17/4.17 —
  the renice mechanism works in both directions, live.
- Pipeline pair #1 was rejected with `window-spans-session-break`
  because the experiment daemon had been running for less than the
  full 60 s pre-window: the honesty gate fired correctly in the field.
- DEFECT found and fixed: the worker never recorded `before_nice`,
  so `nice_delta` was NULL in the ledger — discovered by this pipeline,
  not by review. Worker now records the pre-apply read as
  `before_nice` (test-locked in both the worker suite and the pipeline
  suite). The two actions of this session were taken BEFORE the fix, so
  their delta reads NULL; future rows carry it.
- Model state after the session: 1 usable pair < 8 -> still abstains
  everywhere; `fit_summary` publishes the n=1 shift (-3.35 points of
  machine-total CPU share) with the caveat, and the SPEC keeps the
  abstention posture. Nothing is claimed as validated.

### 15.4 Branch decision (per the Phase 4 plan)

Branch (a) — "candidate policy demonstrably beats hold" — is PARTLY
met by a mechanical blend (median-for-stress features, hold-for-level
features), which raises the bar for RL further: a Phase-4 policy must
beat the blend on the same harness, and the blend is beatable mainly
where recorded pressure actually evolves (needs real sessions with
events — the current bank has zero events). Branch (b) — accumulate
real action evidence — is now PROCESS-VALIDATED end-to-end (2 actions,
1 usable pair, defect fixed); reaching the 8-pair activation bar needs
more spread-out operator-approved `--execute` runs (per-target spacing
300 s). PPO still has no reason to exist here yet: no policy to
control has beaten the blend, and the event bank is empty.

### 15.5 Tests: 169 passing (155 before)

New: tests/test_policy_eval.py (7) incl. hysteresis-vs-naive value,
zero-positives honesty, prealarm-only runs; twin suite +9 more
(ewma/median semantics, spike robustness vs hold, evidence extraction
with session-break rejection, model activation arithmetic at 8 pairs,
horizon-mismatch + delta-direction refusal, SPEC honesty). Worker
suite: before_nice audit assertion. Full suite 169; ruff clean.
