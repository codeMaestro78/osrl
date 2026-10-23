# OS-RL-Agent

**A local-first freeze-risk watchdog for Linux.** It samples kernel pressure
stalls (PSI), memory/swap state, and per-process telemetry every 2 seconds,
classifies freeze risk, and permits only guarded, reversible interventions.
Reinforcement learning is a candidate policy that must beat a measured
baseline and pass an evaluation gate — it is never on the safety path.

`Python 3.10+` · `Linux with PSI` · `253 tests` · `unprivileged by default`

[Architecture](DESIGN.md) · [Quickstart](#quickstart) · [Dashboard](#dashboard) ·
[Deployment](#deployment) · [Safety model](#safety-model)

---

## Why PSI?

CPU load alone does not explain freezes. A machine can sit at 30% CPU and
still be unusable because it is thrashing swap or stalled on memory reclaim.
Linux PSI (Pressure Stall Information) reports the fraction of time tasks
were actually stalled waiting for CPU, memory, or I/O.

OS-RL-Agent treats **occupancy as context and pressure as evidence**. Swap
that is 98% full with zero swap traffic and zero PSI stalls is correctly
classified `none` — not a false alarm. Derived values are `None` when the
kernel cannot provide them; nothing is fabricated as zero.

## Status at a glance

| Layer | What ships | State |
| --- | --- | --- |
| **Telemetry** | 2 s collector: CPU total/per-core, RAM, swap, PSI `some`/`full`, `/proc/vmstat` deltas, GPU via NVML, top-15 processes, birth/death events | ✅ |
| **Storage** | SQLite WAL, schema v3: `samples`, `rollups`, `events`, `meta`, `heartbeat`, `intents`, `actions_ledger`, `rl_metrics` | ✅ |
| **Risk** | `none` / `elevated` / `critical` / `unknown`, multi-signal escalation, hysteresis, staleness gate, PSI evidence requirement | ✅ |
| **Actions** | `renice_back` only, behind `worker → guard → pidfd → executor → ledger` | ✅ gated; `--execute` required |
| **Twin** | Session slicing, `noop`/`hold`/`linear`/`ewma`/`rolling-median` baselines, 30/60/120 s evaluation, renice forward model | ✅ abstains below 8 real pairs |
| **RL** | Gymnasium env, action masking, multi-term reward, MaskablePPO training, inference/intent bridge | ✅ training; **no validated policy** |
| **Dashboard** | Seven-page Streamlit control plane | ✅ local + Cloud demo |
| **Deployment** | systemd units, Docker/supervisord, Fly/Render/Railway configs, Streamlit Cloud demo | ✅ |

**Deliberately not enabled**

- No autonomous enforcement. The daemon never starts the worker; the only
  target-mutation path is the worker CLI with an explicit `--execute` flag.
- No `SIGKILL`, `SIGSTOP`, or termination primitive exists anywhere.
- `nice` lowering (restore) is blocked: unprivileged lowering returns `EPERM`
  both cross-process and for self, so `restore_capable=False`.
- No policy is promoted. Training runs produce local checkpoints, but none has
  passed the evaluation gate (all `UNVALIDATED`), and the inference bridge is
  not started by the daemon.
- `unknown` is never conflated with `elevated`, and missing metrics cast no vote.

## Architecture

```text
 /proc/pressure/*   /proc/vmstat   psutil   NVML
          \             |            |       /
           \            v            v      /
            +------ osrl-collector (2 s) ---+
                            |
                            v
                 SQLite WAL store (schema v3)
                 ├── samples / rollups / events / heartbeat
                 ├── intents ──► osrl.worker ──► guard ──► pidfd
                 │                                        │
                 │                                        v
                 │                                   executor (setpriority)
                 │                                        │
                 │                                        v
                 │                                  actions_ledger
                 └── twin / policy_eval / RL training
                            |
                            v
                  osrl-dashboard (Streamlit)
```

**Structural safety invariants** (enforced by tests in
`tests/test_safety_invariants.py`):

1. `osrl.guard`, `osrl.worker`, `osrl.executor`, `osrl.pidfd`, and
   `osrl.daemon` never import `osrl.rl` or `osrl.dashboard`.
2. `osrl.daemon` never imports or starts the worker.
3. The guard chain never imports `torch`, `gymnasium`, or `stable-baselines3`.
4. The only module permitted to mutate process priority is
   `osrl/executor.py` (via `psutil.Process.nice`, which maps to the
   `setpriority` syscall).

## Requirements

- **Linux with PSI** — `/proc/pressure/{cpu,memory,io}` (kernel 4.20+;
  verified on Ubuntu 24.04, kernel 7.0).
- **Python 3.10+**.
- Optional: NVIDIA GPU + NVML for GPU utilization, VRAM, temperature, power.
- No root required for the shipped `renice_back` path.
- Streamlit Community Cloud is supported for the **recorded demo UI only** —
  it cannot read your host's `/proc` or run the collector.

## Quickstart

```bash
git clone https://github.com/codeMaestro78/osrl.git
cd osrl

python3 -m venv .venv
./.venv/bin/pip install -e ".[dashboard]"      # core + Streamlit UI

# Smoke test (first tick writes NULL for derived rates)
./.venv/bin/python -m osrl.daemon --db data/telemetry.db --once

# Run the collector forever
./.venv/bin/python -m osrl.daemon --db data/telemetry.db
```

In a second terminal:

```bash
./.venv/bin/osrl-dashboard --db data/telemetry.db
# open http://localhost:8501
```

Alternatively, `./.venv/bin/streamlit run streamlit_app.py` uses the same
entrypoint as the Cloud demo and auto-selects a live or recorded database.

For development and tests:

```bash
./.venv/bin/pip install -e ".[dev,dashboard]"
./.venv/bin/python -m pytest -q        # 252 passed, 1 optional fixture skipped
```

For RL training (installs PyTorch, Gymnasium, SB3):

```bash
./.venv/bin/pip install -e ".[dev,dashboard,rl]"
```

## Dashboard

The Streamlit control plane auto-refreshes its fragments; data comes from the
same SQLite store the collector writes.

| Page | Contents |
| --- | --- |
| **Overview** | Risk status, live gauges, PSI stalls, 5-minute charts, event stream |
| **Processes** | Top CPU/RSS, top-10 memory consumers, per-core distribution, process events, action ledger |
| **Data Explorer** | Auto-train status, dataset coverage, feature distributions, correlations, raw sample browser |
| **Agent** | Mode gate (observe/advise/autopilot), checkpoint status, reward history, action ledger |
| **Evaluation** | Data quality, chronological split readiness, risk-policy and forecast comparisons, downloadable evidence |
| **Training** | Per-run reward curves, reward breakdown, checkpoint reports, training launcher, durable log |
| **History** | 15-minute rollups, 7-day trends, hour-of-day CPU heatmap, full event log |

The Cloud demo disables the training launcher and labels itself as a recorded
snapshot. See [Streamlit Community Cloud](#streamlit-community-cloud-recorded-demo).

## Guarded actions

The worker is the **only** intent → action path. Dry mode is the default:

```bash
./.venv/bin/python -m osrl.worker --db data/telemetry.db          # dry peek
./.venv/bin/python -m osrl.worker --db data/telemetry.db --execute
```

Intent lifecycle:

```text
pending → claimed → applied | rejected | failed | expired
```

Ledger lifecycle:

```text
proposed → approved → applied | rejected | failed | restored | expired
```

The guard is deterministic and first-failure-wins: kind whitelist, pid sanity
(never self), `(pid, start_time)` identity, same-uid ownership, telemetry
staleness gate, per-`(pid, kind)` rate limiting (failed attempts count), and a
live intent lease. It never mutates, never retries, and records exactly one
denial reason.

The dashboard currently **does not write or approve intents** — it displays
state and sets the mode flag. `osrl/rl/inference.py` is a library bridge that
can write intents when invoked; nothing starts it automatically.

## Evaluation and RL

### Digital twin

```bash
./.venv/bin/osrl-twin --db data/telemetry.db --horizons 30,60,120
```

Prints a JSON report with per-baseline coverage (`n_pairs`, `n_abstained`),
MAE, and bias. Sessions are cut wherever a sample gap exceeds
`max(3 × median interval, 10 s)`, so no prediction bridges a daemon outage.
The renice forward model requires **8 usable real pairs**; below that it
abstains rather than inventing an effect size.

### Training (optional)

```bash
./.venv/bin/osrl-train --db data/telemetry.db --steps 100000 --synthetic
./.venv/bin/osrl-train --db data/telemetry.db --eval-only
```

- One RL step = one 20 s decision window (10 samples at 2 s).
- Action space: `NOOP`, `ALERT`, `RENICE_BACK`, `REQUEST_APPROVAL`,
  `TUNE_THRESHOLDS`.
- Observation: 114 floats = 57 features + 57 validity-mask bits.
- Reward terms: PSI (memory/IO/CPU), swap-out, swap urgency, memory headroom,
  swap free, foreground share, OOM risk, stability, trend, action cost.
- Training is CPU-only and re-nices itself `+10` so it does not compete with
  the desktop.
- Each run gets its own directory. Every checkpoint has a matching
  `.vecnorm.pkl` and `.json` evaluation report. `latest` points to the newest
  checkpoint; `candidate` points to the best replay score in the latest run.
- Training uses the oldest 80% of a frozen dataset; evaluation uses later
  contiguous data after a ten-sample embargo. Both policies see identical
  held-out episodes and rewards are reported without reward normalization.
- The sign-correct reward diagnostic requires improvement by 5% of the
  absolute NOOP reward. **Replay cannot promote a policy:** actions do not
  change recorded future telemetry. All replay checkpoints remain
  `UNVALIDATED`; `best` is reserved for separately validated policies.
- Evaluation reports fingerprint the held-out rows. `--eval-only` loads the
  checkpoint, its exact normalization statistics and original held-out range,
  and refuses evaluation if that data has changed or expired.
- Older checkpoints lacking matching normalization and provenance are kept,
  but cannot be resumed automatically. Start a fresh run to create a complete
  artifact set.

**Current state:** training runs produce checkpoints locally, but none has
passed the gate (all `UNVALIDATED`); no checkpoint is shipped with the
repository. Inference supports `observe`, `advise`, and `autopilot` gates, but
it is not started by the daemon; even `autopilot` intents still require the
worker, guard, and executor.

### Auto-training scheduler

```bash
./.venv/bin/osrl-auto-train --db data/telemetry.db \
    --interval 3600 --min-samples 500
```

Defaults: poll every 60 s, require 500 new samples and at least 1 hour between
runs, 50k steps per run, resume from a compatible `best` or `latest` checkpoint. Training and scheduler
instances use separate per-database locks. Failed launches retain a durable
cooldown; samples arriving during training count toward the next run using the
snapshot watermark. Training waits when collector telemetry is stale. Child
output is retained in `<checkpoint-dir>/training.log`.

## Configuration

All knobs live in `osrl/config.py` and can be overridden with environment
variables. The most useful:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OSRL_DB` | `~/os-rl-agent/data/telemetry.db` | SQLite database path |
| `OSRL_INTERVAL` | `2.0` | Collector sample period (seconds) |
| `OSRL_RAW_DAYS` | `7` | Raw sample retention |
| `OSRL_EVENTS_DAYS` | `30` | Event retention |
| `OSRL_ROLLUP_SECONDS` | `900` | Rollup window |
| `OSRL_HEARTBEAT_S` | `30` | Daemon heartbeat period |
| `OSRL_BATCH_SIZE` / `OSRL_BATCH_INTERVAL_S` | `10` / `10` | Write batching |
| `OSRL_WAL_CHECKPOINT_S` | `3600` | WAL checkpoint period |
| `OSRL_HEAVY_CPU_PCT` / `OSRL_HEAVY_RSS_MIB` | `50` / `512` | Heavyweight birth thresholds |
| `OSRL_RISK_*` | see `config.py` | Risk thresholds, enter/exit streaks, staleness |
| `OSRL_GUARD_*` | see `config.py` | Guard staleness, rate limit, restore capability |
| `OSRL_RL_*` | see `config.py` | Obs window, training steps, eval gate, synthetic stress |

PSI trigger windows must remain multiples of 2 seconds for unprivileged
monitors. Defaults follow `DESIGN.md` §7 (2 s cadence, 7-day raw retention,
15-minute rollups).

## Deployment

### Local systemd

```bash
./deploy/install-services.sh
```

Installs and starts `osrl-collector`, `osrl-dashboard`, and
`osrl-auto-train` as user services. The auto-train unit needs the `rl` extra;
disable it if you have not installed `.[rl]`.

### Remote access without a cloud host

```bash
./deploy/cloudflare-tunnel.sh
```

Starts a quick Cloudflare tunnel to `http://localhost:8501`. This is the
lowest-latency way to view **your own machine** remotely (2 s freshness).
The quick-tunnel URL changes on restart; use a named tunnel for a stable
domain. The laptop must stay awake and online.

### Docker / cloud VM

```bash
docker build -t osrl .
docker run --rm -p 8501:8501 -v osrl_data:/data osrl
```

The image runs the collector and dashboard together under `supervisord`, with
the database on the `/data` volume. Ready-made configs:

- `fly.toml` — Fly.io, Mumbai region, persistent volume
- `render.yaml` — Render, 5 GB disk
- `railway.json` — Railway, Docker builder

This is genuinely live, but it monitors the **container/VM it runs in**, not
your laptop. On Linux you can add `--pid=host` to see host processes, though
cgroup limits still apply.

### Streamlit Community Cloud (recorded demo)

Streamlit Community Cloud runs only the Streamlit process. It has no cron, no
daemon, no persistent disk, and no access to your laptop's `/proc`. The Cloud
deployment therefore replays a compact recording of real telemetry and labels
itself as a snapshot in the UI.

**What gets deployed**

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | Cloud entrypoint; resolves the DB before the dashboard imports its cached connection |
| `requirements.txt` | Dashboard-only dependencies (no torch / SB3, keeping the Cloud build small) |
| `osrl/dashboard/bootstrap.py` | DB resolution: `OSRL_DB` → local live DB → bundled snapshot → empty v3 store |
| `data/demo_telemetry.db` | ≈7 h / 12k-sample recording, copied to `/tmp` at startup |
| `.streamlit/config.toml` | Dark theme and headless defaults |

**Deploy**

1. Push the repo to GitHub (public, or private with Streamlit's GitHub app
   granted access). `.gitignore` already carries an exception for
   `data/demo_telemetry.db`.
2. Open <https://share.streamlit.io> → **Create app** → **Yup, I have an app**.
3. Repository `codeMaestro78/osrl`, branch `main`, main file path
   `streamlit_app.py`.
4. Python 3.12 is the default; no secrets are required.
5. Deploy. Startup copies the snapshot to `/tmp/osrl-dashboard/demo.db`.

**Regenerate the snapshot** after collecting more telemetry:

```bash
python deploy/build_demo_db.py \
    --src data/telemetry.db \
    --dst data/demo_telemetry.db \
    --hours 24 --max-procs 8
```

The builder uses SQLite's backup API (safe while the collector writes),
preserves schema version/rollups/events/ledger rows, trims old samples,
compacts per-process JSON, and sanitizes `/home/<user>/` paths.

**Cloud limitations (by design)**

- Metrics are from the recording, not the Cloud container. The banner says
  **recorded demo snapshot** and the Overview card reads **Dataset → Recorded**.
- Mode changes, rollups, and intents last only for the container's lifetime.
- Training launches are disabled because the Cloud build omits torch.
- The app sleeps after inactivity; the snapshot does not advance.

**Keeping the Cloud URL updated**

Cron cannot run inside Streamlit Cloud. The scheduler must run on the machine
that can see `/proc`:

| Approach | Freshness | Trade-off |
| --- | --- | --- |
| **Local dashboard + Cloudflare Tunnel** | ~2 s | Laptop must be awake; no Streamlit Cloud |
| **Cron/systemd timer publishes a compact DB** to R2/S3/Supabase/VPS; Cloud app would fetch it | 1–15 min | Needs object storage + a small `OSRL_REMOTE_DB_URL` fetch bridge (not implemented yet) |
| **Remote database** (Postgres/Supabase/Turso) | seconds | Requires a data-access refactor away from the SQLite driver |
| **GitHub Actions cron committing the DB** | minutes | Not recommended: the runner cannot see your laptop, and large DB commits bloat the repo and redeploy constantly |

A `--once` collector run from cron is also a trap: the first tick has no
previous counters, so CPU%, swap rates, and PSI deltas are `None`. Keep the
2-second collector as a service; use cron only for publishing.

## Data model

| Table | Purpose |
| --- | --- |
| `samples` | One row per 2 s tick: CPU, memory, swap, PSI deltas, vmstat rates, GPU, per-process JSON |
| `rollups` | 15-minute windows (stall seconds, averages/maxima); missing windows stay dense with `sample_count=0` |
| `events` | Risk changes, PSI trigger firings, process births/deaths, training metrics, scheduler events |
| `heartbeat` | Single-row durable daemon heartbeat with measured own-CPU% and `risk_level` |
| `meta` | `agent_mode`, rollup cursor, demo-snapshot metadata |
| `intents` | Durable intent channel with atomic single-owner claims and expiry |
| `actions_ledger` | Proposed → approved → applied/failed audit trail with guard checks |
| `rl_metrics` | Reserved for structured RL metrics (training events currently go to `events`) |

Retention: raw samples 7 days, events and ledger rows 30 days, rollups
retained. A reboot or unreadable `/proc` read yields `NULL` and resets the
relevant counter history, so diffs never span outages.

## Measured evidence (2026-10-20, Ubuntu 24.04)

- Collector cadence **2.000 s ± 0.2 ms**; own CPU **0.15–0.25%**; worst single
  tick **80 ms** on ~450 processes. Both metrics stay under the 1% budget.
- Storage: **~4 KB/sample** in this recording (≈170 MB/day at 2 s); WAL
  checkpoints plus 7-day retention bound disk use.
- Digital twin on a 1.78 h / 223-sample bank: `hold` beats `linear` at
  30/60/120 s — CPU MAE **1.66 vs 1.82**, **2.14 vs 2.50**, **2.59 vs 3.45**.
- `rolling-median` wins bursty stress features (PSI stall ratio, swap-out,
  allocstall) by **2–3×**; `hold` wins level features (memory available, swap
  occupancy, CPU total). The Phase 4 bar is the per-feature blend.
- Live renice validation on a 2 s experiment bank: reniced child
  **4.17% → 0.79%**, contender **4.17% → 7.52%** (one-core conservation);
  both reniced → **4.17/4.17**.
- Forward model: **1 usable pair < 8**, so it abstains everywhere; the fit
  summary publishes `n=1` with the caveat rather than an effect size.
- Policy evaluation: hysteresis removed one false-alarm run on the labeled
  scenario at a documented **18 s** latency cost.

## Safety model

- **Guarded but gated.** The only live target-mutation path is the worker CLI
  in explicit `--execute` mode, consuming intents from the audited channel.
- **RL is structurally outside enforcement.** The guard chain never imports
  RL/dashboard modules (AST test), and the daemon never starts the worker
  (test).
- **Restore is evidence-capped.** Raising `nice` is unprivileged-safe and
  read-back verified; lowering it returns `EPERM` unprivileged (cross-process
  and self), so `restore_capable=False` until an elevated helper is
  deliberately configured.
- **No termination primitives.** No `SIGKILL`, `SIGSTOP`, or kill path exists;
  `renice_back` is monotonic and pidfd-pinned against pid reuse.
- **Observe remains the default.** Every execution in this repo's history
  targeted operator-owned test children, with an auditable intent → ledger row.
- **Honest abstention.** Stale telemetry, missing metrics, insufficient
  evidence, and unvalidated checkpoints produce `unknown`/abstain — never a
  fabricated number.

## Repository layout

```text
osrl/
  collector.py      # 2 s sampler and process events
  store.py          # SQLite schema v3, intents, ledger, rollups
  daemon.py         # single-instance collector service
  risk.py           # deterministic risk classifier + hysteresis
  psi_trigger.py    # kernel PSI monitors (poll/POLLPRI)
  guard.py          # post-action veto, first failure wins
  worker.py         # intent -> action; dry by default
  executor.py       # the only priority-mutation surface
  pidfd.py          # pid pinning against reuse
  twin.py           # session slicing, baselines, forward model
  policy_eval.py    # alarm-policy evaluation against frozen labels
  rl/               # env, obs, mask, reward, train, inference, auto_train
  dashboard/        # Streamlit app, pages, shared helpers, bootstrap
deploy/             # systemd units, Docker/supervisord, tunnel, demo builder
tests/              # 252 passed, 1 optional fixture skipped: collector, store, guard, twin, RL, safety
```

## Known limitations

- Linux/PSI only; tuned on a mostly-idle Ubuntu desktop.
- One action primitive (`renice_back`); restore needs elevation.
- No validated policy is promoted, and RL inference is not continuously run.
- The dashboard does not write/approve intents yet; the worker CLI does.
- Event scarcity: synthetic stress is a curriculum aid, not evidence.
- Streamlit Cloud is a recorded demo unless a live-data bridge or tunnel is
  added.

**Next steps:** policy validation + human approval UI, a remote live-data
bridge for Cloud, an optional elevated restore helper, and a broader (still
guarded) action set.

## Implementation records

`DESIGN.md` carries the phase-by-phase engineering record and research audit:

- §11 — research-audit amendments
- §12 — Phase 1: risk estimator and the PSI-trigger kernel contract
- §13 — Phase 2: guard, pidfd, the restore-capability finding, schema v2
- §14 — Phase 3: twin measurements and forward-model abstention
- §15 — Phase 4: policy evaluation, evidence pipeline, live validation

[IMPROVEMENTS.md](IMPROVEMENTS.md) records the correctness pass, validation, and remaining limits.

The Git history is also organized by phase. The later RL stack, dashboard,
and deployment work is documented in this README and the commit history.

## References

- [`DESIGN.md`](DESIGN.md) — architecture, autonomy ladder, research audit
  (§11), twin measurements (§14.4), policy evaluation (§15.1).
- [Linux PSI documentation](https://docs.kernel.org/accounting/psi.html)
- [Streamlit Community Cloud docs](https://docs.streamlit.io/deploy/streamlit-community-cloud)
