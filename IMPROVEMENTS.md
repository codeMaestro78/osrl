# Correctness, training and dashboard improvements

Reviewed the Python runtime, RL environment/inference/training/scheduler,
forecast and alarm evaluation, SQLite flows, all dashboard pages, existing
tests, architecture notes, and deployment configuration. Local changes only;
existing untracked checkpoints were preserved.

## Fixed issues

| Area | Root cause and resulting behavior |
| --- | --- |
| Checkpoint status | Substring matching classified `UNVALIDATED` as `VALIDATED`. Status now uses the exact suffix, and unvalidated inference cannot queue intervention intents. |
| Training/evaluation leakage | Both environments previously loaded the entire database. Training now freezes one dataset and uses a chronological 80/20 split with a ten-sample embargo and contiguous-session boundaries. |
| Evaluation fairness | Fresh/default normalization and unmatched episodes distorted comparison. Policy and NOOP now replay identical held-out episodes with frozen training statistics and raw rewards. |
| Negative rewards | Multiplying a negative baseline by 1.05 allowed worse policies through. The diagnostic threshold now adds 5% of the baseline's absolute magnitude. |
| Replay validity | Recorded transitions do not depend on actions. Reports explicitly identify this limit, and replay cannot promote checkpoints to live validation. |
| Save/resume | Extensionless broken symlinks, shared overwritten statistics, an unsupported policy keyword, and incorrectly attached resume environments broke the pipeline. Each run now saves unique artifacts and matching statistics/reports, and resumptions attach the restored environment correctly. |
| PyTorch compatibility | Installed PyTorch 2.14.1 failed reading nested unbuffered ZIP streams during SB3 reload, despite valid weights. New artifacts use supported legacy tensor serialization inside the standard SB3 archive; third-party loaders and existing files are unchanged. |
| Provenance | Re-evaluation checks the exact held-out range, count and SHA-256 content fingerprint. Incompatible resume provenance is rejected. |
| Observations | Encoding mutated agent state, the first action received an empty observation, GPU features disappeared during replay, and risk state never advanced. Encoding is now repeatable, reset exposes a real initial window, GPU data survives reconstruction, and risk updates drive masks. |
| Rewards/curriculum | Zero available memory escaped OOM penalties; a swap trend ending at zero was ignored; invalid intervals could fabricate PSI ratios; synthetic stress could reduce real pressure. These cases now preserve their intended meaning. |
| Automatic training | An unread subprocess pipe could block PPO. Launchers write durable logs. Atomic per-database locks prevent duplicate trainers/schedulers, retry state survives restarts, completion/failure is recorded, and snapshot watermarks retain samples arriving during a run. |
| Collector | CPU recovery could diff across an outage, counter resets could produce invalid percentages, and mixed RSS/CPU units could rank unknown newborns above busy processes. Recovery resets history, invalid deltas remain unknown, ranking uses comparable values, and configured top-K is honored. |
| Daemon/config | CLI overrides reset unrelated settings and `--interval` alone was ignored. Overrides now preserve the configuration; omitted environment wiring was added. Buffered samples are flushed before completing rollups. PSI open failures are typed registration failures. |
| Worker/guard/store | Malformed payloads could throw, future/invalid telemetry could pass freshness checks, and lease reconciliation could overwrite a concurrently completed intent. Parsing rejects malformed data, freshness checks fail closed, and reconciliation reads and updates under the same transaction. |
| Alarm evaluation | Prealarm-only runs could make precision negative; endpoint-only overlap tests missed events; state bridged outages. Precision uses alarm runs and interval overlap, and policies/runs reset across outages. False alarms per hour excludes outage time. |
| Dashboard | Ledger queries referenced nonexistent columns and silently displayed empty results. Both ledger views now use real audit fields, statuses and reasons. Native process tables retain full names and missing values. |
| Dashboard integrity | Missing metrics and empty heatmap buckets appeared as zero; missing features were zero-imputed in correlations; idle GPU telemetry appeared absent. Missing data now remains distinguishable. |
| Dashboard operation | Cloud entrypoint imports could skip rendering on rerun, demo mode could disappear after bootstrap, and automatic page navigation linked to blank pages. The entrypoint executes on every rerun, demo state persists, and only the working navigation is displayed. |
| Deployment | The Docker build attempted package installation before copying its declared packages and swallowed failures with an incomplete dependency fallback. Source is copied before installation and dependency failures now fail the build visibly. |

## New features

- **Evaluation & Data Quality** page: PSI coverage, pressure counts, training
  readiness, chronological split counts, risk versus naive alarm policies,
  five forecasting baselines, forecast coverage/MAE/bias, and JSON export.
- Training page focused on the latest run, paired policy/NOOP curves, actual
  step counts, checkpoint reports/downloads, reward components and log tail.
- Unique run directories, `latest` and `candidate` links, reproducible seeds,
  bounded Torch CPU threads, and atomic checkpoint/link publication.
- Responsive telemetry cards, more readable muted copy, keyboard focus styles,
  reduced-motion support, and no repeated entrance animation on live data.
- Inference suggestions are auditable events even when no intent is admitted.

## Validation

- `.venv/bin/python -m pytest -o addopts='' -q`: **252 passed, 1 skipped**.
  The skip requires an external historical hold-test database that is absent.
- New regression tests were first run against the old implementation and
  reproduced eight failures before fixes.
- Real tiny MaskablePPO runs exercise train, save, standard SB3 reload, resume,
  inference and repeated evaluation; no model or trainer stand-ins in that
  integration test. Larger production training was not run.
- Scheduler tests exercise failure cooldown over multiple polls and a restart,
  snapshot watermark accounting, lock contention and file logging.
- Streamlit flow tests render every page with partial telemetry and real
  ledger rows, and generate an evaluation report with no labelled events.
- Browser navigation checked all seven pages without application exceptions.
  Overview checked at desktop, 390px and 320px; mobile document width matched
  viewport width, and reduced-motion mode had zero running animations.
- Focused Ruff checks pass on new/rebuilt modules and new tests; undefined/
  unused-name checks pass across `osrl` and `tests`. `git diff --check` passes.
- Visual checks use a separate synthetic fixture database under `/tmp`, not
  the user's telemetry or checkpoints.

## Evidence still needed

There is no demonstrated increase in freeze-prevention accuracy or live action
benefit. Replay rewards mostly measure conditions already recorded, and all
non-NOOP actions incur a cost without changing those conditions. A useful
control policy needs a calibrated action-conditioned model and representative
held-out pressure/action outcomes, including foreground regressions and
uncertainty. Synthetic stress and a higher training reward cannot supply that
proof. Mechanical pressure labels are proxies for desktop freezes.

The existing worker approval boundary remains in place. This pass does not
start live inference, enable autonomous execution, install services, deploy,
or modify existing checkpoint files. Docker configuration was inspected and
corrected but an image build/cloud deployment was not executed. Visual checks
are focused checks, not a complete accessibility or cross-browser audit.

Library behavior was checked against the installed packages and official
[VecNormalize documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html),
[MaskablePPO documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html),
and [PyTorch serialization documentation](https://docs.pytorch.org/docs/stable/notes/serialization).
