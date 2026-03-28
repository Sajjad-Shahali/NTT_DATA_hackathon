# Heuristic Handoff: Classical Meanings vs Implemented Code

Prepared on: 2026-03-17

Scope:
- Active codebase only: `D:\UROP\Dynamic-Job-Shop-Scheduling-RL\`
- Main files reviewed:
  - `experiments/evaluate_variation.py`
  - `env_company.py`
  - `experiments/composite_score.py`
  - `experiments/configs/base_config.yaml`
  - `reports/project_presentation_v2.md`

This note exists because the baseline names in the project are familiar scheduling names, but the actual implementations are simplified online dispatch rules adapted to this environment. The difference matters for reporting, supervision, benchmarking, and future refactoring.

## 1. Executive Summary

Short answer:

- Yes, the current code is different from the classical meanings in several cases.
- The project does not implement full classical DJSSP/FJSSP heuristics exactly.
- It implements deterministic priority rules over the current set of dispatchable jobs.
- The heuristic chooses only the job.
- The environment, not the heuristic, chooses the machine.
- Because of that, names such as `SPT`, `LPT`, `FIFO`, and `LIFO` should be treated as heuristic-inspired labels, not exact textbook implementations.

The most important differences are:

1. The rule is applied only to jobs that are dispatchable now.
2. The action is "choose a job", not "choose an operation-machine pair".
3. Machine selection is fixed by the environment as "first eligible idle machine".
4. `SPT` and `LPT` are based on total remaining job work, not current operation processing time.
5. `FIFO` and `LIFO` are based on original job arrival time into the system, not ready-queue entry time for the current operation.

## 2. Which Code Path Is the Active One

This project contains two parallel codebases:

- Root-level legacy files under `D:\UROP\`
- Active production code under `D:\UROP\Dynamic-Job-Shop-Scheduling-RL\`

For this handoff, the relevant implementation is the active production path:

- Heuristic definitions are in `experiments/evaluate_variation.py`
- Environment behavior is in `env_company.py`

This distinction is important. Some older files in the root directory use similar names but represent older logic and should not be used to describe the current experiment pipeline.

## 3. How the Heuristics Actually Work in This Project

The implemented baseline process is:

1. Build the set of dispatchable jobs.
2. Compute a scalar priority key for each dispatchable job.
3. Pick the minimum or maximum key, depending on the rule.
4. Pass that job index into the environment.
5. The environment assigns the chosen job's next operation to the first eligible idle machine.

This means the baselines are job-level online dispatch rules inside a constrained environment.

### 3.1 Dispatchable jobs in the environment

A job is dispatchable only if all of the following are true:

- It has already arrived.
- It is not completed.
- It still has a remaining operation.
- At least one eligible machine for its next operation is idle.

This already changes the classical meaning of some rules, because the rule is not applied to all jobs and not to future jobs, only to the current feasible ready set.

### 3.2 Machine assignment is not part of the heuristic

The heuristic does not choose the machine. The environment assigns the next operation to the first eligible idle machine found in the operation's eligibility list.

That means even when a heuristic name sounds classical, the implemented behavior is not a full flexible job-shop heuristic in the textbook sense. In classical FJSP settings, many heuristics choose both:

- which operation to schedule
- which machine to assign

Your code only chooses the job.

### 3.3 Dynamic setting implication

Because this is a dynamic job shop setting, future arrivals are unknown to the policy at decision time. Therefore:

- any valid online heuristic must operate only on current state information
- global "last to arrive overall" logic is impossible online
- queue-based or release-time-based interpretations are acceptable

This is why your `LIFO` implementation is valid as an online rule, but only in the local sense of "prefer the most recently arrived among currently dispatchable jobs"

## 4. What Each Named Rule Means in Your Code

Below, "implemented meaning" refers to the current code, not the classical literature definition.

### 4.1 EDD

Implemented meaning:
- Pick the dispatchable job with the smallest due date.

Classical meaning:
- Earliest Due Date first.

Assessment:
- This is close to the classical definition.
- The main simplification is that the rule is still only a job-level priority rule.
- It does not choose the machine.

Best description to use:
- "Earliest Due Date dispatch rule over dispatchable jobs"

### 4.2 LDD

Implemented meaning:
- Pick the dispatchable job with the largest due date.

Classical meaning:
- Latest Due Date first.

Assessment:
- This is also close to the literal meaning of the name.
- It is less common as a primary benchmark in the literature, but the implementation itself is straightforward.

Best description to use:
- "Latest Due Date dispatch rule over dispatchable jobs"

### 4.3 FIFO

Implemented meaning:
- Pick the dispatchable job with the smallest original system arrival time.

Classical meaning:
- First-In-First-Out or First-Come-First-Served.
- Usually interpreted as serving the job that entered the relevant queue first.

Difference:
- Your code uses the job's original release time into the system.
- It does not track when the current operation entered a machine queue or ready queue.
- In a multi-operation environment, that is not the same as classical queue-based FIFO.

Assessment:
- Partial match only.
- It is better described as release-time FIFO than classical operation-queue FIFO.

Best description to use:
- "Release-time FIFO over dispatchable jobs"

### 4.4 CR

Implemented meaning:
- Pick the dispatchable job with the smallest critical ratio:
- `(due_date - current_time) / remaining_work`

Classical meaning:
- Critical Ratio rule, usually smallest ratio first.

Difference:
- Your denominator is total remaining work for the job.
- Machine assignment is still external to the rule.

Assessment:
- This is close to the classical idea.
- It is one of the more defensible baselines in your current setup.

Best description to use:
- "Minimum critical ratio dispatch rule"

### 4.5 SPT

Implemented meaning:
- Pick the dispatchable job with the smallest total remaining work across all unfinished operations.

Classical meaning:
- Shortest Processing Time.
- Usually interpreted as the smallest current operation processing time, or sometimes smallest job processing time in a simpler setting.

Difference:
- Your code does not use the next operation duration.
- It uses total remaining work of the entire job.

Assessment:
- This is not classical SPT.
- It is much closer to a least-remaining-work rule.

Best description to use:
- "Least Remaining Work dispatch rule"

Recommended rename if exact terminology matters:
- `SPT` -> `LRW`

### 4.6 SLACK

Implemented meaning:
- Pick the dispatchable job with the smallest slack:
- `due_date - current_time - remaining_work`

Classical meaning:
- Least Slack Time first.

Difference:
- As with `CR`, the machine assignment remains outside the rule.
- Remaining work is total remaining job work, which is still a reasonable interpretation.

Assessment:
- This is close to the classical rule.
- It is one of the most meaningful baselines for a dynamic due-date setting.

Best description to use:
- "Least slack dispatch rule"

### 4.7 LIFO

Implemented meaning:
- Pick the dispatchable job with the largest original system arrival time.

Classical meaning:
- Last-In-First-Out or Last-Come-First-Served.
- Usually interpreted at queue level: most recently queued job first.

Difference:
- Your code uses original release time into the system.
- It does not mean "the job that will arrive last overall" because that would require future knowledge.
- It also does not track ready-queue entry for each operation.

Assessment:
- Partial match only.
- Valid online rule, but not a full classical queue-based LIFO implementation.

Best description to use:
- "Release-time LIFO over dispatchable jobs"

### 4.8 LPT

Implemented meaning:
- Pick the dispatchable job with the largest total remaining work across all unfinished operations.

Classical meaning:
- Longest Processing Time.
- Usually the largest current operation processing time, or largest job processing time in simpler single-stage settings.

Difference:
- Your code does not use current operation duration.
- It uses total remaining job work.

Assessment:
- This is not classical LPT.
- It is much closer to a most-remaining-work rule.

Best description to use:
- "Most Remaining Work dispatch rule"

Recommended rename if exact terminology matters:
- `LPT` -> `MRW`

## 5. Side-by-Side Comparison Table

| Rule name in code | Implemented key | Classical meaning | Exact match? | Better description |
|---|---|---|---|---|
| `EDD` | min due date | Earliest Due Date | Close | EDD over dispatchable jobs |
| `LDD` | max due date | Latest Due Date | Close | LDD over dispatchable jobs |
| `FIFO` | min original arrival time | First queued / first come first served | No | Release-time FIFO |
| `CR` | min `(due - now) / remaining_work` | Critical Ratio | Close | Minimum critical ratio |
| `SPT` | min total remaining job work | Shortest Processing Time | No | Least Remaining Work |
| `SLACK` | min `due - now - remaining_work` | Least Slack Time | Close | Least slack |
| `LIFO` | max original arrival time | Last queued / last come first served | No | Release-time LIFO |
| `LPT` | max total remaining job work | Longest Processing Time | No | Most Remaining Work |

## 6. Why These Differences Exist

The current implementation is shaped by the environment design:

- Single global action: choose a job
- Machine assignment handled inside the environment
- Discrete-time simulation
- Flexible eligibility handled after the job is chosen
- Action masking only exposes jobs that are feasible right now

This design is practical and valid for RL comparison, but it means the heuristics are implemented as lightweight dispatch rules, not as full scheduling procedures.

That is not necessarily a flaw. It just needs to be described accurately.

## 7. Consequences for Interpreting Experimental Results

### 7.1 Internal comparison is still valid

The benchmark remains valid as an internal comparison because:

- all heuristics and RL policies run in the same environment
- they see the same arrivals, due dates, and machine eligibility
- they are evaluated with the same metric function

So the comparison is fair inside this project.

### 7.2 External or literature comparison needs care

You should not claim that the project compares PPO against exact textbook `SPT`, `LPT`, `FIFO`, or `LIFO` unless you add a caveat.

A safer statement is:

"The project compares PPO against eight deterministic heuristic-inspired online dispatch rules adapted to the current `CompanyFJSSPEnv`."

### 7.3 `LIFO` in a dynamic system

`LIFO` does not require knowledge of the last future arrival.

Your implementation only uses known jobs that are dispatchable now. Therefore it is online-valid.

However, it is still not classical queue-level LIFO because it uses original job arrival time rather than operation ready-queue entry time.

### 7.4 `SPT` and `LPT` are the biggest reporting risk

These two names are the most likely to mislead a reviewer because:

- people usually read `SPT` as shortest current processing time
- people usually read `LPT` as longest current processing time

That is not what the code does.

If exact language matters in the final report, these should either be renamed or explicitly footnoted.

## 8. Recommended Wording for the Thesis, Report, or Presentation

Use this wording if you want to stay accurate without renaming the code:

"The baseline set consists of eight deterministic online dispatch rules evaluated in the same environment as the RL agent. At each decision step, the rule is applied only to currently dispatchable jobs. The selected job is then assigned by the environment to the first eligible idle machine. Therefore, the baseline names follow classical scheduling terminology but should be interpreted as environment-adapted dispatch rules rather than exact textbook implementations."

If you want a shorter version:

"Baseline heuristics are implemented as online job-priority rules over the dispatchable set, with machine selection handled by the environment."

## 9. Recommended Renaming Map

If you want the naming to be fully technically consistent, use this mapping:

| Current code label | Recommended exact label |
|---|---|
| `FIFO` | `ReleaseTimeFIFO` |
| `LIFO` | `ReleaseTimeLIFO` |
| `SPT` | `LeastRemainingWork` |
| `LPT` | `MostRemainingWork` |
| `EDD` | `EDD` |
| `LDD` | `LDD` |
| `SLACK` | `LeastSlack` |
| `CR` | `MinCriticalRatio` |

If you prefer to keep the legacy names for continuity, add one explanatory note in the methodology section rather than changing every figure and CSV.

## 10. Additional Project Caveats Found During This Review

These are not part of the heuristic definitions themselves, but they affect interpretation of results.

### 10.1 Composite score mismatch

The code and config currently disagree on the composite weights:

- `experiments/composite_score.py` uses 60 percent makespan, 15 percent deadlines, 25 percent utilization
- `experiments/configs/base_config.yaml` documents 50 percent makespan, 30 percent deadlines, 20 percent utilization

This means the reported "best heuristic" depends partly on which source the reader trusts.

### 10.2 Composite score omits total tardiness

The formal objective in the project write-up includes:

- makespan
- total tardiness
- idle capacity or utilization

But the composite score does not include total tardiness directly.

This is important because some rules can look good on makespan and utilization while being poor on lateness performance.

### 10.3 Current experiment default is toy data

The experiment configuration reviewed here defaults to the toy dataset rather than the real-company dataset.

That does not invalidate the heuristic definitions, but it does matter when discussing whether a baseline is representative of the real production setting.

## 11. If Classical Implementations Are Desired Later

If the goal later is to implement baselines that match the literature more closely, the following changes would be needed.

### 11.1 For classical FIFO and LIFO

Track ready-queue entry time for each operation, not just original job release time.

### 11.2 For classical SPT and LPT

Use the processing time of the next operation, not total remaining job work.

### 11.3 For stronger FJSP baselines

Allow the heuristic to choose:

- the operation or job
- the machine assignment

Possible classical-style flexible job-shop variants could rank candidates by:

- earliest completion time
- earliest finish time
- shortest processing time on selected machine
- least slack with machine-aware completion estimate

### 11.4 For cleaner DJSSP semantics

An event-driven simulation with explicit machine queues would make classical queue-based rules easier to define and defend.

## 15. Implementation Changes (March 18, 2026) — Academic Publication Preparation

These changes were implemented based on joint Claude + Codex analysis across two sessions (March 17–18, 2026).
All three topics (data generation, early stopping, training pipeline) were addressed and bug-audited.

### 15.1 Files Changed

| File | Change |
|---|---|
| `generate_toy_company_csv.py` | Full rewrite — multi-seed, multi-size benchmark bank |
| `experiments/run_experiment.py` | Fixed step_log bug, added checkpoints, LR decay, early stopping integration |
| `experiments/early_stopping_callback.py` | **New file** — academic-grade early stopping |
| `experiments/configs/base_config.yaml` | Added `training:`, `clip_range`, fixed `log_every_n_steps` |

### 15.2 Topic 1 — generate_toy_company_csv.py

**What changed:**
- Added `generate_benchmark_bank()` function
- Sizes: n=20, 50, 100 (main); n=500, 1000 (stress test, `include_stress=False` by default)
- **5 seeds per size (0..4)** — constant `N_SEEDS = 5` (earlier version of this doc said 10; that was incorrect)
- Proportional arrival scaling: arrival window scales with n_jobs to preserve load factor ≈ 0.70
- Per-instance `meta.json` with full reproducibility metadata
- `bank_manifest.json` at root of `data/benchmark/`
- `load_benchmark_instance(n_jobs, seed)` and `load_benchmark_size(n_jobs)` helpers
- Legacy `main()` still writes `data/processed/toy_*.csv` for backward compat

**Stress test note (must appear in paper):**
Bootstrap diversity is limited by ~20 real route templates. At n=1000, each template is reused ~50×. Label stress test results as "synthetic stress benchmark", not "realistic industrial distribution".

**Statistical protocol for paper:**
- Wilcoxon signed-rank for pairwise comparisons (paired per instance seed)
- Friedman + Holm/Nemenyi for multi-method comparison
- Cliff's delta or Vargha-Delaney A12 as effect sizes
- Report mean ± std and 95% CI, never just mean

### 15.3 Topic 2 — Early Stopping (`experiments/early_stopping_callback.py`)

**`EarlyStoppingCallback` design:**
- Metric: composite score on **separate fixed-seed eval env** (not training env signal)
- `eval_freq = 100_000` timesteps
- `patience = 10` evaluations = 1M timestep tolerance window
- `burn_in_steps = 1_000_000` — no stop decisions before this
- `min_delta = 0.005` relative improvement required
- Best model saved to `models/best_model.zip` on every improvement
- Entropy collapse guard: accelerates patience counter when entropy < 0.01 AND plateaued — does NOT bypass patience (corrected from initial version)
- Evaluation results logged to `logs/eval_history.csv` per eval checkpoint
- Running min/max makespan tracked across ALL evaluations; updated once per batch before scoring (order-independent normalisation — corrected from initial version)
- `no_improve_count` written to eval_history.csv AFTER improvement check (stale value bug corrected from initial version)

**Why NOT raw episode reward:**
`step_time_penalty` is UNSCALED → episode reward magnitudes differ across variations. Composite score is reward-weight-agnostic.

**Smoke test behavior:** Early stopping is disabled automatically when `total_timesteps < 200_000`.

### 15.4 Topic 3 — Training Pipeline Fixes

**Critical bug fixed (Codex finding):**
`info["step_log"]` was never populated by `env_company.py` — the key does not exist in the env's `info` dict. `step_logs.csv` was therefore always empty. Fixed: `RewardLoggerCallback` now reads directly from `info["reward_terms"]`, `info["busy_frac"]`, `info["idle_frac"]`, `info["invalid_action"]`, `info["num_late_unfinished"]`, `info["jobs_completed"]`, `info["current_time"]`.

**Other fixes:**
- `CheckpointCallback`: saves model every `checkpoint_freq` (default 500K) steps to `models/checkpoint_<N>k.zip`
- Linear LR decay: `learning_rate` decays to `initial_lr * final_lr_factor` (default 0.1×) — controlled by `training.lr_decay` in config
- `clip_range: 0.2` explicitly declared in base config under `ppo:` block
- `log_every_n_steps` raised from 1 to 100 — was generating enormous step_logs.csv files
- `n_envs` still supports `null` (auto) but hard-coding recommended for reproducibility
- `training_summary.json` now includes `stopped_early`, `best_eval_composite`

**episode_stats.csv now includes:**
makespan, total_tardiness, utilization, deadlines_met, frac_deadlines_met, total_jobs, unscheduled_jobs, total_reward, length

**step_logs.csv now includes:**
busy_frac, idle_frac, invalid_action, num_late_unfinished, jobs_completed, current_time, full reward decomposition (rw_job_complete, rw_job_on_time, rw_job_late, rw_machine_idle, rw_utilization_bonus, rw_step_time_penalty, rw_invalid_action, rw_final_unscheduled, rw_total)

### 15.5 Post-Audit Bug Fixes (Codex resume session, March 18, 2026)

A second Codex audit pass (`gpt-5.3-codex`, `high` reasoning) identified 3 additional bugs in the initial implementation and confirmed 1 item as valid. All 3 bugs were fixed and re-smoke-tested.

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `generate_toy_company_csv.py:116-117` | Additive `randint(0,3)` noise (per-op) was folded into the multiplicative jitter factor, overestimating workload and making `arrival_span` too large → instances underloaded | Separated: `avg_work = total_work * avg_mult + avg_ops * 1.5` |
| 2 | `early_stopping_callback.py` | Running min/max makespan updated INSIDE the eval episode loop → composite scores within the same batch were order-dependent (first episode always scored 0 on makespan) | Collect all episode metrics first, update min/max ONCE, then score all |
| 3 | `early_stopping_callback.py` | `no_improve_count` written to result dict BEFORE improvement check → stale placeholder value in `eval_history.csv` | Moved `_log_result()` to after `_check_improvement()` |
| ✓ | `generate_toy_company_csv.py` | `rng_np.choice(slack_real)` — confirmed VALID for `np.random.default_rng` Generator objects | No fix needed |

### 15.6 Remaining Issues NOT Yet Fixed

| Issue | Status |
|---|---|
| Config override bug in v11/v12 (Codex finding: top-level YAML keys not read by trainer) | **Not fixed** — requires auditing each v11/v12 YAML and re-running those variations |
| `variation_name: "base"` in some old training_summary.json | Old artifact issue — all new runs write correct variation name |
| Composite weight mismatch in old run artifacts (50/30/20 vs 60/25/15) | Old artifacts unaffected; new runs use base_config.yaml values (60/25/15) |
| Multi-seed training (3-5 seeds per variation) | Not yet done — requires running each variation 3–5 times with different train seeds |

## 16. Run Commands — Full Pipeline (March 18, 2026)

All commands assume working directory `D:\UROP\Dynamic-Job-Shop-Scheduling-RL`.

### 16.1 Smoke Test (verify pipeline works, ~30 seconds)

```
python experiments/run_experiment.py --config experiments/configs/v1_ppo_default.yaml --timesteps 1000
```

### 16.2 Step 1 — Generate Datasets

```
python generate_toy_company_csv.py
```

Writes:
- `data/processed/toy_*.csv` — legacy toy dataset (n=20, seed=42), used as training data
- `data/benchmark/n20/seed_00/ ... seed_09/` — 10-seed benchmark instances
- `data/benchmark/n50/` and `data/benchmark/n100/` — scalability instances
- `data/benchmark/bank_manifest.json` — full reproducibility manifest

### 16.3 Step 2 — Train All 13 Variations

Run each line using the new Python environment. Order is best-first by expected quality based on prior smoke test results:

```
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v8_critical_path_extreme.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v9_high_gamma_makespan.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v12_composite_tight.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v7_makespan_util_focus.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v11_utilization_shaped.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v10_wide_network_makespan.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v13_tassel_reward.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v4_ppo_composite.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v3_ppo_deadline_focus.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v2_ppo_makespan_focus.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v1_ppo_default.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v5_ppo_large_network.yaml
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v6_ppo_entropy_boost.yaml
```

Each run creates `experiments/YYYYMMDD_HHMMSS_<variation>/` containing:
- `models/model.zip` — final weights
- `models/best_model.zip` — best checkpoint by eval composite score (early stopping)
- `models/checkpoint_500k.zip`, `checkpoint_1000k.zip`, ... — periodic checkpoints
- `logs/episode_stats.csv` — per-episode scheduling metrics
- `logs/step_logs.csv` — per-step reward decomposition + env diagnostics
- `logs/eval_history.csv` — early stopping evaluation history
- `logs/tb/` — TensorBoard logs
- `plots/learning_curve.png`
- `metrics/training_summary.json`
- `config/run_config.yaml` — exact config used (for reproducibility)

### 16.4 Step 3 — Evaluate Each Variation

Run after each training completes. Replace `<EXPERIMENT_FOLDER>` with the timestamped folder name printed at end of training:

```
python experiments/evaluate_variation.py --experiment experiments/<EXPERIMENT_FOLDER>
```

Example:
```
python experiments/evaluate_variation.py --experiment experiments/20260318_010000_v1_ppo_default
```

Writes into the experiment folder:
- `metrics/metrics.csv` — 1 RL row + 8 heuristic rows, all metrics + composite score
- `metrics/summary.json`
- `plots/bar_*.png`, `plots/scatter_*.png`, `plots/ratio_vs_rl.png`, `plots/rank_heatmap.png`, `plots/composite_scores.png`
- `gantt/<policy>_overview.png`, `gantt/<policy>_detailed.png`

### 16.5 Step 4 — Compare All Variations

Run after ALL evaluations are complete:

```
python experiments/compare_all.py
```

Or with explicit root:

```
python experiments/compare_all.py --root experiments/
```

Writes `experiments/compare_all_models/`:
- `all_metrics.csv` — all policies merged
- `composite_scores.csv` — with makespan_norm, composite_score, composite_rank
- `summary_table.csv` — sorted by composite score (best first)
- `plots/bar_makespan.png`, `bar_total_tardiness.png`, `bar_utilization.png`, `bar_frac_deadlines_met.png`
- `plots/scatter_makespan_vs_tardiness.png`, `ratio_vs_best_heuristic.png`, `rank_heatmap.png`, `composite_scores.png`

### 16.6 Output File Map Summary

| File | Purpose |
|---|---|
| `logs/episode_stats.csv` | Main training metric log — makespan, tardiness, utilization, deadlines per episode |
| `logs/step_logs.csv` | Dense diagnostic — reward terms, busy/idle frac, invalid actions per step |
| `logs/eval_history.csv` | Early stopping curve — composite score per 100K steps (starts after 1M burn-in) |
| `models/best_model.zip` | Best policy weights by eval composite score — use for final evaluation |
| `models/model.zip` | Final policy weights at training end (may be worse than best_model) |
| `metrics/metrics.csv` | Post-training evaluation: RL vs all 8 heuristics |
| `compare_all_models/summary_table.csv` | Cross-variation ranking — headline result for paper |

---

## 14. Environment Bug Fixes (March 17, 2026)

Seven correctness bugs were identified in `env_company.py` through independent analysis + Codex agent cross-check and fixed in the same session. All are now resolved.

| # | Bug | Severity | Fix |
|---|-----|----------|-----|
| 1 | **Double-dispatch**: `_get_dispatchable_jobs()` did not check if a job's op was already running → same op placed on two machines simultaneously, `job_next_op` double-incremented, op skipped | **Critical** | Added `in_progress_jobs` set; skip job if any machine is running it |
| 2 | **`_gantt_log` not reset**: `_reset_state()` did not clear the Gantt log → log accumulated across all episodes, charts showed mixed schedules | **High** | Added `self._gantt_log = []` to `_reset_state()` |
| 3 | **Multi-job completion reward overwrite**: `for j in finished_jobs` loop overwrote `completed_job_id` → only last job in the batch got its reward | **High** | Reward now accumulated inside the loop with separate log variables |
| 4 | **Float time drift**: completion threshold `1e-9` too tight for IEEE-754 accumulated subtraction → ops could run one step too long | **Medium** | Widened to `1e-6` |
| 5 | **`step_time_penalty` logged wrong**: logged as `dt_norm * penalty` (≈ 0.001) but applied as flat `penalty` (= −10.0) | **Medium** | Log entry now uses unscaled value matching actual reward |
| 6 | **`_remaining_work` includes in-progress op at full duration**: op k counted even when already partially elapsed → obs and heuristics overstated remaining work | **Medium** | Subtracts elapsed time of currently-running op |
| 7 | **`reward_terms["total"]` set before `final_unscheduled`**: terminal-step total in logs was wrong by the unfinished-job penalty | **Low** | `"total"` assignment moved to after `final_unscheduled` is added |

> **Note on false positive**: Codex also flagged `invalid_action` as never initialized to `False`. This was incorrect — line 464 already contains `invalid_action = False` at the top of `step()`. Not fixed (already correct).

---

## 12. Recommended Immediate Reporting Position

The safest current position is:

- keep the baselines
- keep the experiments
- describe them accurately as environment-adapted online dispatch rules
- explicitly note that `SPT` and `LPT` are remaining-work variants
- explicitly note that `FIFO` and `LIFO` are release-time variants

This preserves the value of the experiments without overstating theoretical equivalence to classical scheduling heuristics.

## 13. Final Bottom Line

The code is valid as an internal benchmarking framework, but the heuristic names are only exact in some cases.

Best one-line summary:

"The project currently uses classical heuristic labels for simplified job-selection rules inside `CompanyFJSSPEnv`; these should be interpreted as adapted online dispatch rules rather than strict textbook implementations."

---

## 17. Extended Session Changes (March 17–18, 2026, continued)

### 17.1 New Python Virtual Environment — gitdjssp

A new environment was created specifically for the RTX 5070 Ti Blackwell GPU (compute capability sm_120):

| Item | Value |
|---|---|
| Path | `D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\` |
| Python | 3.12 |
| PyTorch | 2.10+cu128 (force-reinstalled for sm_120 support) |
| CUDA | 12.8 |
| Key packages | sb3_contrib 2.7.1, numpy <2.0, matplotlib >=3.8, pyyaml, openpyxl |

All new training commands use:
```
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe
```

The old `djssp` environment (Python 3.10, PyTorch 2.6+cu124) is retained as legacy but does not support sm_120 and should not be used for new training runs.

**Why sm_120 required force-reinstall:** PyTorch 2.6+cu124 only compiled CUDA kernels up to sm_90. RTX 5070 Ti is Blackwell architecture (sm_120). Running any GPU operation would raise a CUDA kernel architecture mismatch error. PyTorch 2.7+ with cu128 adds sm_120 support.

**numpy incompatibility after upgrade:** PyTorch 2.10 pull upgraded numpy to 2.3.5, which broke matplotlib 3.7.5. Fixed by pinning `numpy<2.0` and upgrading matplotlib to `>=3.8`.

---

### 17.2 Classical Heuristic Variants Added (_C suffix)

Eight additional heuristics were added in `evaluate_variation.py` alongside the original eight. These use operation-level rather than job-level quantities:

| Rule | Key | Difference from original |
|---|---|---|
| `FIFO_C` | min arrival time | Same as FIFO — retained for symmetry |
| `LIFO_C` | max arrival time | Same as LIFO — retained for symmetry |
| `SPT_C` | min **next operation duration** | True classical SPT definition |
| `LPT_C` | max **next operation duration** | True classical LPT definition |
| `EDD_C` | min due date | Same as EDD — retained for symmetry |
| `LDD_C` | max due date | Same as LDD — retained for symmetry |
| `SLACK_C` | min `due - now - next_op_duration` | Slack using next-op only, not total remaining work |
| `CR_C` | min `(due - now) / next_op_duration` | CR using next-op duration as denominator |

**Key difference:** The `_C` variants use only the next operation's processing time (classical textbook interpretation) whereas the original variants use total remaining work across all remaining operations.

**SPT_C vs SPT:** `SPT_C` is the standard literature SPT. `SPT` in the original set is more accurately "Least Remaining Work". For paper reporting, `SPT_C` is the defensible classical baseline.

**Notes on redundancy:**
- `EDD_C` is identical to `EDD` (both use job due date)
- `LDD_C` is identical to `LDD`
- `FIFO_C` is identical to `FIFO`
- `LIFO_C` is identical to `LIFO`
- These are kept for config-list symmetry but can be excluded from the paper table with a footnote

**Config control:** Both sets are controlled by explicit lists in `base_config.yaml`:
```yaml
eval:
  heuristics_original:
    - "FIFO"
    - "LIFO"
    - "SPT"
    - "LPT"
    - "EDD"
    - "LDD"
    - "SLACK"
    - "CR"
  heuristics_classical:
    - "FIFO_C"
    - "LIFO_C"
    - "SPT_C"
    - "LPT_C"
    - "EDD_C"
    - "LDD_C"
    - "SLACK_C"
    - "CR_C"
```

Delete any line from either list to exclude that rule from evaluation outputs.

---

### 17.3 v13 Tassel Reward Variation

Added `experiments/configs/v13_tassel_reward.yaml` — implements the scheduled-area reward from Tassel (2021) and used in the base paper Alcamo et al. (2024):

```
R(s,a) = p_aj − Σ idle_time_all_machines
```

Approximated in the existing env reward structure as:
- `utilization_bonus = 100.0` (per-step reward for busy fraction)
- `machine_idle = −100.0` (per-step penalty for idle fraction)
- All deadline-related terms zeroed (job_complete, job_on_time, job_late, tardiness_step, late_unfinished_step all = 0)
- `final_unscheduled = −500.0` retained to prevent trivially leaving all jobs unscheduled
- `invalid_action = −2.0` retained

**Purpose:** Direct comparison against the base paper's reward function. Expected to learn makespan-minimizing behavior without deadline awareness.

---

### 17.4 run_scaling_eval.py — Overnight Multi-Seed / Multi-Size Evaluation

New script: `experiments/run_scaling_eval.py`

**Purpose:** After training 13 models on the n=20 training instance, evaluate each trained model on all benchmark instances (3 sizes × 5 seeds = 15 instances per model).

**Key design decisions:**
- Reads `scaling_eval` block from `experiments/configs/base_config.yaml`
- Loads pre-generated benchmark instances from `data/benchmark/` using `load_benchmark_instance()` — does NOT regenerate datasets at runtime
- Resume support: skips any (model, size, seed) triple that already has `metrics.csv` on disk
- Evaluates both RL models and heuristics on each instance; saves per-instance results
- Aggregates across seeds per size, then plots scaling curves

**Output structure:**
```
experiments/scaling_results/
    <variation_name>/
        n20/seed_00/metrics.csv
        n20/seed_01/metrics.csv
        ...
        n100/seed_04/metrics.csv
        n20_aggregate.csv
        n50_aggregate.csv
        n100_aggregate.csv
    scaling_curves.png
    summary_table.csv
```

**Config keys used:**
```yaml
scaling_eval:
  dataset_sizes:    [20, 40, 80]   # sizes to evaluate (must exist in benchmark bank)
  n_seeds:          5              # seeds per size (must match N_SEEDS in generator = 5)
  seed_start:       0
  results_dir:      "experiments/scaling_results"
  models_root:      "experiments"
  bank_dir:         "data/benchmark"
  eval_time_step:   10.0
  n_eval_episodes:  5
  rl_device:        "cpu"          # use cpu for eval to keep GPU free for training
  models_to_eval:   []             # fill with experiment folder substrings after training
```

**Important size mismatch note:** `generate_toy_company_csv.py` uses `BENCHMARK_SIZES = [20, 50, 100]` but `scaling_eval.dataset_sizes` in the config is `[20, 40, 80]`. These must match. Either:
- Change `dataset_sizes` in config to `[20, 50, 100]` to match what was generated, or
- Regenerate the benchmark bank with the alternate sizes

The benchmark bank is generated once and fixed. Do not change sizes after generation without regenerating.

---

### 17.5 Dataset Generation Biases (Critical for Paper)

Joint analysis (Claude + Codex) found systematic biases in `generate_toy_company_csv.py` that affect the realism of benchmark instances:

| Bias | Description | Impact |
|---|---|---|
| **Uniform IID arrivals** | `arr = randint(0, arrival_span)` — flat distribution, not Poisson | Underestimates burstiness; real shop has correlated job batches |
| **Slack independent of work** | `slack = bootstrapped_from_real * jitter` — no correlation with job's own work content | Makes tight-deadline detection harder; real deadlines are set relative to work |
| **Limited route diversity** | Only ~20 real templates; at n=100, each reused ~5×; at n=1000, ~50× | Benchmark does not represent real variety beyond n~20 |
| **Narrow jitter range** | Jitter in [0.80, 1.25] → max duration spread ≈ 1.56× | Real shops have higher variance; agent may overfit to narrow distribution |
| **Proportional arrival scaling** | `arrival_span = total_work / (n_machines * 0.70)` — exact target utilization | Deterministic utilization; real shops have random load variation |

**What to write in paper:** "Benchmark instances are generated by bootstrapping from 20 real job templates with multiplicative jitter (0.80–1.25×). Arrival times are drawn IID uniform over a proportionally scaled window targeting ~70% utilization. Slack is bootstrapped from the real distribution independently of job work content. These simplifications limit external validity for high-burstiness or tightly-coupled deadline settings."

**These biases do not invalidate the comparison (all methods see the same instances) but should be disclosed in the methodology section.**

---

### 17.6 Config Fixes Applied

| Fix | Old value | New value | Why |
|---|---|---|---|
| `ppo.device` | `"gpu"` | `"cuda"` | SB3/MaskablePPO requires `"cuda"` not `"gpu"`; `"gpu"` silently falls back to CPU |
| `scaling_eval.n_seeds` | `10` | `5` | Must match `N_SEEDS = 5` in `generate_toy_company_csv.py`; benchmark bank only has seeds 0–4 |

---

### 17.7 Scatter Plots Added to evaluate_variation.py

Three scatter plots are now saved per evaluation run to `plots/`:

| File | X-axis | Y-axis | Better is... |
|---|---|---|---|
| `scatter_makespan_vs_tardiness.png` | makespan | total tardiness | lower-left |
| `scatter_makespan_vs_utilization.png` | makespan | utilization | lower-right (low makespan, high util) |
| `scatter_makespan_vs_deadlines.png` | makespan | frac_deadlines_met | lower-right (low makespan, high deadlines) |

These replaced a single makespan vs tardiness scatter. The three plots together give a Pareto-front view of all policies.

---

### 17.8 Base Paper — Alcamo et al. (2024)

File: `D:\UROP\data\A+reinforcement+learning+algorithm+for+Dynamic+Job+Shop+Scheduling.pdf`

Full citation: Alcamo, Bruno, Giovenali — Politecnico di Torino. Reinforcement Learning for Dynamic Job Shop Scheduling.

| Aspect | Paper | This project |
|---|---|---|
| RL algorithm | MaskablePPO | MaskablePPO |
| Action space | 3 meta-strategy actions (SPT / EDD / RANDOM) | Direct job selection (n_jobs actions) |
| State representation | Binary matrix (job × machine eligibility + timing) | Continuous obs vector per job |
| Reward | Scheduled area: op_duration − Σ idle_time | Composite reward (v1–v12), Tassel variant (v13) |
| Heuristic baselines | 3 (FIFO, SPT, EDD) | 16 (8 original + 8 classical variants) |
| Dataset | Synthetic 5 machines, 5 jobs | Real company data, 7 machines, 20 jobs |

v13 was specifically created to replicate the paper's scheduled-area reward, enabling a direct comparison of reward formulations.

---

### 17.9 Training / Scaling Eval Workflow Summary

**13 training runs total** — each trained once on n=20, seed=0 (the legacy toy dataset):

```
v8 → v9 → v12 → v7 → v11 → v10 → v13 → v4 → v3 → v2 → v1 → v5 → v6
(best expected quality first)
```

**After all training completes**, populate `models_to_eval` in `base_config.yaml`:
```yaml
models_to_eval:
  - "v8_critical_path_extreme"
  - "v9_high_gamma_makespan"
  - "v12_composite_tight"
  - ...  # all 13 variation substrings
```

**Then run scaling evaluation** (evaluates each model on 15 benchmark instances = 195 total eval runs):
```
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/run_scaling_eval.py
```

**Final comparison** (single experiment for training-set metrics):
```
D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe experiments/compare_all.py
```

---

## 18. Dataset Generator — Full Academic Rewrite (March 18–19, 2026)

Three sessions of joint Claude + Codex independent analysis identified systematic realism problems in `generate_toy_company_csv.py`, assessed academic value, implemented a full rewrite (v3), and performed three rounds of correctness audits. All issues are now resolved.

### 18.1 Realism Problems Found (v1/v2 → fixed in v3)

Joint analysis (Claude + Codex, independent parallel runs) identified the following systematic biases in the previous generator:

| Bias | Severity | Root cause | Fix applied |
|---|---|---|---|
| IID uniform arrivals | High | `randint(0, arrival_span)` — no clustering | Replaced with compound Poisson (exponential inter-arrivals) |
| Slack independent of job work | High | Bootstrap from real slack, ignores job content | DDT/RDD formula: `slack = U(DDT±RDD) × W_j` where `W_j = total work` |
| Only ~16 real route templates | High | Pure bootstrap reuse; n=100 → 6× reuse per template | Sub-route stitching with Jaccard novelty filter |
| Uniform duration jitter [0.80,1.25] | Medium | Symmetric, bounded; real mfg is right-skewed | Log-normal with CoV parameter (default 0.25) |
| Fixed deterministic arrival span | Medium | `arrival_span = total_work / (n_machines × 0.90)` same for all seeds | Per-seed random utilization target U(0.75, 0.95) |
| `main()` overrode slack with (0.25,0.65) | Medium | Hidden override created different distribution from defaults | Removed; bank uses default parameters throughout |
| `_meta["n_real_templates"]` stored job count | Low | Stored `n_real` (total jobs) not unique template count | Fixed to store unique route fingerprint count |
| Mixed RNG (`random.Random` + `np.random`) | Low | Reproducibility fragility | Unified to single `np.random.default_rng` throughout |
| Docstring said "10 seeds", "load ~0.70" | Low | Documentation stale after code changes | Corrected to N_SEEDS=5, target util 0.75–0.95 |

**Previous paper disclosure text (17.5 above) is now outdated — the generator was fully rewritten to fix all listed biases.**

### 18.2 Academic Value Assessment

The dataset has **conditional academic value**:

| Venue | Realistic? | Condition |
|---|---|---|
| Master's thesis / workshop paper | ✅ Strong | Current design is sufficient |
| OR conference (EURO, INFORMS) | ✅ Adequate | Frame as "industry case study" + disclose template reuse |
| Top-tier ML venue (NeurIPS, AAAI) | ⚠️ Insufficient alone | Must add standard FJSP benchmarks (Brandimarte etc.) |
| OR journal (EJOR, OMEGA) | ⚠️ Marginal | Real data helps; route diversity gap must be disclosed |

**Critical remaining limitation**: ~16 unique route templates. At n≥50, structural route variety collapses. All stitched routes still use ops from the real data pool. Label n≥50 results as "synthetic stress benchmark" in the paper.

### 18.3 Files Changed — v3 Rewrite

#### `generate_toy_company_csv.py` — full rewrite

New public API:

| Function | Description |
|---|---|
| `build_toy_dataset_from_real(...)` | Core generator — updated params below |
| `compute_instance_stats(dataset)` | Returns FI, load, tightness, unique_route_ratio for paper table |
| `generate_benchmark_bank(...)` | Single difficulty setting, multi-seed multi-size bank |
| `generate_difficulty_sweep(...)` | **NEW** — sweeps ddt × rho × batch_mean for academic sensitivity analysis |
| `load_benchmark_instance(n_jobs, seed)` | Unchanged |
| `load_benchmark_size(n_jobs)` | Unchanged |

New parameters for `build_toy_dataset_from_real`:

| Parameter | Default | Meaning |
|---|---|---|
| `duration_cov` | `0.25` | Log-normal CoV for operation durations. Replaces `duration_jitter`. |
| `ddt` | `1.5` | Due-date tightness factor. Slack = U(DDT±RDD) × W_j. Replaces `slack_tightness`. |
| `rdd` | `0.2` | Due-date range (±20% around DDT). Clamped to < 1.0. |
| `target_util` | `None` | Target utilization. None → per-seed random U(0.75,0.95). |
| `batch_mean` | `1.0` | Mean batch size. 1.0 = simple Poisson; 3.0 = compound Poisson batches. |
| `max_batch` | `6` | Maximum batch size cap. |
| `route_mode` | `"mixed"` | `"real_only"` / `"stitched"` / `"mixed"` |
| `stitch_prob` | `0.5` | Stitching probability when mode="mixed". |
| `novelty_threshold` | `0.25` | Minimum Jaccard distance for route novelty filter. |

**Breaking API change**: `duration_jitter` and `slack_tightness` / `slack_jitter` parameters removed. Any caller using old keyword arguments will get `TypeError`. Update call sites.

**`_meta` dict now contains:**
`schema_version`, `n_jobs`, `seed`, `arrival_span`, `realized_util`, `ddt`, `rdd`, `duration_cov`, `batch_mean`, `route_mode`, `stitch_prob`, `n_real_jobs`, `n_unique_templates`, `stats` (from `compute_instance_stats`)

#### `preprocess_company_real.py` — new functions appended

| Function | Description |
|---|---|
| `parse_fjs_instance(path)` | Parse standard `.fjs` format (Brandimarte/Hurink/Kacem). Returns project-schema dict with `op["durations"]` (per-machine times) and `op["duration"]` (min, backward compat). |
| `export_dataset_to_fjs(dataset, path)` | Export to `.fjs` + sidecar JSON. Sidecar contains arrivals, due dates, generation params. |

**Schema extension**: ops loaded from `.fjs` files include `"durations": [p1, p2, ...]` (per-machine processing times). Company-data ops only have `"duration"` (single value). `env_company.py` does NOT yet use per-machine durations — update it to use `op["durations"][assigned_machine_idx]` for full FJSP fidelity with standard benchmarks.

**Also cleaned**: removed duplicate `import pandas as pd`, unused `import os`, moved `import json` to top level.

### 18.4 Bug Fixes — Three Audit Rounds

**Round 1** (initial v3 implementation, Codex verification):

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `generate_toy_company_csv.py` | Infinite loop if `max_batch ≤ 0` in compound Poisson | `max_batch = max(1, max_batch)` guard |
| 2 | `preprocess_company_real.py` | Sidecar path used `with_suffix` chain → truncated dotted filenames | Changed to `Path(str(path) + ".sidecar.json")` |
| 3 | `preprocess_company_real.py` | `.fjs` export: `zip(eligible, durations)` silently truncated if lengths differ | Added length check; falls back to `[op["duration"]] * len(eligible)` |

**Round 2** (second full audit, Claude + Codex independent):

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `preprocess_company_real.py` | `json.dump` crashes on `float("inf")` in sidecar (static FJSP: `arrival_span=0` → `load=inf`) | `_json_safe()` sanitizer converts inf/nan → None |
| 2 | `preprocess_company_real.py` | `min(durations)` raises `ValueError` when op has 0 eligible machines | `min(durations) if durations else 0` |
| 3 | `preprocess_company_real.py` | `parse_fjs_instance` silently returned fewer jobs than `n_jobs` if file truncated | Added `len(job_ops) != n_jobs` check with descriptive `ValueError` |
| 4 | `generate_toy_company_csv.py` | `compute_instance_stats` returned `NaN` for empty datasets | All `np.mean()` calls guarded with `if list else 0.0` |
| 5 | `generate_toy_company_csv.py` | `load_estimate = float("inf")` stored in stats → JSON serialization failure | Returns `None` instead of `inf` |
| 6 | `preprocess_company_real.py` | `arrival_times.tolist()` fails if arrivals is a plain Python list | Changed to `[int(x) for x in dataset["arrival_times"]]` |
| 7 | `preprocess_company_real.py` | Sidecar print showed wrong filename (stale `with_suffix` pattern) | Print now uses `sidecar_path` variable |
| 8 | `generate_toy_company_csv.py` | `n_jobs ≤ 0` causes division by zero in arrival span formula | Added `raise ValueError` guard |
| 9 | `generate_toy_company_csv.py` | `rdd ≥ 1.0` makes `lo_ddt ≤ 0` → degenerate due dates | `rdd_clamped = min(rdd, 0.99)` |
| 10 | `generate_toy_company_csv.py` | `target_util=0.0` causes division by zero in arrival span | `realized_util = max(0.01, float(target_util))` |

**Dead code removed in same session:**

| Item | File |
|---|---|
| `import hashlib` — imported but never used | `generate_toy_company_csv.py` |
| `arr_real = np.array(...)` — computed but never used | `generate_toy_company_csv.py` |
| `import json` as local function import | `preprocess_company_real.py` (moved to top-level) |
| Duplicate `import pandas as pd` | `preprocess_company_real.py` |
| Unused `import os` | `preprocess_company_real.py` |

### 18.5 Run Commands — Updated Dataset Generation

**Must regenerate benchmark bank** — old CSVs used the previous generator schema and are stale.

```
cd D:\UROP\Dynamic-Job-Shop-Scheduling-RL

# Step 1: Regenerate standard benchmark bank (n=20,50,100 × 5 seeds)
gitdjssp\Scripts\python.exe generate_toy_company_csv.py
```

Writes:
- `data/processed/toy_*.csv` — legacy toy dataset (n=20, seed=42, log-normal durations, compound Poisson)
- `data/benchmark/n20/seed_00/ ... seed_04/` — 5-seed benchmark instances
- `data/benchmark/n50/`, `data/benchmark/n100/`
- `data/benchmark/bank_manifest.json` — reproducibility manifest (schema_version=3.0)

```
# Step 2 (optional): Difficulty sweep for sensitivity analysis
gitdjssp\Scripts\python.exe -c "
import generate_toy_company_csv as g
g.generate_difficulty_sweep(
    real_excel_path='data/dataset_scheduling.xlsx',
    out_base='data/benchmark_sweep',
    n_jobs=20, n_seeds=5,
    ddt_levels=(1.2, 1.5, 2.0),
    rho_levels=(0.75, 0.85, 0.95),
    batch_levels=(1.0, 3.0),
)
"
```

Writes `data/benchmark_sweep/` with 3×3×2×5 = 90 instances across all difficulty combinations.

### 18.6 Known Remaining Limitations

These were identified but NOT fixed (architectural changes required):

| Item | Status |
|---|---|
| `env_company.py` does not use `op["durations"]` (per-machine times) | Not fixed — requires env refactor. Current env uses `op["duration"]` (single value). Brandimarte instances load correctly but run with min-time, not machine-specific times. |
| Route stitching still limited to ops from real data pool | Not fixed — no hallucinated operations. At n≥50, structural variety improves but op vocabulary is fixed. Disclose in methodology. |
| `mean_batch_ia` slightly biased when `max_batch` truncates large batches | Not fixed — minor statistical inaccuracy. `realized_util` in meta.json reflects actual load regardless. |
| `scaling_eval.dataset_sizes` in `base_config.yaml` still set to `[20, 40, 80]` | **Fixed 2026-03-19** — changed to `[20, 50, 100]`. See Section 19. |
| Novelty filter uses set Jaccard (ignores operation order) | Design choice — order-aware distance (e.g. Levenshtein) would be more precise but much slower. Acceptable for academic use. |

---

## Section 19 — Config & Scaling Eval Cleanup (2026-03-19)

### 19.1 Context

Following the v3 generator rewrite (Section 18), the `base_config.yaml` `scaling_eval` block and `experiments/run_scaling_eval.py` still referenced old v2 API parameters that were removed from the generator. These caused a size mismatch and would have caused `KeyError` / `TypeError` failures if `run_scaling_eval.py` was run without the benchmark bank being regenerated first.

### 19.2 Changes Made

**`experiments/configs/base_config.yaml`** — `scaling_eval` block:

| Key | Old value | New value | Reason |
|---|---|---|---|
| `dataset_sizes` | `[20, 40, 80]` | `[20, 50, 100]` | Must match `BENCHMARK_SIZES` in `generate_toy_company_csv.py` |
| `duration_jitter` | `[0.80, 1.25]` | removed | Old v2 uniform-jitter API; replaced by log-normal `duration_cov` |
| `slack_jitter` | `[0.85, 1.25]` | removed | Old v2 API; replaced by DDT/RDD model |
| `tight_deadlines` | `true` | removed | Old v2 post-hoc deadline override; v3 DDT/RDD sets due dates at generation time |
| `alpha_slack_min` | `1.1` | removed | Part of removed `tight_deadlines` override |
| `alpha_slack_max` | `1.8` | removed | Part of removed `tight_deadlines` override |
| `duration_cov` | — | `0.25` | New v3 param: CoV for log-normal duration distribution |
| `ddt` | — | `1.5` | New v3 param: due-date tightness factor |
| `rdd` | — | `0.2` | New v3 param: relative due-date dispersion |
| `batch_mean` | — | `1.0` | New v3 param: mean Poisson batch size (1.0 = simple Poisson) |

**`experiments/run_scaling_eval.py`**:

| Location | Change |
|---|---|
| `load_scaling_cfg()` return dict | Removed `tight_deadlines`, `alpha_slack_min`, `alpha_slack_max`, `duration_jitter`, `slack_jitter` keys |
| `load_scaling_cfg()` return dict | Added `duration_cov`, `ddt`, `rdd`, `batch_mean` keys (v3 params, for future on-the-fly generation use) |
| `load_dataset()` signature | Removed `tight_deadlines`, `alpha_min`, `alpha_max` parameters |
| `load_dataset()` body | Removed the tight-deadline due-date override block (`if tight_deadlines: ...`) |
| `main()` call to `load_dataset()` | Removed the three stale keyword arguments |
| CLI `--sizes` help text | Updated example from `--sizes 20 40 80` → `--sizes 20 50 100` |
| Module docstring Usage example | Updated from `--sizes 20 40 80` → `--sizes 20 50 100` |

### 19.3 Rationale

The `tight_deadlines` override in `load_dataset()` was a v2 workaround that re-computed due dates post-hoc using a uniform `alpha * total_work` formula. This is superseded by the v3 generator's DDT/RDD model, which sets due dates correctly at instance-generation time (conditioned on job work content with proper tightness and dispersion controls). Keeping the override would have contradicted the DDT/RDD dates already stored in the bank files.

### 19.4 Outstanding Next Steps (as of 2026-03-19)

| Priority | Task |
|---|---|
| **High** | Regenerate benchmark bank: `gitdjssp\Scripts\python.exe generate_toy_company_csv.py` — stale `data/benchmark/` artifacts use old schema |
| **High** | Update `env_company.py` to use `op["durations"]` (per-machine processing times from `.fjs` files) for full FJSP fidelity with Brandimarte instances |
| **Medium** | Download Brandimarte MK01–MK10 `.fjs` files into repo — parser exists (`parse_fjs_instance`) but no instances present |
| **Medium** | Fix Config v11/v12 YAML top-level override bug (keys not read by trainer) |
| **Low** | Run multi-seed training (3–5 seeds) for each of the 13 reward-weight variations |
