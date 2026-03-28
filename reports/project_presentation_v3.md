# Dynamic Job Shop Scheduling with RL — Architecture Update
## Technical Handoff v3 (2026-03-18)

> **Previous version**: `D:\UROP\reports\project_presentation_v2.md`
> Sections 1–5 (Problem, Why RL, Environment, Reward, Data Pipeline) of v2 are still accurate and unchanged.
> This document covers only what was **added or modified** in the refactoring session of 2026-03-18.

---

## What Changed

| v2 state | v3 state |
|----------|----------|
| Single `run_experiment.py` monolith (PPO only) | Thin dispatcher; PPO logic in `experiments/ppo/`, DQN in `experiments/dqn/` |
| No DQN support | 4 DQN-family variants with correct action masking |
| `compare_all.py` PPO-only colors | Algorithm-family-aware coloring (PPO=steelblue, DQN=darkorange, heuristics=coral) |
| No `--algo` filter in compare | `--algo {all,ppo,dqn}` working |

---

## Updated File Map

```
Dynamic-Job-Shop-Scheduling-RL/
│
├── env_company.py                        ← UNCHANGED — CompanyFJSSPEnv
├── preprocess_company_real.py            ← UNCHANGED
├── generate_toy_company_csv.py           ← MODIFIED (2026-03-18) — benchmark bug fix
│
├── experiments/
│   ├── run_experiment.py                 ← MODIFIED: thin dispatcher + re-exports only
│   ├── evaluate_variation.py             ← MODIFIED: PPO/DQN auto-detection and dispatch
│   ├── compare_all.py                    ← MODIFIED: DQN-family coloring + --algo filter
│   ├── early_stopping_callback.py        ← UNCHANGED
│   ├── composite_score.py                ← UNCHANGED
│   ├── gantt_utils.py                    ← UNCHANGED
│   │
│   ├── configs/                          ← UNCHANGED — 13 PPO variation YAMLs + base
│   │   ├── base_config.yaml
│   │   ├── v1_ppo_default.yaml
│   │   └── ... (v2 through v13)
│   │
│   ├── common/                           ← NEW — shared utilities (no PPO/DQN imports)
│   │   ├── __init__.py
│   │   ├── dataset.py                    build_dataset(), compute_max_steps()
│   │   └── output_utils.py               deep_merge(), create_output_dir(),
│   │                                      save_config_copy(), save_notes(),
│   │                                      save_learning_curve(), save_training_summary()
│   │
│   ├── ppo/                              ← NEW — all PPO logic extracted from run_experiment.py
│   │   ├── __init__.py
│   │   ├── trainer.py                    load_config(), make_env_factory(), build_model(),
│   │   │                                  calibrate_makespan_bounds(), main()
│   │   ├── callbacks.py                  RewardLoggerCallback, CheckpointCallback
│   │   └── evaluate.py                   simulate_ppo(), load_ppo_model()
│   │
│   └── dqn/                              ← NEW — DQN-family training stack
│       ├── __init__.py
│       ├── agent.py                      DQNAgent (vanilla/double/dueling/d3qn)
│       │                                  StandardQNetwork, DuelingQNetwork
│       ├── replay_buffer.py              ReplayBuffer (stores obs, action, reward,
│       │                                  next_obs, done, mask, next_mask)
│       ├── trainer.py                    load_dqn_config(), make_single_env(),
│       │                                  get_epsilon(), DQNEpisodeLogger, main()
│       ├── evaluate.py                   simulate_dqn(), load_dqn_model()
│       └── configs/
│           ├── base_dqn_config.yaml      shared DQN defaults (2M steps, lr=1e-4, ...)
│           ├── dqn_vanilla.yaml          standard DQN
│           ├── dqn_double.yaml           Double DQN
│           ├── dqn_dueling.yaml          Dueling DQN
│           └── dqn_d3qn.yaml             Dueling + Double (strongest baseline)
│
└── experiments/commands.txt             ← NEW — all training / eval commands
```

---

## Backward Compatibility

`run_experiment.py` re-exports all PPO symbols from `ppo/trainer.py` and `ppo/callbacks.py`:

```python
from experiments.run_experiment import (
    load_config, build_dataset, compute_max_steps,
    make_env_factory, build_model, make_linear_schedule,
    calibrate_makespan_bounds, save_learning_curve,
    RewardLoggerCallback, CheckpointCallback,
    main,           # ← now dispatches to PPO or DQN based on config
)
```

All existing callers (`run_smoke_tests.py`, `run_all_experiments.py`, `run_scaling_eval.py`) continue to work without modification.

---

## DQN Integration Details

### Algorithm Dispatch

`run_experiment.main()` peeks at `algorithm_family:` in the YAML:

```python
# Detected as PPO (default — no key needed):
#   experiments/configs/v1_ppo_default.yaml

# Detected as DQN (algorithm_family: dqn in YAML):
#   experiments/dqn/configs/dqn_vanilla.yaml

# Force override at CLI:
python experiments/run_experiment.py --config ... --algo dqn
```

### Four DQN Variants

| Variant | Key | What it adds |
|---------|-----|-------------|
| `vanilla` | `dqn_vanilla.yaml` | Baseline DQN with action masking |
| `double` | `dqn_double.yaml` | Decouples action selection from eval (reduces Q-overestimation) |
| `dueling` | `dqn_dueling.yaml` | V(s) + A(s,a) - mean(A) decomposition |
| `d3qn` | `dqn_d3qn.yaml` | Dueling + Double — expected strongest DQN baseline |

### Action Masking in DQN

Masking is enforced at **four stages** in `dqn/agent.py`:

| Stage | Implementation |
|-------|---------------|
| **Greedy selection** | `q[~mask] = -inf` → argmax always picks valid action |
| **Epsilon-greedy** | `np.random.choice(np.where(mask)[0])` — random only from valid actions |
| **Target computation** | `next_q[~next_mask] = -inf` before max/argmax — target never propagates invalid actions |
| **Evaluation** | `select_action(obs, mask, epsilon=0.0)` — fully deterministic greedy |

Double/D3QN target: `a* = argmax_{valid} Q_online(s')`, then `Q_target(s', a*)` — masked selection on online net, masked evaluation on target net.

**Important difference from PPO**: DQN uses a **bare** `CompanyFJSSPEnv` with no `ActionMasker` wrapper. `env.get_action_mask()` is called manually at each step and stored in the replay buffer alongside each transition.

### DQN vs PPO Environment Handling

| | PPO | DQN |
|-|-----|-----|
| Env wrapper | `ActionMasker(env, mask_fn)` | None — bare `CompanyFJSSPEnv` |
| Parallelism | `make_vec_env(factory, n_envs=N)` | Single env, sequential |
| Mask access | Via SB3 `get_action_masks()` | `env.get_action_mask()` called per step |
| Mask storage | Not stored (on-policy) | Stored per-transition in `ReplayBuffer.masks` + `next_masks` |

### Model Save/Load

| Algorithm | Saved format | Load path passed to loader |
|-----------|-------------|---------------------------|
| PPO | `models/model.zip` | path **without** `.zip` → `MaskablePPO.load(path)` |
| DQN | `models/model.pt` | path **without** `.pt` → `DQNAgent.load(path)` |

`evaluate_variation.find_model_path()` detects the format automatically from the file extension.

### DQN Base Config Defaults

```yaml
# experiments/dqn/configs/base_dqn_config.yaml
dqn:
  variant:              "vanilla"
  total_timesteps:      2000000
  batch_size:           64
  buffer_size:          50000       # transitions kept in replay buffer
  learning_rate:        0.0001
  gamma:                0.99
  epsilon_start:        1.0
  epsilon_end:          0.05
  epsilon_decay_steps:  500000      # linear decay from start to end
  target_update_freq:   1000        # hard copy online → target every N steps
  learning_starts:      5000        # collect this many steps before first update
  train_freq:           4           # update every N steps
  hidden_sizes:         [256, 256]  # same as PPO default network
  device:               "cuda"
```

---

## Evaluation and Comparison

`evaluate_variation.py` reads `config/run_config.yaml` from the experiment folder to detect `algorithm_family`, then dispatches to:

- `simulate_ppo(model_path, env, n_episodes)` — MaskablePPO.load + deterministic predict with masks
- `simulate_dqn(agent, env, n_episodes)` — DQNAgent.load + select_action(epsilon=0.0)

Both return the same metrics dict:
`{makespan, total_tardiness, utilization, frac_deadlines_met, deadlines_met, total_jobs}`

`compare_all.py` discovers all timestamped folders under `experiments/`, loads `metrics/metrics.csv` from each, and tags each RL row with `algorithm_family` from `config/run_config.yaml`. Plot colors:
- **steelblue** = PPO agents
- **darkorange** = DQN-family agents
- **coral** = heuristics (always shown in all filter modes)

---

## Output Structure (same for PPO and DQN)

```
experiments/YYYYMMDD_HHMMSS_<variation_name>/
    models/
        model.zip          (PPO) or model.pt (DQN) — final
        best_model.zip     (PPO only — saved by EarlyStoppingCallback)
        checkpoint_500k.*  periodic checkpoints
    logs/
        episode_stats.csv  per-episode metrics
        step_logs.csv      per-step reward decomposition (sampled every 100 steps)
        eval_history.csv   (PPO only — from EarlyStoppingCallback)
    metrics/
        training_summary.json
    plots/
        learning_curve.png
    gantt/                 (written by evaluate_variation.py, not training)
    config/
        run_config.yaml    merged config (algorithm_family key present)
        variation_name.txt
    notes/
        notes.md
```

---

## Benchmark Fix (2026-03-18) — Critical

**Bug**: All 8 heuristics and all RL models produced identical makespan/tardiness/deadlines_met,
causing all methods to rank 1 in comparison plots.

**Root cause in `generate_toy_company_csv.py`** — two problems:

1. **Arrival span formula used a `max()` floor that always won.**
   The proportional formula computed ~19,707 min, but then the code floored it to
   `max(proportional, np.max(arr_real)+1)` = ~37,441 min (the real dataset span).
   With 20 jobs spread over 37,000 min and each job taking ~4,800 min, average
   interarrival = 1,750 min >> job duration → queue depth always ≤1 →
   all heuristics pick the same job every time → identical results.

2. **Slack was too generous.**
   `slack_jitter=(0.85, 1.25)` × real slacks (~28,000 min) → slacks of 23,800–35,000 min
   for ~4,800 min jobs. Every job always finishes well before its due date → tardiness=0
   always → no differentiation between EDD, CR, SLACK, FIFO.

**Three fixes applied:**

```python
# experiments/generate_toy_company_csv.py  (lines 126–138)

# BEFORE (broken):
arrival_span = max(
    int(math.ceil(total_work_est / (n_machines * 0.70))),  # ~19,707 — ignored
    int(np.max(arr_real)) + 1,                             # ~37,441 — always wins
)

# AFTER (fixed):
target_util  = 0.90                           # tighter packing (was 0.70)
arrival_span = int(math.ceil(total_work_est / (n_machines * target_util)))
# = ceil(20 × 4,828 / (7 × 0.90)) = 15,328 min  ← dense enough for competition
```

```python
# main() benchmark call — slack_jitter changed
generate_benchmark_bank(
    ...
    slack_jitter=(0.25, 0.65),   # was (0.85, 1.25) — creates real due-date urgency
)
```

**Verified outcome** — 8 heuristics on new n20/seed_00 benchmark:

| Rule  | Makespan | Tardiness | dl_met |
|-------|----------|-----------|--------|
| FIFO  | 25,640   | 12,618    | 16/20  |
| LIFO  | 24,900   | 15,849    | 17/20  |
| SPT   | 24,920   | 11,589    | 17/20  |
| LPT   | 24,920   | 18,758    | 16/20  |
| EDD   | 24,900   | 11,479    | 17/20  |
| LDD   | 25,230   | 18,098    | 16/20  |
| SLACK | 24,950   | 11,529    | 17/20  |
| CR    | 24,900   | 11,559    | 16/20  |

**Action required**: All n20 experiments trained before 2026-03-18 must be **retrained** —
they learned on the degenerate dataset where all dispatch choices produce identical outcomes.

---

## Known Limitations (added in this session)

1. **No DQN best-model saving** — PPO saves `best_model.zip` when composite score improves during training. DQN has no online evaluation loop; only step-count checkpoints + final model are saved.

2. **No DQN early stopping** — DQN always trains for the full `total_timesteps`. PPO has `EarlyStoppingCallback` with composite-score patience.

3. **Reward weights untuned for DQN** — Reward weights in `base_config.yaml` were optimised for PPO. DQN may benefit from reward normalisation (`clip(r, -1, 1)`) due to Q-learning divergence sensitivity.

4. **DQN is ~2–3× slower wall-clock than PPO** — PPO uses `n_envs` parallel environments; DQN uses a single sequential env. For 2M timesteps: PPO ≈ 20–40 min, DQN ≈ 60–120 min (GPU).

5. **Gym deprecation warning** — The env still imports `gym` (not `gymnasium`). Functional but prints a warning on startup. No change to env was made (out of scope).

---

## Quick Command Reference

All commands documented in:
`D:\UROP\Dynamic-Job-Shop-Scheduling-RL\experiments\commands.txt`

Essential commands:

```bash
# Smoke test all algorithms (~30 s)
python experiments/run_experiment.py --config experiments/configs/v1_ppo_default.yaml --timesteps 1000
python experiments/run_experiment.py --config experiments/dqn/configs/dqn_d3qn.yaml   --timesteps 1000

# Train (full 2M steps)
python experiments/run_experiment.py --config experiments/configs/v8_critical_path_extreme.yaml --n_jobs 20 --seed 0
python experiments/run_experiment.py --config experiments/dqn/configs/dqn_d3qn.yaml             --n_jobs 20 --seed 0

# Evaluate one experiment
python experiments/evaluate_variation.py --experiment experiments/<YYYYMMDD_HHMMSS_name>/

# Compare all
python experiments/compare_all.py
python experiments/compare_all.py --algo ppo
python experiments/compare_all.py --algo dqn
```

---

## Document History

| Version | Date | Description |
|---------|------|-------------|
| v1 | Dec 2025 | Initial project notes |
| v2 | Mar 2026-03-17 | Full code-verified reference: env, reward, obs space, heuristics, PPO training |
| v3 | 2026-03-18 | DQN integration, modular refactor, updated file map, known limitations |
| v3.1 | 2026-03-18 | Added Benchmark Fix section: arrival_span bug + slack_jitter fix; verified heuristics now differ |

**For env, reward function, observation space, action masking, PPO hyperparams, heuristic definitions, data pipeline details — read v2.**
