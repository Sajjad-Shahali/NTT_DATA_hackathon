# Dynamic Job Shop Scheduling with Reinforcement Learning
## Complete Technical Reference (Code-Verified)

> **Project**: Flexible Dynamic Job Shop Scheduling Problem (FJSSP/DJSSP) solved with Proximal Policy Optimization on Real Manufacturing Data
> **Period**: December 2025 – March 2026
> **Purpose of this document**: Exact, code-verified reference. Every detail below was confirmed from the actual source files. Intended as context for AI assistants, collaborators, or reviewers who need to understand the complete system without reading all source files.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Why Reinforcement Learning?](#2-why-reinforcement-learning)
3. [System Architecture & File Map](#3-system-architecture--file-map)
4. [Environment: `CompanyFJSSPEnv`](#4-environment-companyfpsspenv)
   - 4.1 Dataset Schema Expected
   - 4.2 Internal State Variables
   - 4.3 Observation Space (exact)
   - 4.4 Action Space & Machine Selection
   - 4.5 Action Masking (exact logic)
   - 4.6 Time Advancement
   - 4.7 Step Logic (full flow)
   - 4.8 Termination Conditions
   - 4.9 `info` Dict Contents
   - 4.10 `compute_metrics()` Output
   - 4.11 Gantt Log
5. [Reward Function (exact)](#5-reward-function-exact)
6. [RL Algorithm: MaskablePPO](#6-rl-algorithm-maskableppo)
7. [Data Pipeline](#7-data-pipeline)
   - 7.1 Real Company Data (Excel)
   - 7.2 Toy/Synthetic Dataset
   - 7.3 CSV Schemas
   - 7.4 Actual Dataset Numbers
8. [Training Infrastructure](#8-training-infrastructure)
9. [Evaluation & Baselines](#9-evaluation--baselines)
10. [Key Design Decisions & Known Quirks](#10-key-design-decisions--known-quirks)
11. [Dependencies](#11-dependencies)
12. [Glossary](#12-glossary)

---

## 1. Problem Definition

### 1.1 The Flexible Dynamic Job Shop Scheduling Problem

**Job Shop Scheduling Problem (JSSP)**: A set of $N$ jobs must be processed on $M$ machines. Each job has an ordered sequence of operations; each operation runs on one machine for a known duration. Goal: build a schedule minimising a cost objective.

**Flexible (FJSSP)**: Each operation can execute on **multiple eligible machines** (not just one). The scheduler must also decide *which machine* to use.

**Dynamic (DJSSP)**: Jobs arrive over time with individual `arrival_time > 0`. The full job set is NOT known at the start. A job cannot be dispatched before its arrival time.

This project addresses the combined **Flexible DJSSP**.

### 1.2 Real-World Context

The dataset comes from a real CNC/milling manufacturing facility:

- **Operations** (process steps): `Fre_20`, `Fre_30`, `Fre_40`, `Fre_50`, `Fre_51`, `Fre_60`, `Fre_70`, `Fre_22` (Fresatura operations)
- **Machines**: `F2_1`, `F2_2`, `F2_3`, `F2_4`, `F2_5`, `F1_2`, `F1_3` — **7 machines total**
- **Eligibility scores**: `2` = machine always capable, `1` = machine occasionally capable
- **Source**: `data/dataset_scheduling.xlsx` with three sheets: `"job"`, `"dati_cicli"`, `"operazioni_macchine"`

### 1.3 Formal Problem Statement

$$\text{Minimise} \quad \underbrace{C_{\max}}_{\text{makespan}} + \;\underbrace{\sum_{j=1}^N \max(0,\, C_j - D_j)}_{\text{total tardiness}} + \;\underbrace{(1-U)}_{\text{idle machines}}$$

where all three objectives are simultaneously handled through reward shaping.

### 1.4 Decision at Each Step

At each discrete time step: **which job's next operation should be dispatched right now?**

The environment then automatically picks a free eligible machine for that job.

---

## 2. Why Reinforcement Learning?

| Approach | Limitation for DJSSP |
|----------|---------------------|
| **Exact methods** (ILP/MIP) | NP-hard; intractable beyond ~20 jobs |
| **Metaheuristics** (GA, SA) | Need full problem upfront; not dynamic |
| **Dispatching rules** (SPT, EDD, …) | Fixed policy; cannot adapt to multi-objective state |
| **RL (PPO)** | Online, adaptive; balances objectives via reward; near-instant inference |

**RL framing**:

| RL Concept | Scheduling Meaning |
|------------|-------------------|
| State $s_t$ | Machine busy times, job urgency, remaining work, current clock |
| Action $a_t$ | Index of the job to dispatch (or no-op) |
| Reward $r_t$ | On-time completions, utilisation bonus, tardiness penalty |
| Policy $\pi$ | The learned scheduling strategy |
| Episode | One complete production schedule (all jobs finish or time-up) |

---

## 3. System Architecture & File Map

```
D:\UROP\
│
├── Dynamic-Job-Shop-Scheduling-RL/          ← MAIN MODULE
│   ├── env_company.py                       ← CompanyFJSSPEnv (primary RL env)
│   ├── env.py                               ← Older toy DJSSP env (not used for company data)
│   ├── train.py                             ← Training script (MaskablePPO)
│   ├── evaluate.py                          ← Benchmark: PPO vs 8 heuristics
│   ├── preprocess_company_real.py           ← Excel → dataset dict → CSV export/load
│   ├── generate_toy_company_csv.py          ← Synthetic dataset (bootstrapped from real)
│   ├── utils.py                             ← plot_gantt(), load_dataset() helpers
│   ├── main.py                              ← Entry point (calls train.py logic)
│   ├── test_company_real_env.py             ← Smoke test for CompanyFJSSPEnv
│   ├── req.txt                              ← Full requirements (gymnasium 0.29.1, sb3 2.3.2)
│   │
│   ├── data/
│   │   ├── dataset_scheduling.xlsx          ← Real company source (3 sheets)
│   │   └── processed/
│   │       ├── toy_jobs.csv                 ← 20-job toy dataset (jobs)
│   │       ├── toy_operations.csv           ← toy op sequences
│   │       ├── toy_eligibility.csv          ← toy machine eligibility
│   │       ├── company_fjsp_7m_jobs.csv     ← Full real company dataset
│   │       ├── company_fjsp_7m_operations.csv
│   │       └── company_fjsp_7m_eligibility.csv
│   │
│   ├── results/                             ← Training run outputs (run_YYYYMMDD_HHMMSS/)
│   └── results_company_real/               ← Evaluation outputs
│
├── data/                                    ← Root-level legacy datasets
│   ├── jobs.csv, job_ops.csv
│   ├── op2machines.csv, ops_template.csv
│   └── company_dataset_flat.csv
│
├── env.py                                   ← Root FlexibleJsspEnv (older version)
├── train.py, evaluate.py, utils.py          ← Root-level scripts (older)
├── generate_dataset_final.py               ← Synthetic dataset generator
├── reports/
│   └── project_presentation_v2.md          ← This file
└── djssp/                                  ← Python virtualenv
```

### Key Distinction: Two Parallel Codebases

| | Root (`D:\UROP\`) | Production (`Dynamic-Job-Shop-Scheduling-RL/`) |
|-|----|----|
| Environment class | `FlexibleJsspEnv` (env.py) | `CompanyFJSSPEnv` (env_company.py) |
| Dataset | Simple CSV (jobs × machines matrix) | Rich dict with per-job op routes + eligibility |
| Machines | Fixed per job | Flexible per operation |
| Arrival | Supported | Supported |
| Status | Legacy / reference | **Active** |

---

## 4. Environment: `CompanyFJSSPEnv`

**File**: `Dynamic-Job-Shop-Scheduling-RL/env_company.py`
**Class**: `CompanyFJSSPEnv(gym.Env)`

### 4.1 Dataset Schema Expected by the Environment

The environment `__init__` takes a `dataset: dict` with this exact structure:

```python
dataset = {
    "arrival_times": np.ndarray[int],   # shape (N,), minutes from t0
    "due_dates":     np.ndarray[int],   # shape (N,), minutes from t0
    "job_names":     list[str],         # length N
    "machine_names": list[str],         # length M (7 for real company)
    "job_ops":       list[list[dict]],  # length N; each job is a list of ops
    "t0":            pd.Timestamp,      # reference datetime
    "time_unit":     str,               # "minute" (training default)
}

# Each op dict:
{
    "op_id":              str,        # e.g. "Fre_20"
    "duration":           int,        # in time_unit
    "eligible_machines":  list[int],  # machine indices into machine_names
    "scores":             list[int],  # same length; 1=occasional, 2=always
}
```

**Inside `__init__`**: eligibility is immediately filtered — only machines with `score == 2` are kept. If none have score 2, falls back to all `score > 0`. If still none, allows all machines.

### 4.2 Internal State Variables

All state is reset by `_reset_state()` at each episode start:

| Variable | Type | Description |
|----------|------|-------------|
| `current_time` | `float` | Simulation clock in minutes |
| `job_next_op[j]` | `int[N]` | Index of next pending op for job $j$ |
| `job_done[j]` | `bool[N]` | True when job $j$ fully completed |
| `job_completion_time[j]` | `float[N]` | Set to `current_time` when job finishes; -1 if unfinished |
| `machine_status[m]` | `None` or `dict` | `None` = idle; dict has `{"job": j, "op_index": k, "remaining": float}` |
| `total_machine_time` | `float` | Cumulative $M \times \Delta t$ (for utilisation) |
| `busy_machine_time` | `float` | Cumulative $\sum_m \mathbf{1}[\text{busy}] \times \Delta t$ |
| `steps_taken` | `int` | Steps elapsed this episode |
| `jobs_completed` | `int` | Count of finished jobs |
| `_gantt_log` | `list[dict]` | Log of all dispatched operations |

**Scaling constant** (computed once, not reset):

$$\tau = \max\!\left(\max_j D_j,\; \sum_j W_j,\; 1.0\right)$$

where $W_j = \sum_{\text{op}} \text{duration}$ is total work for job $j$.

### 4.3 Observation Space (exact)

**Space declared**: `Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=float32)`

> ⚠️ **Important quirk**: `slack_norm` uses `tanh((D_j - t) / τ)`. When a job is **overdue** (`t > D_j`), `slack_norm < 0`. The space is declared `[0,1]` but can contain negative values for overdue jobs. This is a known minor inconsistency in the code.

**Observation layout** (total dim = $1 + 4N + M$):

```
[ current_time_norm | job_0_feats | job_1_feats | ... | job_{N-1}_feats | mach_0_feat | ... | mach_{M-1}_feat ]
```

**Global (1 dim)**:
- `current_time_norm = tanh(t / τ)`

**Per-job features (4 dims × N)**:
| Index offset | Name | Formula |
|---|---|---|
| +0 | `arrived_flag` | `1.0` if `arrival_times[j] <= t` else `0.0` |
| +1 | `done_flag` | `1.0` if `job_done[j]` else `0.0` |
| +2 | `slack_norm` | `tanh((due_dates[j] - t) / max(1, τ))` |
| +3 | `remaining_norm` | `tanh(remaining_work(j) / max(1, τ))` |

**Per-machine features (1 dim × M)**:
| Index offset | Name | Formula |
|---|---|---|
| +0 | `time_to_free_norm` | `tanh(max(0, machine_status[m]["remaining"]) / max(1, τ))` |

**Toy dataset example**: 20 jobs + 7 machines → obs_dim = $1 + 4(20) + 7 = \mathbf{88}$

### 4.4 Action Space & Machine Selection

**Action space**: `Discrete(num_jobs + 1)`

| Action value | Meaning |
|---|---|
| `0` to `N-1` | Dispatch job $j$'s next pending operation |
| `N` | No-op: do nothing this step (only valid when NO jobs are dispatchable — see 4.5) |

**Machine selection** (`_assign_job_to_machine(j)`):
The environment searches through the job's operation's eligible machine list **in order** and picks the **first idle** one:

```python
for m in op["eligible_machines"]:   # first eligible idle machine
    if machine_status[m] is None:
        machine_status[m] = {"job": j, "op_index": k, "remaining": dur}
        return True
```

This is **not** minimum-start-time selection — it is purely first-eligible-idle. Machine order comes from the pre-filtered eligibility list (score-2 machines first by their index order).

### 4.5 Action Masking (exact logic)

```python
def get_action_mask(self) -> np.ndarray:   # shape (num_jobs + 1,)
    mask = np.zeros(self.num_jobs + 1, dtype=bool)
    dispatchable = self._get_dispatchable_jobs()
    for j in dispatchable:
        mask[int(j)] = True
    # no-op is ONLY valid when there are zero dispatchable jobs
    mask[self.num_jobs] = (len(dispatchable) == 0)
    return mask
```

> ⚠️ **Critical**: The no-op action (`action = N`) is **NOT always valid**. It is only valid when `_get_dispatchable_jobs()` returns empty. This means if any job is dispatchable, the no-op is masked out — the agent MUST dispatch a job.

**`_get_dispatchable_jobs()`** returns job indices where ALL of:
1. `arrival_times[j] <= current_time` (job has arrived)
2. `not job_done[j]` (job not finished)
3. `job_next_op[j] < job_num_ops[j]` (job has remaining ops)
4. At least one of its eligible machines is currently `None` (idle)

### 4.6 Time Advancement

`_advance_time()` runs every step (after the dispatch action):

1. Clock advances by `time_step` (default **10 minutes**)
2. All busy machines have `remaining -= time_step`
3. Machines whose `remaining <= 1e-9` complete their operation:
   - `job_next_op[j]` increments
   - If all ops done: `job_done[j] = True`, `job_completion_time[j] = current_time`
4. Computes and returns:
   - `finished_jobs`: list of job indices completing this step
   - `busy_frac`, `idle_frac`: machine occupancy fractions
   - `num_late_unfinished`: jobs that are both unfinished and past their due date
   - `total_lateness`: total accumulated lateness (finished + unfinished overdue)

### 4.7 Step Logic (full flow)

```
step(action):
  1. Validate action against dispatchable set
     - If valid job & dispatchable: _assign_job_to_machine(j)
     - If action == N (no-op): pass
     - Otherwise: invalid_action = True
  2. _advance_time() → progress clock by time_step
  3. Compute reward (see Section 5)
  4. Check termination (see 4.8)
  5. _build_observation()
  6. Return (obs, reward, terminated, truncated=False, info)
```

### 4.8 Termination Conditions

An episode ends (`terminated = True`) when EITHER:
- `steps_taken >= max_steps` (timeout)
- `jobs_completed >= num_jobs` (all jobs finished)

`truncated` is always `False` (truncation is not used).

At termination, **if there are unfinished jobs**, a large penalty is added:
```python
extra = rw["final_unscheduled"] * (num_jobs - jobs_completed)
```

### 4.9 `info` Dict Contents

Every `step()` returns this info dict:

```python
info = {
    "current_time":           float,
    "jobs_completed":         int,
    "busy_frac":              float,   # fraction of machines busy this step
    "idle_frac":              float,
    "num_late_unfinished":    int,     # jobs overdue+unfinished after this step
    "late_unfinished_frac":   float,
    "total_lateness":         float,
    "job_completed_this_step": bool,
    "job_completed_late":      bool,
    "invalid_action":          bool,
    "reward_terms": {
        "job_complete":        float,
        "job_on_time":         float,
        "job_late":            float,
        "late_unfinished_step": float,
        "machine_idle":        float,
        "utilization_bonus":   float,
        "step_time_penalty":   float,
        "invalid_action":      float,
        "final_unscheduled":   float,  # only non-zero on terminal step
        "total":               float,
    },
}
```

### 4.10 `compute_metrics()` Output

Called after an episode to get industrial KPIs:

```python
metrics = {
    "makespan":        float,   # max(job_completion_time) if all done, else current_time
    "total_tardiness": float,   # sum of max(0, C_j - D_j) for all jobs
    "utilization":     float,   # busy_machine_time / (makespan * num_machines)
    "deadlines_met":   int,     # jobs with C_j <= D_j
    "total_jobs":      int,     # N
    "unscheduled_jobs": int,    # jobs that never completed
}
```

> ⚠️ **Unfinished jobs in compute_metrics**: For tardiness/deadline calculations, unfinished jobs are treated as completing at `makespan` (i.e., `current_time`). This means unfinished jobs can accidentally appear to meet their deadline if `current_time < due_date` — which is a known conservative assumption.

### 4.11 Gantt Log

Every dispatched operation is stored in `_gantt_log`:

```python
# One entry per operation assignment:
{
    "job":      int,    # job index j
    "op":       int,    # operation index k within job j
    "machine":  int,    # machine index m
    "start":    float,  # current_time when assigned
    "duration": float,  # op duration
    "end":      float,  # start + duration
}
```

Access via `env.get_gantt_log()`. Reset each episode.

Converted by `evaluate.py:build_numeric_gantt()` → list of `(job, machine, start, end)` tuples for `plot_gantt()`.

---

## 5. Reward Function (exact)

All reward computation is in `env_company.py:step()`.

### 5.1 Reward Weights (defaults in `env_company.py`)

```python
self.rw = {
    "job_complete":        8.0,     # base reward: job finishes
    "job_on_time":        40.0,     # bonus: finished before/on due date
    "job_late":          -80.0,     # coefficient × (tardiness / τ)
    "late_unfinished_step": -80.0,  # coefficient × (overdue_fraction) × dt_norm
    "machine_idle":      -40.0,     # coefficient × idle_frac × dt_norm
    "utilization_bonus":  30.0,     # coefficient × busy_frac × dt_norm
    "step_time_penalty": -10.0,     # UNSCALED flat per-step penalty
    "final_unscheduled": -100.0,    # × number of unfinished jobs at episode end
    "invalid_action":     -1.0,     # = -2.0 × invalid_action_penalty(0.5)
    "tardiness_step":   -120.0,     # coefficient × total_late_amount_norm × dt_norm
}
```

> ⚠️ **Important**: `train.py` calls `CompanyFJSSPEnv(..., reward_weights={"tardiness": 1.0, "delta_makespan": 0.1, ...})` but these keys (`"tardiness"`, `"delta_makespan"`, `"on_time_bonus"`, `"slack"`, `"step_penalty"`) **do NOT match** the keys in `self.rw`. The `dict.update()` call adds them as new keys but never overrides the active defaults. **As a result, training always uses the default reward weights above.**

### 5.2 Full Reward Calculation Per Step

```
dt_norm = time_step / τ                        # normalisation factor

# --- Dense tardiness term ---
late_amount = sum( max(0, t - D_j) for unfinished jobs j )
late_amount_norm = late_amount / max(1e-9, τ)
reward += dt_norm * rw["tardiness_step"] * late_amount_norm

# --- Job completion (if a job finished this step) ---
if job_completed:
    reward += rw["job_complete"]
    if tardiness == 0:
        reward += rw["job_on_time"]
    else:
        reward += rw["job_late"] * (tardiness / τ)

# --- Per-step shaping (dt_norm scaled) ---
reward += rw["step_time_penalty"]                           # NOTE: unscaled flat
reward += dt_norm * rw["late_unfinished_step"] * late_unfinished_frac
reward += dt_norm * (rw["machine_idle"] * idle_frac + rw["utilization_bonus"] * busy_frac)

# --- Invalid action ---
if invalid_action:
    reward += rw["invalid_action"]   # = -1.0

# --- Episode end ---
if done and unfinished > 0:
    reward += rw["final_unscheduled"] * unfinished
```

> ⚠️ `step_time_penalty` is applied **without** `dt_norm` scaling — it is a flat -10.0 per step regardless of `time_step`.

### 5.3 Objective Priority (by weight magnitude)

1. **Tardiness step** (−120 × norm): highest-frequency signal, dominant training pressure
2. **On-time bonus** (+40): biggest single positive signal
3. **Late completion** (−80 × norm): strongest completion-level penalty
4. **Idle penalty** (−40 × frac × dt_norm): utilisation pressure
5. **Late unfinished** (−80 × frac × dt_norm): overdue clearance
6. **Job complete** (+8): baseline completion reward
7. **Step penalty** (−10 flat): schedule duration pressure
8. **Utilisation bonus** (+30 × frac × dt_norm): positive utilisation signal
9. **Final unscheduled** (−100 × count): catastrophic penalty

---

## 6. RL Algorithm: MaskablePPO

### 6.1 Why MaskablePPO

Standard PPO samples from ALL actions including invalid ones. In scheduling, ~50-90% of actions can be invalid at a given step (unarrived/done jobs). MaskablePPO zeros out invalid logits before softmax:

$$\pi(a \mid s) \propto \exp(z_a) \cdot m_a \quad \text{where } m_a \in \{0,1\}$$

`sb3_contrib.MaskablePPO` requires an `ActionMasker` wrapper that calls `env.get_action_mask()` at each step.

### 6.2 PPO Objective

$$\mathcal{L} = \mathbb{E}_t\!\left[\min\!\left(\hat{r}_t \hat{A}_t,\; \text{clip}(\hat{r}_t, 1\!-\!\epsilon, 1\!+\!\epsilon)\hat{A}_t\right)\right] - c_1 \mathcal{L}_V + c_2 H[\pi]$$

- $\hat{r}_t = \pi_\theta / \pi_{\text{old}}$ (probability ratio)
- $\hat{A}_t$: Generalised Advantage Estimate (GAE)
- $c_1 = 0.5$ (value loss), $c_2 = 0.01$ (entropy coef)

### 6.3 Hyperparameters (from `train.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `learning_rate` | `3e-4` | Adam LR |
| `n_steps` | `8192` | Rollout buffer length per env |
| `batch_size` | `1024` | SGD mini-batch size |
| `gamma` | `0.999` | Discount factor (near 1 for long eps) |
| `gae_lambda` | `0.95` | GAE bias-variance tradeoff |
| `ent_coef` | `0.01` | Entropy bonus coefficient |
| `vf_coef` | `0.5` | Value function loss weight |
| `max_grad_norm` | `0.5` | Gradient clipping |
| `clip_range` | `0.2` | PPO clip epsilon |
| `n_epochs` | `10` (default) | Optimisation epochs per rollout |
| `total_timesteps` | `2,000,000` | Training budget |
| `device` | `"cpu"` | Inference device |
| `n_envs` | `max(1, cpu_count // 2)` | Parallel envs |

### 6.4 Policy Network Architecture

```python
policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
```

Both policy and value heads: `obs_dim → 256 → Tanh → 256 → Tanh → output`

- Policy output: `num_jobs + 1` logits → masked softmax
- Value output: scalar

---

## 7. Data Pipeline

### 7.1 Real Company Data (Excel → Dataset)

**Source file**: `data/dataset_scheduling.xlsx`

| Excel Sheet | Contents | Parser function |
|-------------|----------|-----------------|
| `"job"` | Job list: `ID pezzo`, `Data arrivo` (datetime), `Due date` (datetime) | `job_df = pd.read_excel(..., sheet_name="job")` |
| `"dati_cicli"` | Per-product operation routes: block per part code (4-digit + letter + 4-digit), operation codes `Fre_XX`, durations in minutes | `parse_dati_cicli()` |
| `"operazioni_macchine"` | Eligibility matrix: rows = operation codes, columns = machine names, values ∈ {0,1,2} | `parse_operazioni_macchine()` |

**`build_company_fjsp_dataset(excel_path, time_unit="hour")`** (main entry point):
1. Reads all three sheets
2. Sets `t0 = min(arrival_date)` as time origin
3. Converts all datetimes to integer steps in `time_unit`
4. Matches job's part code to route from `dati_cicli`
5. Builds `job_ops` dicts with eligibility from `operazioni_macchine`
6. Returns dataset dict

**`load_company_dataset_from_csv(out_dir, prefix, time_unit="minute")`**: reconstructs the same dict from pre-exported CSVs (no Excel needed at runtime).

### 7.2 Toy/Synthetic Dataset (`generate_toy_company_csv.py`)

Generated by **bootstrapping from real data**:
- Samples `n_jobs` (default = same as real, or 20 for "toy" prefix) jobs
- Bootstraps arrival times and slack from real distributions
- Randomly picks an operation route template from real jobs
- Jitters durations by `×U(0.80, 1.25)` + small integer noise
- Generates fake job names: `TOY_0000`, `TOY_0001`, ...

**Current toy dataset** (`data/processed/toy_*`): confirmed stats:
- **20 jobs**, **7 machines** (`F2_1..F2_5, F1_2, F1_3`)
- Arrival range: 0 to 27,360 minutes
- Due date range: 17,122 to 89,469 minutes
- Num operations per job: 8–12 (varies by route template sampled)

### 7.3 CSV Schemas

Three normalised tables exported by `export_company_dataset_csv()`:

**`{prefix}_jobs.csv`**
```
job_idx | job_name | arrival_time | due_date
```

**`{prefix}_operations.csv`**
```
job_idx | job_name | op_index | op_id | duration
```

**`{prefix}_eligibility.csv`**
```
job_idx | job_name | op_index | op_id | machine_idx | machine_name | score
```

### 7.4 Actual Dataset Numbers

| Parameter | Toy dataset (`toy_`) | Real company (`company_fjsp_7m_`) |
|-----------|---------------------|----------------------------------|
| Jobs | 20 | ~thousands (from Excel) |
| Machines | 7 | 7 |
| Ops per job | 8–12 | Varies (route-dependent) |
| Time unit | minute | minute (training) / hour (preprocessing) |
| Arrival range | 0–27,360 min | Real timestamps |
| Due date range | 17,122–89,469 min | Real timestamps |
| Obs dim | 88 | Depends on job count |

---

## 8. Training Infrastructure

### 8.1 Dataset Selection Flag

`train.py` has a top-level flag:

```python
USE_TOY_DATASET = True    # True  → loads toy_jobs.csv etc (20 jobs)
                          # False → builds from dataset_scheduling.xlsx
TOY_PREFIX = "toy"
TIME_UNIT = "minute"
```

**Currently**: `USE_TOY_DATASET = True` — training uses the **20-job toy dataset**.

Same flag exists identically in `evaluate.py`.

### 8.2 Max Steps per Episode

```python
TIME_STEP = 10.0   # minutes per simulation step
max_due = max(dataset["due_dates"])
max_steps = max(int(max_due * 1.2 / TIME_STEP), 20_000)
```

For the toy dataset: `max(89469 * 1.2 / 10, 20000) = max(10736, 20000) = **20,000 steps**`.

### 8.3 Vectorised Environments

```python
n_envs = max(1, os.cpu_count() // 2)
vec_env = make_vec_env(_make_env, n_envs=n_envs)
```

Each env wraps `CompanyFJSSPEnv` with `ActionMasker(env, mask_fn)`.

### 8.4 Training Outputs per Run (`results/run_YYYYMMDD_HHMMSS/`)

| File/Dir | Contents |
|---|---|
| `manifest.txt` | Configuration snapshot (USE_TOY_DATASET, time_unit, max_steps, device) |
| `tb/` | TensorBoard event files (logs `mean_reward_last_step`, `num_timesteps` every 10k steps) |
| `episode_stats.csv` | Per-episode: `episode_id, env_index, total_reward, length, makespan, total_tardiness, utilization, deadlines_met, unscheduled_jobs, frac_deadlines_met` |
| `step_logs.csv` | Optional per-timestep logs (if `step_log` key present in `info`) |
| `ppo_company_fjsp.zip` | Saved MaskablePPO model (SB3 format) |
| `learning_curve_callback.png` | Episode rewards + 10-episode rolling mean |
| `ppo_eval_metrics.csv` | Post-training evaluation summary (mean/std makespan, tardiness, utilisation, deadlines_met_pct) |
| `info.txt` | Timesteps trained, time_unit, max_steps |

### 8.5 Training Timeline

| Period | Activity |
|--------|---------|
| Dec 7–8, 2025 | Toy environment validation, initial PPO experiments |
| Feb 20, 2026 | Real company data integration, reward weight tuning |
| Mar 16, 2026 | Experiment framework built; v1–v6 smoke-tested (1000 steps each) |
| Mar 17, 2026 | v7–v12 smoke-tested; classical _C heuristics added; priority weights updated to 60/25/15 |

~30+ legacy runs in `results/` + `results_company_real/`. 12 experiment-framework runs in `experiments/2026*/`.

### 8.6 Smoke-Test Results (1000 timesteps — heuristics meaningful, RL is noise)

All 28 policies evaluated with 60/25/15 composite weights. Top rankings:

| Rank | Policy | Makespan | Util | DL% | Score | Note |
|------|--------|----------|------|-----|-------|------|
| 1 | **SPT_C** | 75,450 | 0.870 | 65% | 0.915 | Classical shortest next-op duration — best by far |
| 2 | LPT | 82,700 | 0.848 | 10% | 0.595 | Adapted most-remaining-work |
| 3 | CR_C | 83,480 | 0.873 | 20% | 0.591 | Classical critical ratio wins over adapted CR (#24) |
| 4 | LIFO | 86,550 | 0.837 | 50% | 0.529 | |
| 5 | v9_high_gamma | 90,660 | 0.848 | 55% | 0.407 | Best RL at smoke-test level |
| 6 | v12_composite_tight | 90,230 | 0.849 | 45% | 0.406 | |

**Key insight**: `SPT_C` (next-op duration) outperforms adapted `SPT` (total remaining work) by a large margin — confirms the classical definition is more effective for makespan in this env. RL rankings at 1000 steps are meaningless; expect reversal after 4M timesteps.

---

## 9. Experiment Framework (`experiments/`)

> **Added March 2026.** Clean reproducible pipeline replacing ad-hoc `train.py` / `evaluate.py`.

### 9.1 Directory Structure

```
experiments/
  configs/
    base_config.yaml          - shared defaults (deep-merged by all variations)
    v1_ppo_default.yaml       - control (no overrides)
    v2_ppo_makespan_focus.yaml
    v3_ppo_deadline_focus.yaml
    v4_ppo_composite.yaml
    v5_ppo_large_network.yaml
    v6_ppo_entropy_boost.yaml
    v7_makespan_util_focus.yaml   - 60/25 target rewards (mine)
    v8_critical_path_extreme.yaml - extreme time penalties ablation
    v9_high_gamma_makespan.yaml   - gamma=0.999 for horizon awareness
    v10_wide_network_makespan.yaml- deep net + vf_coef=0.75
    v11_utilization_shaped.yaml   - util_bonus=80 + gamma=0.999 (Codex)
    v12_composite_tight.yaml      - funnel arch + gae_lambda=0.97 (Codex)
  run_experiment.py           - train one variation
  evaluate_variation.py       - evaluate trained model vs heuristics
  compare_all.py              - cross-experiment comparison
  composite_score.py          - composite score formula + plots
  gantt_utils.py              - Gantt chart helpers
  <YYYYMMDD_HHMMSS_vN_name>/  - one folder per run
    models/model.zip
    logs/episode_stats.csv  episode_rewards.csv  step_logs.csv  tb/
    metrics/metrics.csv  summary.json
    plots/  gantt/  config/  notes/
  compare_all_models/         - cross-experiment comparison outputs
```

### 9.2 Config System

`base_config.yaml` is deep-merged with each variation YAML — variation keys override only what they declare. Full merged config saved to `config/run_config.yaml` per run for reproducibility.

Key base config sections:

| Key | Purpose |
|-----|---------|
| `variation_name` | Folder suffix: `<YYYYMMDD_HHMMSS>_<variation_name>` |
| `ppo.*` | All MaskablePPO hyperparameters |
| `reward_weights.*` | Exact env key names (`job_complete`, `step_time_penalty`, etc.) |
| `eval.heuristics_original` | Explicit list of adapted rules to run — delete a line to exclude |
| `eval.heuristics_classical` | Explicit list of `_C` rules to run — delete a line to exclude |
| `composite_weights` | `makespan: 0.60, utilization: 0.25, deadlines: 0.15` |

**Per-rule control** (updated March 17): instead of boolean toggles, `base_config.yaml` now has two explicit lists. Delete a rule name from either list and it disappears from all evaluation outputs and comparison plots — no retraining, no code change required:

```yaml
eval:
  heuristics_original:        # adapted rules — delete lines to exclude
    - "FIFO"
    - "LPT"
    - "EDD"
    - "SLACK"
    # removed: SPT, LIFO, LDD, CR (redundant or poor performers)

  heuristics_classical:       # textbook _C rules — delete lines to exclude
    - "SPT_C"
    - "CR_C"
    # removed: EDD_C, LDD_C (identical to EDD/LDD), LPT_C (worst), SLACK_C (degenerates)
```

Both `evaluate_variation.py` and `compare_all.py` read these lists live from `base_config.yaml` at run time. Legacy boolean keys (`eval_original_variants`, `eval_classical_variants`) are still honoured as fallback for old saved `run_config.yaml` files.

### 9.3 Composite Score (updated March 2026)

**Weights**: 60% makespan, 25% utilization, 15% deadlines met.
*(Changed from original 50/30/20 — deadline weight reduced because real dataset deadlines are generous and almost always satisfied.)*

```
makespan_norm = (max_ms - ms) / max(max_ms - min_ms, 1e-9)   # best=1.0, worst=0.0
composite     = 0.60 * makespan_norm + 0.25 * utilization + 0.15 * frac_deadlines_met
```

**Weighted rank bar chart** (`weighted_rank_scores.png`): each bar = `0.60×rank_makespan + 0.25×rank_util + 0.15×rank_deadlines`. Lower = better.

### 9.4 Commands

```bash
# Train one variation (4M timesteps)
D:\UROP\djssp\Scripts\python.exe experiments/run_experiment.py --config experiments/configs/v7_makespan_util_focus.yaml

# Evaluate one experiment (uses live base_config.yaml toggles)
D:\UROP\djssp\Scripts\python.exe experiments/evaluate_variation.py --experiment experiments/<YYYYMMDD_vN_name>/

# Force-enable classical variants from CLI (overrides base_config toggle)
D:\UROP\djssp\Scripts\python.exe experiments/evaluate_variation.py --experiment experiments/<folder>/ --classical

# Skip original adapted heuristics (only _C + RL)
D:\UROP\djssp\Scripts\python.exe experiments/evaluate_variation.py --experiment experiments/<folder>/ --no-original --classical

# Re-evaluate ALL experiments
for d in experiments/2026*/; do
  D:\UROP\djssp\Scripts\python.exe experiments/evaluate_variation.py --experiment "$d"
done

# Cross-experiment comparison → outputs to experiments/compare_all_models/
D:\UROP\djssp\Scripts\python.exe experiments/compare_all.py --root experiments/
```

---

## 11. Evaluation & Baselines

> **Active pipeline**: `experiments/evaluate_variation.py` and `experiments/compare_all.py` (see Section 9).
> Legacy `evaluate.py` / `train.py` still present for reference but not used for new runs.

### 11.1 Protocol

`evaluate_variation.py` loads the saved model from `models/model.zip`, reads dataset/env config from `config/run_config.yaml`, and reads **eval toggles** from the live `base_config.yaml`. All policies run on the same `CompanyFJSSPEnv` with identical seed/arrivals.

### 11.2 Heuristic Dispatching Rules

The framework provides **8 adapted rules** (original implementation) plus **8 classical variants** (suffix `_C`), controlled by `eval_original_variants` and `eval_classical_variants` in `base_config.yaml`.

**Adapted (original) rules** — job-level priority over the dispatchable set; machine assigned by env:

| Rule | Key | Direction | Notes |
|------|-----|-----------|-------|
| `FIFO` | `arrival_times[j]` | min | Release-time FIFO, NOT queue-entry FIFO |
| `LIFO` | `arrival_times[j]` | max | Release-time LIFO |
| `SPT` | `_remaining_work(j)` | min | **Total remaining work**, NOT next-op duration |
| `LPT` | `_remaining_work(j)` | max | **Total remaining work**, NOT next-op duration |
| `EDD` | `due_dates[j]` | min | Matches classical definition |
| `LDD` | `due_dates[j]` | max | Matches classical definition |
| `SLACK` | `due - t - remaining_work(j)` | min | Uses total remaining work |
| `CR` | `(due - t) / remaining_work(j)` | min | Uses total remaining work |

**Classical variants (`_C`)** — textbook implementations:

| Rule | Key | Difference from adapted |
|------|-----|------------------------|
| `FIFO_C` | op ready-queue entry time | Tracks when this operation became dispatchable (not job arrival) |
| `LIFO_C` | op ready-queue entry time | Same, reversed |
| `SPT_C` | `job_ops[j][next_op]["duration"]` | **Next-operation duration only** |
| `LPT_C` | `job_ops[j][next_op]["duration"]` | **Next-operation duration only** |
| `EDD_C` | `due_dates[j]` | Identical to EDD |
| `LDD_C` | `due_dates[j]` | Identical to LDD |
| `SLACK_C` | `due - t - next_op_duration` | Uses next-op duration |
| `CR_C` | `(due - t) / next_op_duration` | Uses next-op duration |

> See `reports/heuristic_handoff_20260317.md` for full analysis of adapted vs classical differences.

When no jobs are dispatchable, all heuristics issue Action `N` (no-op).

Control via `base_config.yaml` (takes effect immediately, no retraining needed):
```yaml
eval:
  eval_original_variants: true    # toggle adapted rules on/off
  eval_classical_variants: true   # toggle _C rules on/off
```

### 11.3 Evaluation Metrics

| Metric | Formula | Direction |
|--------|---------|-----------|
| `makespan` | `max(C_j)` or `current_time` if unfinished | ↓ |
| `total_tardiness` | $\sum_j \max(0, C_j - D_j)$ | ↓ |
| `utilization` | `busy_time / (makespan × M)` | ↑ |
| `frac_deadlines_met` | `deadlines_met / N` | ↑ |

### 11.4 Evaluation Outputs (per-experiment folder)

| File | Description |
|------|-------------|
| `metrics/metrics.csv` | One row per policy: makespan, tardiness, util, deadlines, composite_score |
| `metrics/summary.json` | Full results dict with composite weights |
| `gantt/<policy>_overview.png` | Gantt chart — first 5 jobs (readable) |
| `gantt/<policy>_detailed.png` | Gantt chart — all 20 jobs |
| `plots/bar_{metric}.png` | Bar chart per metric, all policies |
| `plots/scatter_makespan_vs_tardiness.png` | Trade-off scatter |
| `plots/ratio_vs_rl.png` | Heuristic / RL ratio; >1 = RL wins |
| `plots/rank_heatmap.png` | Ranking matrix (1=best) |
| `plots/composite_scores.png` | Composite score bar chart (60/25/15) |
| `logs/episode_rewards.csv` | `episode_id, total_reward, length` per training episode |
| `logs/episode_stats.csv` | Full per-episode stats (all metrics) |

**Cross-experiment outputs** (`experiments/compare_all_models/`):

| File | Description |
|------|-------------|
| `all_metrics.csv` | All 28 policies (12 RL + 16 heuristics), all raw metrics |
| `composite_scores.csv` | With makespan_norm, composite_score, composite_rank |
| `summary_table.csv` | Sorted by composite rank |
| `plots/weighted_rank_scores.png` | Weighted rank bar: `0.60×rank_ms + 0.25×rank_util + 0.15×rank_dl` (lower=better) |
| `plots/composite_scores.png` | Composite score bar |
| `plots/rank_heatmap.png` | Full ranking matrix all 28 policies |
| `plots/bar_*.png` | Per-metric bars |

---

## 12. Key Design Decisions & Known Quirks

### Notable Implementation Details

| Issue | What the code does |
|-------|-------------------|
| **Machine selection** | First idle eligible machine in list order (not argmin start time) |
| **No-op validity** | Only valid when zero jobs are dispatchable, not always |
| **`step_time_penalty` scaling** | Applied UNSCALED (-10/step), all other step rewards use `dt_norm`; **log now matches** (was logging scaled value before Mar 17 fix) |
| **`reward_weights` in train.py** | **FIXED in experiment framework**: old `train.py` passed wrong keys; new YAML configs use exact env key names — all variations correctly override defaults |
| **Observation space bounds** | **FIXED**: `low=-1.0` now declared (was `0.0`); `slack_norm = tanh(...)` returns negative for overdue jobs |
| **`compute_metrics` for unfinished jobs** | Uses `current_time` as proxy completion (not a penalty) — can falsely count as on-time if `current_time < due_date` |
| **Eligibility filtering** | Done in `__init__`, not at step time — score-2 only, with fallback |
| **Gantt filter in evaluate.py** | Only first 5 jobs and time 0–20,000 plotted (not all) |
| **`USE_TOY_DATASET = True`** | Both train and evaluate currently use 20-job toy data |
| **n_jobs** in toy generator | Default is `n_jobs=20` (hard-coded in `generate_toy_company_csv.main()`) |

### Environment Bug Fixes Applied (March 17, 2026)

Seven correctness bugs fixed in `env_company.py` after cross-validated analysis (independent + Codex agent):

| # | Bug | Severity | Status |
|---|-----|----------|--------|
| 1 | **Double-dispatch**: `_get_dispatchable_jobs()` allowed re-dispatching a job whose op was already running on a machine → same op on two machines, `job_next_op` double-incremented, operation skipped | Critical | **Fixed** |
| 2 | **`_gantt_log` not reset** between episodes → Gantt log accumulated across all training episodes | High | **Fixed** |
| 3 | **Multi-job completion reward**: only the last job completing in a time step got its reward; earlier jobs silently received nothing | High | **Fixed** |
| 4 | **Float time drift**: completion threshold `1e-9` too tight for IEEE-754 float subtraction → ops could run one step longer than specified | Medium | **Fixed** (widened to `1e-6`) |
| 5 | **`step_time_penalty` logged at wrong scale**: logged as `dt_norm × penalty ≈ −0.001` per step, applied as flat `−10.0` | Medium | **Fixed** |
| 6 | **`_remaining_work` overstated**: included full duration of in-progress op; now subtracts elapsed time for the currently-running operation | Medium | **Fixed** |
| 7 | **`reward_terms["total"]` excluded `final_unscheduled`** on terminal steps | Low | **Fixed** |

> Previous smoke-test results (v1–v12 at 1000 steps) were generated with all these bugs present. Heuristic rankings remain valid (heuristics are not affected by bugs 3, 5, 6, 7); RL reward magnitude and training signal were distorted. Re-evaluation after full 4M-step training is required for valid comparison.

### Reward Engineering Rationale

- `gamma = 0.999` needed because episodes can span 20,000 steps; with `gamma=0.99` a reward at step 100 from now would be discounted to 0.99^100 ≈ 0.37, making distant rewards nearly invisible
- Dense `tardiness_step` (-120) is the dominant training signal by design — prevents agent from ignoring due dates
- `final_unscheduled` (-100 per job) ensures the agent never deliberately ignores jobs

### Action Masking Subtlety

The no-op is masked out when any job is dispatchable. This means **the agent must always make progress** when possible. The no-op is only available when the environment is in a "waiting" state (all machines busy, no arrivals yet). This prevents the agent from learning to stall for better opportunities.

---

## 13. Dependencies

### Active (`Dynamic-Job-Shop-Scheduling-RL/req.txt`)

```
gymnasium==0.29.1          # Farama gymnasium (not gym)
stable-baselines3==2.3.2
sb3_contrib                # MaskablePPO
torch==2.2.2
numpy==1.26.4
pandas==2.3.3
matplotlib==3.7.5
openpyxl==3.1.5            # Excel reading
imageio==0.35.1
tensorboard==2.14.0
tqdm
rich
```

### Legacy Root (`req.txt`)
```
torch==1.13.1
gym==0.21.0
stable-baselines3==1.7.0
numpy
```

### Virtual Environments

| Env | Path | Python | PyTorch | Notes |
|-----|------|--------|---------|-------|
| **gitdjssp** (active) | `D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\` | 3.12 | 2.10+cu128 | RTX 5070 Ti (sm_120 Blackwell) — use this for all training |
| djssp (legacy) | `D:\UROP\djssp\` | 3.8.10 | 2.2.2+cu118 | Old env — do not use for new runs |

**Active Python executable**: `D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe`

Packages installed in gitdjssp: `sb3_contrib==2.7.1`, `stable_baselines3==2.7.1`, `gymnasium==0.29.1`, `numpy==1.26.4`, `pandas==2.3.3`, `matplotlib==3.10.8`, `pyyaml`, `openpyxl`

---

## 14. Glossary

| Term | Definition |
|------|-----------|
| **JSSP** | Job Shop Scheduling Problem |
| **FJSSP** | Flexible JSSP — multiple eligible machines per operation |
| **DJSSP** | Dynamic JSSP — jobs arrive over time |
| **PPO** | Proximal Policy Optimization (on-policy RL algorithm) |
| **MaskablePPO** | PPO variant from `sb3_contrib` that masks invalid actions before softmax |
| **GAE** | Generalised Advantage Estimation |
| **Makespan** ($C_{\max}$) | Time from schedule start to last job completion |
| **Tardiness** ($T_j$) | $\max(0, C_j - D_j)$ — how late job $j$ finished |
| **Utilisation** ($U$) | `busy_machine_time / (makespan × M)` |
| **Action masking** | Binary mask applied to logits before softmax to zero illegal actions |
| **dt_norm** | `time_step / τ` — per-step normalisation factor used in reward shaping |
| **τ (tau)** | Time scale: `max(max_due_date, sum_of_all_work, 1.0)` |
| **Dispatchable** | A job is dispatchable if it has arrived, has remaining ops, and at least one eligible machine is idle |
| **Eligibility score** | `2` = machine always capable; `1` = occasional; `0` = not eligible |
| **Gantt chart** | Visual schedule showing operations on machines over time |
| **FIFO/SPT/EDD/SLACK/CR** | Adapted dispatching rules used as baselines (see Section 10 + handoff doc) |
| **FIFO_C/SPT_C/…** | Classical textbook variants of the above; enabled via `eval_classical_variants: true` |
| **t0** | Reference datetime (earliest arrival in the job sheet) — all times relative to this |
| **USE_TOY_DATASET** | Flag in `train.py` and `evaluate.py`; `True` = use 20-job toy data |

---

---

## 15. Dispatching Rules Assessment — Which to Discard

> **Assessed March 17, 2026.** Cross-validated with Codex agent analysis.

### 15.1 Summary Table — All 16 Rules

| Rule | Type | Primary Key | Smoke Rank | Quality | Verdict |
|------|------|-------------|-----------|---------|---------|
| **SPT_C** | Classical | next-op duration | **#1** (0.915) | Excellent | **Keep — gold standard** |
| **CR_C** | Classical | `(due-t)/next_op_dur` | #3 (0.591) | Good | **Keep** |
| **FIFO** | Adapted | arrival_time | Mid | Standard | **Keep — important reference** |
| **LIFO** | Adapted | arrival_time (max) | #4 | Useful | **Keep** |
| **LPT** | Adapted | total remaining work (max) | #2 | Strong | **Keep** |
| **SPT** | Adapted | total remaining work (min) | Low | Weak | **Keep — shows SPT_C superiority** |
| **EDD** | Adapted | due date (min) | Mid | Standard | **Keep** |
| **SLACK** | Adapted | `due-t-remaining_work` | Mid | Standard | **Keep** |
| **CR** | Adapted | `(due-t)/remaining_work` | #24 | Weak | Keep (shows adapted CR is poor) |
| **FIFO_C** | Classical | op-queue-entry time | Mid | Comparable to FIFO | Keep |
| **LPT_C** | Classical | next-op duration (max) | **Last** | **Worst** | **Discard candidate** |
| **EDD_C** | Classical | due date (min) | Same as EDD | **Duplicate** | **Discard — identical to EDD** |
| **LDD** | Adapted | due date (max) | Low | Marginal | **Discard candidate** |
| **LDD_C** | Classical | due date (max) | Same as LDD | **Duplicate** | **Discard — identical to LDD** |
| **SLACK_C** | Classical | `due-t-next_op_dur` | Low | Degenerates to EDD | **Discard candidate** |
| **LIFO_C** | Classical | op-queue-entry time (max) | Low | Marginal | **Discard candidate** |

### 15.2 Rules to Discard (and Why)

| Rule | Reason to Discard | Codex Agreement |
|------|-------------------|-----------------|
| **EDD_C** | Mathematically identical to EDD — same key (`due_dates[j]`), same direction. No new information. | ✓ Both analyses agree |
| **LDD_C** | Mathematically identical to LDD. Pure duplicate. | ✓ Both analyses agree |
| **LPT_C** | Consistently worst performer in smoke tests. Longest *next-op* = prioritise jobs that will block machines longest. No theoretical justification for this; actively harmful. | ✓ Both analyses agree |
| **SLACK_C** | When next-op duration << total slack (which is always the case on toy data with slack/work ≈ 10×), `SLACK_C ≈ EDD`. Collapses to duplicate of EDD in this problem regime. | ✓ Both analyses agree |

**Additional candidates (Codex flagged, less certain):**
- **LDD**: Latest due date first — inverted EDD, causes deliberate tardiness. Useful only as a pathological lower bound. Consider keeping as "sanity-check worst case".
- **CR** (adapted): Uses total remaining work which decouples from current machine decision; smoke test rank #24. Codex suggested it may be worth retaining as a comparison point against CR_C.

### 15.3 Recommended Minimal Baseline Set (12 → 6–9 rules)

**Conservative recommendation (6 rules)** — maximum information, no duplicates:
```
SPT_C   CR_C   EDD   SLACK   LPT   FIFO
```

**Codex recommended set (9 rules)** — retains LIFO + SPT comparison + CR contrast:
```
FIFO   LIFO   SPT   SPT_C   LPT   EDD   CR_C   CR   SLACK
```

**For final evaluation runs**: set `eval_classical_variants: true` (for SPT_C, CR_C) and `eval_original_variants: true` (for FIFO, LPT, EDD, SLACK), then manually list only the desired rules in `base_config.yaml` `heuristics:` array. EDD_C, LDD_C, LPT_C, SLACK_C can be removed from that list.

---

## 16. Toy Dataset Assessment — Biases Favouring Heuristics

> **Assessed March 17, 2026.** `generate_toy_company_csv.py` analysed independently + Codex agent (full agreement on all 7 biases).

### 16.1 Key Statistics (confirmed from source code)

| Property | Value | Impact |
|----------|-------|--------|
| Number of jobs | 20 | Too few — queue depth rarely > 3 |
| Number of machines | 7 | Fixed, no bottleneck generated |
| slack / total_work ratio | avg **10.1×** | Deadlines trivially easy → EDD/CR almost useless as differentiators |
| Jobs with slack > 5× work | 11 / 20 | Majority have "vacation-slack" |
| Distinct arrival times | **6** | Very few simultaneous-arrival events |
| Distinct route templates | **7** | Balanced load across machines |
| Random seed | **42** (fixed) | **Single instance** — no statistical validity |
| Duration jitter | U(0.80, 1.25) multiplicative | Symmetric, no heavy tail, no surprises |

### 16.2 The 7 Structural Biases (confirmed by both analyses)

| # | Bias | Code Location | Effect |
|---|------|--------------|--------|
| 1 | **i.i.d. arrival bootstrap** — each job's arrival drawn independently | `line 90` | No bursts, even spacing, queues rarely form |
| 2 | **Slack independent of work** — slack drawn before route selected | `lines 93-97` | CR/EDD work even without real timing logic |
| 3 | **Uniform route selection** — law-of-large-numbers balances load | `line 100` | No bottleneck machine; SPT/LPT near-optimal |
| 4 | **Narrow symmetric jitter** — U(0.80, 1.25), no tail | `lines 105-109` | No unexpected long-blocking operations |
| 5 | **Only 20 jobs** — queue depth shallow | `line 139` | Greedy is nearly globally optimal |
| 6 | **Fixed seed=42** — one instance only | `lines 56-57` | No variance measurement, no statistical test |
| 7 | **No stochastic disruptions** — static job set | `entire fn` | Tests static FJSP, not dynamic FJSP |

**Net effect**: The current toy dataset is a *structurally easy* instance. Heuristics can be competitive with (or superior to) RL not because RL is weak, but because the problem lacks the structural complexity that makes learned policies necessary.

### 16.3 What Is Realistic vs Not

**Realistic** (kept from real data):
- Operation sequence ordering within a job (real `dati_cicli`)
- Machine eligibility per operation (real `operazioni_macchine`)
- Duration magnitudes (base values from real measured minutes)

**Not realistic**:
- Arrival process: i.i.d. bootstrap strips all temporal correlation (shifts, batches, campaigns)
- Due date structure: no capacity-aware negotiation; no shipping-window clustering
- Job mix: uniform draw; real shop likely has Pareto distribution over part types
- No machine breakdowns, setup times, or sequence-dependent constraints

### 16.4 Recommended Fixes to `generate_toy_company_csv.py`

| Priority | Fix | Code change |
|----------|-----|-------------|
| **HIGH** | **Multi-instance, multi-seed**: generate 40 instances (`n_jobs ∈ {20,40,60,100}`, 10 seeds each) | Replace single `main()` call with outer loops |
| **HIGH** | **Couple slack to work content**: `slack = alpha * total_work`, `alpha ~ U(1.05, 2.0)` | Compute route first, then set slack proportional to `sum(op durations)` |
| **HIGH** | **Lognormal duration jitter**: `factor ~ Lognormal(0, 0.35)`, `max(0.5, factor)` | Replace `random.uniform(*jitter)` with `np.random.lognormal(0, 0.35)` |
| **MEDIUM** | **Bursty arrivals**: Poisson inter-arrival gaps + p=0.15 batch clusters | Replace `np.random.choice(arr_real)` with structured arrival generator |
| **MEDIUM** | **Introduce bottleneck**: bias route selection toward already-overloaded machines | Check `machine_load` before accepting template |
| **LOW** | **Machine breakdowns**: inject unavailability intervals (p=0.3/machine, duration U(5,30)) | Pass breakdown schedule to env as `machine_unavailable` parameter |
| **LOW** | **Urgency tiers**: 80% normal (alpha 1.5–2.0), 20% urgent (alpha 1.05–1.15) | Add `job_class` assignment before slack computation |

---

## 17. RL vs Heuristics — Strategies to Demonstrate RL Superiority

> **Assessed March 17, 2026.** Both my analysis and Codex agent converge on the same priority ordering.

### 17.1 Why the Current Setup Understates RL Advantage

The toy dataset (Section 16) is structurally biased toward heuristics. The RL agent is evaluated on:
- A single fixed instance (seed=42) — no variance, no statistical significance
- 20 jobs with shallow queue depth — greedy rules are near-optimal
- Generous deadlines (10× slack) — deadline-aware rules see little signal
- Static job set — no mid-episode disruptions where RL adaptation matters

All of these conditions favour simple greedy heuristics.

### 17.2 Recommended Experiments (Priority Order)

| Priority | Experiment | What It Shows | Effort |
|----------|-----------|---------------|--------|
| **1** | **Scaling sweep** (`n_jobs ∈ {20, 40, 80, 160}`) | RL advantage grows super-linearly with problem size; heuristics plateau | Medium — retrain per size |
| **2** | **500+ random instances** (multiple seeds) | Statistical significance: mean ± std, p-values; eliminates single-instance bias | Low — evaluation only |
| **3** | **Adversarial / tight-deadline instances** | Instances where slack/work ≈ 1.05–1.5× — where RL must genuinely reason about urgency vs throughput | Low — just change generation params |
| **4** | **Real company dataset** | Ultimate validation; removes toy-data criticism entirely | Medium — requires real Excel data run |
| **5** | **Machine breakdown / disruption evaluation** | Tests RL adaptability — heuristics are completely static | Medium — env modification needed |
| **6** | **OOD generalization** | Train on 20-job, test on 40-job; RL agent feature norms scale; heuristics are feature-free | Low — evaluate only |

### 17.3 The Scaling Argument (Theoretical Basis)

For $N$ jobs on $M$ machines, the dispatching decision tree has branching factor $N$ at each step and depth $O(N \cdot \text{ops\_per\_job})$. The global optimum requires search over $O(N^{N \cdot k})$ candidates. Greedy heuristics approximate this with a single linear pass; RL learns a compact policy that encodes non-local state. As $N$ grows:

- Greedy error accumulates: each locally-optimal choice can cascade into globally-suboptimal conflicts
- RL policy quality degrades more slowly because the 88-dimensional observation contains global state (all machines, all job slack values simultaneously)
- Queue depth scales with $N/M$ — at 20 jobs and 7 machines, average queue depth ≈ 2.9; at 80 jobs, ≈ 11.4; at this depth, greedy choice of "shortest next-op" frequently delays a job whose downstream operations will block a machine for much longer

### 17.4 The Tight-Deadline Argument

On the current toy data, `slack/work ≈ 10×` — a job with 500 minutes of work has a deadline 5,000 minutes away. EDD has almost no signal because all jobs are far from their deadlines. CR_C reduces to SPT_C when all critical ratios are large. In this regime, heuristics based on deadline features are effectively random.

Generating instances where `slack/work ≈ 1.1–1.5×` forces real deadline reasoning:
- An RL agent learns: *"job B has 3 remaining operations on machine F2_3; if I dispatch job A on F2_3 now, B will miss its deadline"*
- EDD would dispatch B regardless; CR_C would dispatch the job with shortest next-op
- Neither heuristic reasons about downstream machine conflicts

### 17.5 Summary Recommendation

**Minimum viable demonstration** (high impact, low effort):
1. Set `slack_alpha ~ U(1.1, 1.8)` in `generate_toy_company_csv.py` — generates tight-deadline instances
2. Generate 50 random seeds (change `seed` in a loop)
3. Evaluate all 12 RL variants + SPT_C + CR_C + FIFO on all 50 instances
4. Report: mean composite score ± std, % instances where RL > best heuristic, Mann-Whitney U p-value

This single experiment would produce statistically valid results and likely demonstrate a clear RL advantage under realistic deadline pressure.

---

---

## 18. Paper Reference — Alcamo, Bruno, Giovenali (Politecnico di Torino)

> This is the **academic base paper** for the project. Read it at `D:\UROP\data\A+reinforcement+learning+algorithm+for+Dynamic+Job+Shop+Scheduling.pdf`.

### 18.1 What the Paper Does

- **Algorithm**: MaskablePPO (SB3) — same as this project
- **State**: Binary matrix (rows=machines, cols=jobs, 1=job in queue) — simpler than our feature vector
- **Action space**: Only 3 meta-strategies: (1) LPT, (2) most remaining ops, (3) no-op — NOT direct job selection
- **Reward**: Scheduled-area reward (Tassel 2021): `R(s,a) = p_aj − Σ idle_time_introduced`
- **Baselines**: FIFO, LPT, SPT only — 3 heuristics
- **Problem sizes**: ft06 (6×6), ft10 (10×10) OR dynamic ft06 (6 machines, 6 job types)
- **Result**: RL outperforms all 3 heuristics in all dynamic scenarios

### 18.2 Our Project vs the Paper

| Aspect | Paper | This project |
|--------|-------|-------------|
| State representation | Binary matrix | Rich feature vector (remaining work, slack, due, arrival, util) — **better** |
| Action space | 3 meta-strategies | Direct job selection (N+1 actions) — **more expressive** |
| Reward | Scheduled area only | Dense multi-term (tardiness, deadlines, utilization) — **richer** |
| Baselines | 3 heuristics | Up to 16 heuristics (8 original + 8 classical) — **stronger** |
| Dataset | Academic benchmarks | Real company + synthetic multi-size — **more realistic** |
| Objective | Makespan only | Composite 60/25/15 — **multi-objective** |

### 18.3 v13 — Tassel Reward (Added March 18)

`experiments/configs/v13_tassel_reward.yaml` replicates the paper's reward function:

```yaml
machine_idle:        -100.0   # penalise idle time = Tassel "empty(s,s')" term
utilization_bonus:    100.0   # reward busy machines = Tassel "p_aj" term
job_complete:           0.0   # no per-job reward
job_on_time:            0.0   # no deadline reward
tardiness_step:         0.0   # no tardiness
final_unscheduled:   -500.0   # prevent lazy policy
```

If v13 scores below v7-v12: our richer reward is better. If v13 is competitive: simpler rewards generalise better to scaled instances.

---

## 19. Benchmark Bank & Overnight Training Pipeline

> **Added March 18, 2026.** Full training → evaluation → scaling-eval pipeline.

### 19.1 Benchmark Bank

Pre-generated fixed datasets for statistically valid evaluation across seeds and sizes.

| Size | Seeds | Instances | Location |
|------|-------|-----------|----------|
| n=20 | 0–4 | 5 | `data/benchmark/n20/seed_00/` … `seed_04/` |
| n=50 | 0–4 | 5 | `data/benchmark/n50/seed_00/` … `seed_04/` |
| n=100 | 0–4 | 5 | `data/benchmark/n100/seed_00/` … `seed_04/` |

Generate with: `python generate_toy_company_csv.py` (writes both legacy `toy_*.csv` and full benchmark bank).

**Key file**: `data/benchmark/bank_manifest.json` — records all instance metadata.

### 19.2 Training Strategy

Train **13 models** (v1–v13), each once on `n=20, seed=0`. Then evaluate each on all 15 benchmark instances via `run_scaling_eval.py`. Total: 13 training runs + 13×15=195 evaluation runs.

| Variation | Config | Expected strength | Rationale |
|-----------|--------|------------------|-----------|
| v8 | `v8_critical_path_extreme.yaml` | High | Extreme time/idle penalties → critical path awareness |
| v9 | `v9_high_gamma_makespan.yaml` | High | v7 rewards + gamma=0.999 → horizon-aware |
| v12 | `v12_composite_tight.yaml` | High | Funnel [512,256,128] + Tanh + gae_lambda=0.97 |
| v7 | `v7_makespan_util_focus.yaml` | Good | Balanced makespan+util shaping |
| v11 | `v11_utilization_shaped.yaml` | Good | Util-dominant + gamma=0.999 + n_steps=4096 |
| v10 | `v10_wide_network_makespan.yaml` | Good | Deep [256,256,256] + vf_coef=0.75 |
| v13 | `v13_tassel_reward.yaml` | Good | Paper reward — pure scheduled area |
| v4 | `v4_ppo_composite.yaml` | Medium | Composite reward shaping |
| v3 | `v3_ppo_deadline_focus.yaml` | Medium | Deadline-focused reward |
| v2 | `v2_ppo_makespan_focus.yaml` | Medium | Makespan-focused reward |
| v1 | `v1_ppo_default.yaml` | Baseline | Default PPO hyperparameters |
| v5 | `v5_ppo_large_network.yaml` | Low | May overfit small dataset |
| v6 | `v6_ppo_entropy_boost.yaml` | Low | Over-exploration risk |

### 19.3 Complete Console Command Sequence

Run from `D:\UROP\Dynamic-Job-Shop-Scheduling-RL\`:

```bash
PYTHON=D:/UROP/Dynamic-Job-Shop-Scheduling-RL/gitdjssp/Scripts/python.exe

# Step 0 — Generate benchmark bank (once, ~1 min)
$PYTHON generate_toy_company_csv.py

# Step 1 — Train all 13 models on n=20, seed=0 (overnight, best→worst order)
$PYTHON experiments/run_experiment.py --config experiments/configs/v8_critical_path_extreme.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v9_high_gamma_makespan.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v12_composite_tight.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v7_makespan_util_focus.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v11_utilization_shaped.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v10_wide_network_makespan.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v13_tassel_reward.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v4_ppo_composite.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v3_ppo_deadline_focus.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v2_ppo_makespan_focus.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v1_ppo_default.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v5_ppo_large_network.yaml --n_jobs 20 --seed 0
$PYTHON experiments/run_experiment.py --config experiments/configs/v6_ppo_entropy_boost.yaml --n_jobs 20 --seed 0

# Step 2 — After training: add model names to base_config.yaml scaling_eval.models_to_eval
# Step 3 — Scaling eval: each model × 15 benchmark instances (3 sizes × 5 seeds)
$PYTHON experiments/run_scaling_eval.py

# Step 4 — Per-model evaluation + compare-all plots
for d in experiments/2026*/; do $PYTHON experiments/evaluate_variation.py --experiment "$d"; done
$PYTHON experiments/compare_all.py --root experiments/
```

### 19.4 New Scripts Added

| Script | Purpose |
|--------|---------|
| `experiments/run_scaling_eval.py` | Overnight multi-seed/multi-size evaluation. Reads `scaling_eval` block from `base_config.yaml`. Saves to `experiments/scaling_results/`. Has resume support (skips cached seeds). |

### 19.5 New Plots Added (per model)

`evaluate_variation.py` now saves 3 scatter plots in each model's `plots/` folder:

| File | Axes | Best region |
|------|------|------------|
| `scatter_makespan_vs_tardiness.png` | makespan vs total tardiness | lower-left |
| `scatter_makespan_vs_utilization.png` | makespan vs utilization | lower-right |
| `scatter_makespan_vs_deadlines.png` | makespan vs frac deadlines met | lower-right |

### 19.6 Dataset Generation Analysis Findings

Second full analysis of `generate_toy_company_csv.py` (my analysis + Codex agent, March 18):

**Critical bug found and fixed**: `N_SEEDS = 5` in code but `n_seeds: 10` in `base_config.yaml` → would crash at seed 5. Fixed to `n_seeds: 5`.

**Additional bugs (Codex found)**:
- Missing-machine latent bug: if any machine has zero eligibility rows in a small instance, `load_company_dataset_from_csv` drops it from `machine_names`, corrupting all machine indices
- `_meta` key lost on CSV roundtrip (not saved to CSV, only to `meta.json`) — any code accessing `ds["_meta"]` after loading will crash
- Emoji encoding crash on Windows cp1252 terminals — fixed to plain ASCII

**Load factor scaling**: proportional formula correct in expectation but ±20% per-instance variance for n=20. For n=50/100 the scaling holds better. The `max()` floor clause can silently lower load below 0.70 — no warning issued.

*Generated: March 16, 2026*

**Changelog:**
- **Mar 17 (1)**: Added experiment framework (Section 9), classical `_C` heuristic variants, composite weights 60/25/15, v7–v12 configs, `compare_all_models/` output dir, obs-space `low=-1.0` fix, reward-weight key fix.
- **Mar 17 (2)**: Fixed `variation_name="base"` bug in v7–v12 folders (configs were trained before `variation_name` field was added). Patched `variation_name.txt` + `run_config.yaml` in all 12 experiment folders.
- **Mar 17 (3)**: Eval toggles (`eval_original_variants`, `eval_classical_variants`) now read from live `base_config.yaml` — changes take effect immediately without retraining. Both default to `true`. Added `--classical`/`--no-original` CLI flags to `evaluate_variation.py`.
- **Mar 17 (4)**: Added Sections 15–17: dispatching rules discard analysis, toy dataset bias assessment (7 biases, 7 fixes), and RL superiority demonstration strategies. Cross-validated with 3 independent Codex agents.
- **Mar 17 (5)**: Fixed 7 correctness bugs in `env_company.py` (double-dispatch, gantt log reset, multi-completion reward, float drift, step_time_penalty logging, remaining_work, terminal reward total). Updated Section 12 with bug fix table. All fixes cross-validated with Codex agent re-analysis — confirmed applied. Codex round-2 raised 3 additional items: double-lateness terms (intentional design, documented), no-op penalty (false positive — handled by MaskablePPO mask), busy_frac before completions (false positive — correct by design). No new actionable bugs remain.
- **Mar 17 (6)**: Config heuristic control upgraded from boolean toggles to explicit per-rule lists (`heuristics_original`, `heuristics_classical` in `base_config.yaml`). Delete a rule name → excluded from all outputs immediately. Updated Section 9.2. Scaling-for-publication strategy documented (Section 17 + Section 9.4 commands).
- **Mar 18 (1)**: Three-way pipeline + config audit (Claude Explore agent + Claude Codex-style agent + direct verification). Found 2 silent YAML nesting bugs (v9, v10), `n_epochs` never read, `ortho_init`/`activation_fn` unimplemented, training fully non-reproducible, v9/v10 phantom duplicates of v7. Added Section 20 with all findings and fixes.
- **Mar 18 (2)**: Applied all fixes. See Section 20 for full change log.

---

## Section 20 — Pipeline & Config Audit + Fixes (March 18, 2026)

### 20.1 Audit Summary

Three-way independent audit (Claude Explore agent + Claude deep-reader agent + direct code verification) of `experiments/run_experiment.py`, `early_stopping_callback.py`, and all v1–v13 YAML configs. Two main hypotheses tested:

**Hypothesis 1 — Dataset seed fixed but PPO training randomness uncontrolled: CONFIRMED**

| Source | Status before fix |
|---|---|
| Dataset (toy CSV) | Deterministic — always same file |
| Env transitions | Deterministic — no stochastic branching |
| `CompanyFJSSPEnv._rng` | Dead code — initialized but never used |
| `np.random`, `torch`, `random` | NOT seeded |
| `make_vec_env(seed=...)` | NOT passed |
| `MaskablePPO(seed=...)` | NOT passed |
| Early stopping eval | Nominally seeded per episode, but env `_rng` dead |

Result: every training run produces a statistically independent model. Single-run comparisons have unknown variance.

**Hypothesis 2 — Config bugs, ineffective overrides, phantom duplicates: STRONGLY CONFIRMED**

| Bug | Configs | Effect |
|---|---|---|
| PPO keys at YAML root (not under `ppo:`) | v9, v10 | All PPO overrides silently ignored |
| `n_epochs` never read by run_experiment.py | v10 | n_epochs=15 always ignored regardless of YAML placement |
| `ortho_init` and `activation_fn` not in `policy_kwargs` | v10, v12 | Features claimed but not implemented |
| v9 effective = v7 | v9 | Same reward weights + PPO overrides dropped → phantom duplicate |
| v10 effective = v7 | v10 | Same reward weights + PPO overrides dropped → phantom duplicate |
| Confounded multi-variable configs | v5, v6, v11, v12 | Cannot isolate which change drives outcome |

### 20.2 Fixes Applied

#### run_experiment.py

| Change | Location | Detail |
|---|---|---|
| Added `import random`, `import torch`, `import torch.nn as nn` | Lines 68–70 | Required for seeding and activation_fn |
| Global RNG seeding | `main()` before dataset load | `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` — all set from `cfg["training"]["seed"]` (default 0) |
| `make_vec_env(seed=_seed)` | Line 643 | Vectorized envs now seeded |
| Config validation warning `_validate_ppo_keys()` | Called inside `load_config()` | Warns at startup if any PPO key is at root level → would have caught v9/v10 bugs immediately |
| `n_epochs` added to `MaskablePPO(...)` | `build_model()` line 513 | `n_epochs=int(pc.get("n_epochs", 10))` |
| `ortho_init` support | `build_model()` lines 491–492 | `policy_kwargs["ortho_init"] = True` if `ppo.ortho_init: true` in config |
| `activation_fn` support | `build_model()` lines 493–499 | Maps `"tanh"/"relu"/"elu"` → `nn.Tanh/ReLU/ELU` via `policy_kwargs["activation_fn"]` |
| `calibrate_makespan_bounds()` function | Before `main()` | Runs 5 random-action episodes to establish fixed normalization bounds |
| Fixed normalisation: `ref_makespan_min/max` passed to EarlyStoppingCallback | `main()` lines 673–686 | Phase 3 fix — composite scores now use same denominator throughout training |

#### early_stopping_callback.py

| Change | Detail |
|---|---|
| Added `ref_makespan_min`, `ref_makespan_max` constructor params | If provided, stored as fixed bounds; `_fixed_bounds = True` |
| Skip `_makespan_min/max` update when fixed bounds active | `_run_evaluation()` — bounds unchanged after calibration |
| Backward compatible | If params omitted, falls back to accumulating behavior |

#### base_config.yaml

| Change | Detail |
|---|---|
| Added `seed: 0` under `training:` | Controls all RNG seeding in run_experiment.py |

#### v9_high_gamma_makespan.yaml

| Change | Detail |
|---|---|
| Moved `ent_coef: 0.02` under `ppo:` | Now correctly applied — was at root, ignored |
| Removed `gamma: 0.999` | Same as base default — was a no-op even if correctly placed |
| Real change vs v7 | `ent_coef: 0.02` (mild entropy boost) — now actually applied |

#### v10_wide_network_makespan.yaml — Complete Redesign

Renamed `v10_deep_critic_extreme`. Replaced broken "wide network + ortho_init" that was never applied. New design: takes **v8's reward weights** (best smoke-test performer) and adds 6 architectural improvements correctly under `ppo:`:

| Key | Value | Rationale |
|---|---|---|
| `net_arch_pi/vf` | `[256, 256, 256]` | Deeper network for long-horizon credit assignment |
| `ortho_init` | `true` | Prevents vanishing gradients in deep nets |
| `vf_coef` | `0.75` | Stronger critic training for FJSSP credit assignment |
| `max_grad_norm` | `0.3` | Prevents gradient spikes from v8's high penalty magnitudes |
| `n_epochs` | `15` | More gradient passes per rollout (now actually read and applied) |
| `learning_rate` | `0.00015` | Stabilises training with deeper network + higher n_epochs |

#### v12_composite_tight.yaml

Added `activation_fn: "tanh"` under `ppo:`. Previously claimed in description but not set anywhere. Now explicit and supported by `run_experiment.py`.

#### v6_ppo_entropy_boost.yaml — Redesigned as v6_deadline_tardiness_focus

v5 had total_tardiness rank 13 and deadlines_met rank 8 among all policies. v6 is redesigned to directly fix these weaknesses while keeping v5's proven capacity (network + LR):

| Key | v5 value | v6 new value | Reason |
|---|---|---|---|
| `job_on_time` | 50 | **100** | Major deadline incentive (2× v5) |
| `job_late` | −80 (base) | **−120** | Match tardiness_step strength |
| `utilization_bonus` | 40 | **30** | Less noise from utilization signal |
| `ent_coef` | 0.01 (base) | **0.03** | Mild exploration for deadline strategies |
| `gamma` | 0.999 (base) | **0.998** | Slightly shorter horizon → more deadline focus |
| `net_arch_pi/vf` | `[512,512,128]` | same | Keep v5's proven capacity |
| `learning_rate` | 0.0001 | same | Keep v5's stable LR |

### 20.3 Phase 3 — Fixed Normalisation for Composite Score

**Problem:** The early stopping callback accumulated `_makespan_min/_makespan_max` across all evaluation batches. A model at 1M steps with makespan 95k might score 0.75 on a narrow early range, but after training expands the range to [40k, 120k], the same model would retroactively score differently. This caused the patience counter to increment even when the model genuinely improved, because the normalization denominator changed.

**Fix:** `calibrate_makespan_bounds(env_factory, n_episodes=5)` runs 5 random-action episodes before training starts to establish reference bounds:
- `ref_min = min(random_makespans) * 0.60` — expect RL to beat random by ~40%
- `ref_max = max(random_makespans) * 1.05` — slight headroom above worst random

These fixed bounds are passed to `EarlyStoppingCallback`. All composite scores throughout training use the same denominator. The patience counter now reliably tracks genuine improvement.

### 20.4 Valid Ablations Table (post-fix)

| Variation | Config file | What it tests | Type |
|---|---|---|---|
| v1 | `v1_ppo_default.yaml` | Baseline — all defaults | **Control** |
| v2 | `v2_ppo_makespan_focus.yaml` | Makespan/step-time reward focus | Reward ablation |
| v3 | `v3_ppo_deadline_focus.yaml` | Deadline/tardiness reward focus | Reward ablation |
| v4 | `v4_ppo_composite.yaml` | Composite on-time + util reward | Reward ablation |
| v7 | `v7_makespan_util_focus.yaml` | Full reward redesign (moderate) | Reward ablation |
| v8 | `v8_critical_path_extreme.yaml` | Extreme time/idle penalties | Reward ablation |
| v9 *(fixed)* | `v9_high_gamma_makespan.yaml` | v7 rewards + ent_coef=0.02 | Single PPO param on v7 |
| v13 | `v13_tassel_reward.yaml` | Tassel scheduled-area reward | Paper comparison |
| v5 | `v5_ppo_large_network.yaml` | v4 rewards + larger net + lower LR | Multi-factor (document) |
| v6 *(redesigned)* | `v6_ppo_entropy_boost.yaml` | v5 network + deadline-focused reward | Multi-factor, principled |
| v10 *(redesigned)* | `v10_wide_network_makespan.yaml` | v8 rewards + deep net + critic | Multi-factor architecture |
| v11 | `v11_utilization_shaped.yaml` | Full reward + n_steps change | Multi-factor |
| v12 | `v12_composite_tight.yaml` | Full reward + 7 PPO params | Exploratory |

For paper: use v1–v4, v7, v8, v9, v13 as the primary ablation table. Describe v5, v6, v10, v11, v12 as "multi-factor exploratory configurations" in a separate table or appendix.

### 20.5 Updated Console Commands

```bash
PYTHON="D:\UROP\Dynamic-Job-Shop-Scheduling-RL\gitdjssp\Scripts\python.exe"
cd D:\UROP\Dynamic-Job-Shop-Scheduling-RL

# Step 0 — Generate benchmark bank (once)
$PYTHON generate_toy_company_csv.py

# Step 1 — Train all 13 variations (best-first order)
$PYTHON experiments/run_experiment.py --config experiments/configs/v8_critical_path_extreme.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v10_wide_network_makespan.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v9_high_gamma_makespan.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v12_composite_tight.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v7_makespan_util_focus.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v11_utilization_shaped.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v13_tassel_reward.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v4_ppo_composite.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v3_ppo_deadline_focus.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v2_ppo_makespan_focus.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v1_ppo_default.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v5_ppo_large_network.yaml
$PYTHON experiments/run_experiment.py --config experiments/configs/v6_ppo_entropy_boost.yaml

# Step 2 — After training: fill scaling_eval.models_to_eval in base_config.yaml
# Step 3 — Scaling eval (each model × 15 benchmark instances = 195 runs)
$PYTHON experiments/run_scaling_eval.py

# Step 4 — Per-model evaluation + compare-all
for d in experiments/2026*/; do $PYTHON experiments/evaluate_variation.py --experiment "$d"; done
$PYTHON experiments/compare_all.py --root experiments/
```
