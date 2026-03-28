# Dynamic Job Shop Scheduling with Reinforcement Learning
## Complete Technical Reference

> **Project**: Flexible Dynamic Job Shop Scheduling Problem (FJSSP/DJSSP) solved with Proximal Policy Optimization (PPO) on Real Manufacturing Data
> **Author**: UROP Research Project
> **Period**: December 2025 – March 2026
> **Purpose of this document**: Exact, code-verified reference for the full project — intended as context for AI assistants, collaborators, or reviewers who need to understand the complete system.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Why Reinforcement Learning?](#2-why-reinforcement-learning)
3. [System Architecture](#3-system-architecture)
4. [Environment Design](#4-environment-design)
5. [RL Algorithm: MaskablePPO](#5-rl-algorithm-maskableppo)
6. [Reward Engineering](#6-reward-engineering)
7. [Data Pipeline](#7-data-pipeline)
8. [Training Infrastructure](#8-training-infrastructure)
9. [Evaluation & Baselines](#9-evaluation--baselines)
10. [Results & Analysis](#10-results--analysis)
11. [Key Challenges & Design Decisions](#11-key-challenges--design-decisions)
12. [Conclusion & Future Work](#12-conclusion--future-work)

---

## 1. Problem Definition

### 1.1 The Job Shop Scheduling Problem (JSSP)

The **Job Shop Scheduling Problem** is a classic combinatorial optimization problem in manufacturing:

- A set of **jobs** must be processed through a set of **machines**
- Each job has a prescribed **sequence of operations**
- Each operation requires a specific machine for a specific duration
- Goal: Find a schedule that **minimizes an objective** (e.g., makespan, tardiness)

### 1.2 Extensions in This Project

This project addresses the **Flexible Dynamic** variant (FJSSP), which adds two critical real-world constraints:

| Extension | Description |
|-----------|-------------|
| **Flexible** | Each operation can be executed on **multiple eligible machines** — the scheduler must pick the best one |
| **Dynamic** | Jobs arrive over time (**stochastic arrivals**) — the full job set is not known at the start |

### 1.3 Real-World Context

The project is grounded in **real manufacturing data** from an actual production facility:

- Operations are labelled with real codes: `Fre_20`, `Fre_30`, `Fre_40`, `Fre_50`, `Fre_51`, `Fre_60`, `Fre_70`, `Fre_22`
- Machines follow shop-floor naming: `F2_1`, `F2_2`, `F2_3`, `F1_2`, `F1_3`
- Eligibility scores (`1` = occasional, `2` = always capable) encode real machine capability ratings
- The source data (`dataset_scheduling.xlsx`) contains thousands of real job records

### 1.4 Formal Problem Statement

$$\text{Minimize} \quad \alpha \cdot C_{\max} + \beta \cdot \sum_{j} T_j + \gamma \cdot (1 - U)$$

Where:
- $C_{\max}$ = makespan (finish time of the last job)
- $T_j = \max(0, C_j - D_j)$ = tardiness of job $j$ (lateness beyond due date $D_j$)
- $U$ = machine utilization (fraction of time machines are busy)
- $\alpha, \beta, \gamma$ = objective weights (tuned via reward shaping)

**Decision**: At each time step — which job's next operation should be dispatched, and to which machine?

---

## 2. Why Reinforcement Learning?

### 2.1 Limitations of Classical Approaches

| Approach | Limitation for DJSSP |
|----------|---------------------|
| **Exact methods** (ILP, MIP) | NP-hard; intractable for large instances (>20 jobs) |
| **Metaheuristics** (GA, SA) | Require full problem knowledge upfront; poor dynamic adaptability |
| **Simple dispatching rules** (SPT, EDD) | Fixed policies; cannot adapt to multi-objective tradeoffs or job interactions |

### 2.2 Why RL Works Here

- **Online decision-making**: RL agent makes decisions as jobs arrive — no need for full lookahead
- **Learned heuristics**: Implicitly learns dispatching rules better than handcrafted ones
- **Multi-objective handling**: Reward function can balance makespan, tardiness, and utilization simultaneously
- **Domain generalization**: Once trained, inference is near-instantaneous (just a forward pass through a neural network)

### 2.3 Key RL Framing

| RL Concept | Scheduling Interpretation |
|------------|--------------------------|
| **State** $s_t$ | Current machine occupancies, job statuses, remaining work, urgency |
| **Action** $a_t$ | Select which job to dispatch next (or do nothing) |
| **Reward** $r_t$ | Signal reflecting on-time completions, machine utilization, tardiness |
| **Policy** $\pi(a \mid s)$ | The scheduling strategy learned by the agent |
| **Episode** | One complete scheduling horizon (until all jobs finish) |

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PROJECT PIPELINE                          │
│                                                             │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────┐  │
│  │  Raw Data    │───▶│  Preprocessing  │───▶│  Dataset  │  │
│  │  (Excel/CSV) │    │  (normalize,    │    │  (CSVs)   │  │
│  │              │    │   encode)       │    │           │  │
│  └──────────────┘    └─────────────────┘    └─────┬─────┘  │
│                                                   │        │
│                                                   ▼        │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────┐  │
│  │  Evaluation  │◀───│  Trained Model  │◀───│  Training │  │
│  │  (vs rules)  │    │  (PPO policy)   │    │  (MaskPPO)│  │
│  └──────────────┘    └─────────────────┘    └─────┬─────┘  │
│         │                                         │        │
│         ▼                                         ▼        │
│  ┌──────────────┐                        ┌────────────────┐│
│  │  Gantt Charts│                        │  RL Environment ││
│  │  Metrics CSV │                        │  (env_company) ││
│  │  Comparison  │                        └────────────────┘│
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Codebase Structure

```
D:\UROP\
├── Dynamic-Job-Shop-Scheduling-RL/        # Main production module
│   ├── env_company.py                     # Core RL environment (FJSSP)
│   ├── env.py                             # Toy/baseline environment
│   ├── train.py                           # PPO training entry point
│   ├── evaluate.py                        # Benchmark evaluation
│   ├── preprocess_company_real.py         # Excel → structured CSV pipeline
│   ├── generate_toy_company_csv.py        # Synthetic data generator
│   ├── utils.py                           # Gantt chart + metric plotting
│   ├── main.py                            # Demo/quick-start script
│   ├── data/
│   │   ├── dataset_scheduling.xlsx        # Real company source data
│   │   └── processed/                     # Preprocessed CSV files
│   ├── results/                           # Training run outputs
│   └── results_company_real/              # Evaluation outputs
│
├── data/                                  # Root-level datasets
│   ├── jobs.csv                           # Job metadata
│   ├── job_ops.csv                        # Per-job operations
│   ├── op2machines.csv                    # Operation–machine eligibility
│   └── company_dataset_flat.csv           # Full denormalized join
│
├── generate_dataset_final.py              # Dataset generation script
├── env.py                                 # Root toy environment
├── train.py                               # Root training script
└── evaluate.py                            # Root evaluation script
```

---

## 4. Environment Design

### 4.1 Overview

The environment is implemented as a **Gymnasium-compatible RL environment** (`CompanyFJSSPEnv` in `env_company.py`) with full action masking support.

### 4.2 Internal State

The scheduler tracks the following internal state at every time step $t$:

| Variable | Type | Description |
|----------|------|-------------|
| `current_time` | `float` | Simulation clock (in minutes) |
| `machine_free_time[m]` | `float[M]` | Earliest time machine $m$ is available |
| `job_current_op[j]` | `int[N]` | Index of the next pending operation for job $j$ |
| `job_completion_time[j]` | `float[N]` | Actual completion time (set when job finishes) |
| `job_done[j]` | `bool[N]` | Whether job $j$ is fully scheduled |
| `gantt_log` | `list[dict]` | Full history of operation dispatches (for visualization) |

### 4.3 Observation Space

The observation is a **flattened `float32` vector** with **tanh normalization** to prevent gradient issues:

$$\text{obs} = \left[ t_{\text{norm}}, \; \{f_j^{(1)}, f_j^{(2)}, f_j^{(3)}, f_j^{(4)}\}_{j=1}^{N}, \; \{m_k\}_{k=1}^{M} \right]$$

**Global feature (1 dim)**:
$$t_{\text{norm}} = \tanh\!\left(\frac{t}{\tau}\right)$$

**Per-job features (4 dims each)**:

| Feature | Formula | Meaning |
|---------|---------|---------|
| `arrived_flag` | $\mathbf{1}[t \geq a_j]$ | Has job $j$ arrived yet? |
| `done_flag` | $\mathbf{1}[\text{job complete}]$ | Is job $j$ finished? |
| `slack_norm` | $\tanh\!\left(\frac{D_j - t}{\tau}\right)$ | Time urgency (negative = overdue) |
| `remaining_norm` | $\tanh\!\left(\frac{W_j^{\text{rem}}}{\tau}\right)$ | Remaining processing work |

**Per-machine features (1 dim each)**:

| Feature | Formula | Meaning |
|---------|---------|---------|
| `time_to_free_norm` | $\tanh\!\left(\frac{\max(0, f_m - t)}{\tau}\right)$ | Wait until machine is free |

where $\tau = \max(\max_j D_j, \sum_j W_j, 1.0)$ is the normalization scale.

**Total dimension**: $1 + 4N + M$

*Example*: 50 jobs + 10 machines → **211-dimensional** observation vector.

### 4.4 Action Space

**Discrete space** with $N + 1$ actions:

| Action | Meaning |
|--------|---------|
| $0 \ldots N-1$ | Dispatch job $j$'s next pending operation to the best eligible machine |
| $N$ | **No-op**: advance the clock to the next event without dispatching |

#### Machine Selection (within an action)

When dispatching job $j$, the environment automatically selects the machine minimizing the **earliest start time**:

$$m^* = \arg\min_{m \in \mathcal{E}_{j,\text{op}}} \max(t, f_m)$$

where $\mathcal{E}_{j,\text{op}}$ is the set of eligible machines for job $j$'s current operation.

### 4.5 Action Masking

Invalid actions are masked out to prevent exploration into infeasible regions:

```python
def get_action_mask(self) -> np.ndarray:
    mask = np.zeros(self.num_jobs + 1, dtype=bool)
    mask[self.num_jobs] = True  # no-op always valid
    for j in range(self.num_jobs):
        if (self.job_current_op[j] < len(self.job_ops[j])
                and not self.job_done[j]
                and self.arrival_times[j] <= self.current_time):
            mask[j] = True
    return mask
```

This ensures:
- Jobs that haven't arrived yet cannot be dispatched
- Completed jobs cannot be selected
- The no-op is always available to prevent deadlock

### 4.6 Time Advancement

The environment uses a **discrete event-style** clock with configurable `time_step = 10 minutes`. After each action, if no job is dispatchable, the clock advances to the next event (machine free time or next job arrival).

---

## 5. RL Algorithm: MaskablePPO

### 5.1 PPO Background

**Proximal Policy Optimization (PPO)** is a policy-gradient algorithm that constrains policy updates to a trust region:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\!\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$ and $\hat{A}_t$ is the **Generalized Advantage Estimate (GAE)**:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 5.2 Why MaskablePPO?

Standard PPO samples from the full action distribution, including invalid actions. This leads to:
- Wasted exploration on infeasible dispatches
- Reward hacking via repeated invalid-then-valid cycles

**`sb3_contrib.MaskablePPO`** modifies the softmax to exclude masked actions:

$$\pi(a \mid s) = \frac{\exp(z_a / T) \cdot \mathbf{m}_a}{\sum_{a'} \exp(z_{a'} / T) \cdot \mathbf{m}_{a'}}$$

where $\mathbf{m} \in \{0,1\}^{|A|}$ is the binary mask from `get_action_mask()`.

### 5.3 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | `3e-4` | Standard Adam default |
| `n_steps` | `8192` | Large rollout buffer for long scheduling episodes |
| `batch_size` | `1024` | Large mini-batch for stable gradients |
| `gamma` | `0.999` | Near-undiscounted; episodes span thousands of steps |
| `gae_lambda` | `0.95` | Bias-variance tradeoff in advantage estimation |
| `ent_coef` | `0.01` | Mild entropy bonus for exploration |
| `vf_coef` | `0.5` | Standard value function loss weight |
| `max_grad_norm` | `0.5` | Gradient clipping for stability |
| `clip_range` | `0.2` | PPO clipping threshold |
| `n_epochs` | `10` | Updates per rollout collection |
| `total_timesteps` | `2,000,000` | Training budget |
| `n_envs` | `auto (CPU/2)` | Parallel environments for data efficiency |

### 5.4 Network Architecture

Both policy and value networks share the same MLP:

```
Input (obs_dim) → Linear(256) → Tanh → Linear(256) → Tanh → Output
```

- **Policy head**: Linear(256 → num_actions), then masked softmax
- **Value head**: Linear(256 → 1)

---

## 6. Reward Engineering

### 6.1 Design Philosophy

Reward shaping is the **most critical design choice** in this project. The reward must:

1. **Be dense** — signal per step, not just at episode end
2. **Be normalized** — scale-invariant across different problem sizes
3. **Balance multiple objectives** — makespan, tardiness, utilization
4. **Avoid local optima** — entropy bonuses + step penalties prevent stalling

### 6.2 Reward Components

All rewards are normalized by $\tau = \max(\text{max\_due\_date}, \sum \text{work}, 1.0)$:

#### Job Completion Events
| Component | Value | Trigger |
|-----------|-------|---------|
| `job_complete` | **+8.0** | Any job finishes |
| `job_on_time` | **+40.0** | Completion $\leq$ due date |
| `job_late` | $-80.0 \times \frac{\text{tardiness}}{\tau}$ | Completion $>$ due date |

#### Per-Step Shaping (scaled by $\Delta t / \tau$)
| Component | Value | Signal |
|-----------|-------|--------|
| `late_unfinished_step` | $-80.0 \times \text{(overdue fraction)}$ | Urgency for late jobs |
| `machine_idle` | $-40.0 \times \text{(idle fraction)}$ | Penalize idle capacity |
| `utilization_bonus` | $+30.0 \times \text{(busy fraction)}$ | Reward machine utilization |
| `tardiness_step` | $-120.0 \times \text{(normalized tardiness)}$ | Dense lateness signal |

#### Episode-Level
| Component | Value | Trigger |
|-----------|-------|---------|
| `step_time_penalty` | **-10.0** per step | Mild speed incentive |
| `invalid_action` | $-2.0 \times \text{penalty\_factor}$ | Invalid dispatch attempt |
| `final_unscheduled` | **-100.0** per job | Episode ends with unfinished jobs |

### 6.3 Total Reward per Step

$$r_t = r^{\text{completion}} + \frac{\Delta t}{\tau} \cdot r^{\text{shaping}} + r^{\text{episode}}$$

### 6.4 Why These Weights?

The large negative weight on `tardiness_step` (-120.0) vs `utilization_bonus` (+30.0) reflects the business priority: **on-time delivery is more important than throughput**. The `job_on_time` bonus (+40.0) is the single largest positive signal, reinforcing the primary objective.

---

## 7. Data Pipeline

### 7.1 Source Data

**Real company Excel file** (`dataset_scheduling.xlsx`) containing:
- Production orders (job IDs, arrival times, due dates)
- Routing cards (operation sequences per product type)
- Machine capability matrix (which machine can process which operation)

### 7.2 Preprocessing (`preprocess_company_real.py`)

```
Excel file
    │
    ▼
[Parse jobs]          → arrival_dt, due_dt (timestamps)
    │
    ▼
[Parse operations]    → per-job op sequence, durations
    │
    ▼
[Parse eligibility]   → op_id → [machine_id] mapping with scores
    │
    ▼
[Normalize to minutes]→ arrival_min, due_min (relative to t0)
    │
    ▼
[Output CSVs]         → jobs.csv, job_ops.csv, op2machines.csv
```

### 7.3 Dataset Schema

**`jobs.csv`**:
```
job_id | arrival_min | due_min | job_name
```

**`job_ops.csv`**:
```
job_id | seq_index | op_id | duration_min
```

**`op2machines.csv`**:
```
op_id | machine_id | eligibility_score
```
where `eligibility_score ∈ {1 (occasional), 2 (always capable)}`

### 7.4 Synthetic Data Generation (`generate_dataset_final.py`)

For controlled experiments, synthetic datasets are generated with configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_JOBS` | 200 | Number of jobs |
| `DURATION_SCALE` | 0.5–1.0 | Scale factor for operation durations |
| `DUE_TIGHT_MIN` | 1.4 | Min tightness factor (due = arrival + load × factor) |
| `DUE_TIGHT_MAX` | 1.8 | Max tightness factor |
| Operations | 8 types | `Fre_20` through `Fre_70` |
| Machines | 5 | `F2_1, F2_2, F2_3, F1_2, F1_3` |

---

## 8. Training Infrastructure

### 8.1 Vectorized Environments

Training uses **parallel environment execution** for sample efficiency:

```python
n_envs = max(1, os.cpu_count() // 2)
vec_env = DummyVecEnv([make_env(dataset, ...) for _ in range(n_envs)])
```

Each environment samples a **different subset** of jobs from the dataset per episode, inducing robustness to schedule diversity.

### 8.2 Episode Horizon

The maximum steps per episode is dynamically set to prevent infinite loops:

$$\text{max\_steps} = \max\!\left(\frac{D_{\max} \times 1.2}{\Delta t},\; 20{,}000\right)$$

This gives 20% headroom beyond the latest due date.

### 8.3 Callbacks & Logging

A custom `RewardLoggerCallback` captures per-episode metrics:

```python
episode_stats.csv:
  timestep | ep_reward | ep_length | makespan | tardiness
         | utilization | deadlines_met | on_time_pct
```

Additional outputs per training run (`results/run_YYYYMMDD_HHMMSS/`):
- `tb/` — TensorBoard event files (mean reward, episode length curves)
- `ppo_company_fjsp.zip` — Saved model checkpoint
- `learning_curve.png` — Episode reward + rolling average plot
- `manifest.txt` — Full configuration snapshot

### 8.4 Training Timeline

| Run Date | Focus |
|----------|-------|
| Dec 7–8, 2025 | Initial experiments, toy environment validation |
| Feb 20, 2026 | Real company data integration, reward tuning |
| Mar 2026 | Final evaluation, analysis |

Total: **~30+ training runs** across both toy and real company environments.

---

## 9. Evaluation & Baselines

### 9.1 Evaluation Protocol

All methods are evaluated on the **same** `CompanyFJSSPEnv` instance with identical random seeds for fair comparison. Each method runs **N episodes** and averages metrics.

### 9.2 Heuristic Dispatching Rules

Eight classical rules serve as baselines:

| Rule | Selection Criterion | Known Strength |
|------|--------------------|--------------------|
| **FIFO** | Earliest arrival time | Simple, fair |
| **LIFO** | Latest arrival time | Minimizes WIP |
| **SPT** | Shortest next operation | Minimizes avg flow time |
| **LPT** | Longest next operation | Maximizes utilization |
| **EDD** | Earliest due date | Minimizes max lateness |
| **LDD** | Latest due date | Good makespan in some cases |
| **SLACK** | $D_j - t - W_j^{\text{rem}}$ (min) | Urgency-aware |
| **CR** | $\frac{D_j - t}{W_j^{\text{rem}}}$ (min) | Dynamic urgency ratio |

### 9.3 Metrics Computed

| Metric | Formula | Direction |
|--------|---------|-----------|
| **Makespan** | $C_{\max} = \max_j C_j$ | ↓ Lower is better |
| **Total Tardiness** | $\sum_j \max(0, C_j - D_j)$ | ↓ Lower is better |
| **Utilization** | $\frac{\sum_m \text{busy}_m}{C_{\max} \times M}$ | ↑ Higher is better |
| **Deadlines Met** | $\#\{j : C_j \leq D_j\}$ | ↑ Higher is better |
| **On-Time %** | $\frac{\text{deadlines\_met}}{N}$ | ↑ Higher is better |

### 9.4 Visualization Outputs

| Output | Description |
|--------|-------------|
| `gantt_*.png` | Gantt chart per policy (machine × time) |
| `bar_makespan.png` | Bar chart comparing makespan across all policies |
| `bar_tardiness.png` | Bar chart comparing total tardiness |
| `scatter_makespan_vs_tardiness.png` | Multi-objective trade-off visualization |
| `ratio_vs_ppo.png` | Performance ratio (heuristic / PPO); values >1 mean PPO wins |
| `rank_heatmap.png` | Ranking matrix across all metrics and policies |

---

## 10. Results & Analysis

### 10.1 Training Convergence

Training runs (`results/`) show iterative improvement over ~30 experiments:

- **Episode rewards** increase from strongly negative (early runs) toward positive values
- **Makespan** decreases with training, approaching or beating the best heuristic
- **Utilization** improves as the agent learns to avoid leaving machines idle
- Latest runs (`run_20260220_*`) show stable convergence after ~1.5M steps

### 10.2 PPO vs Heuristics

The evaluation pipeline (`results_company_real/`) compares all 9 methods (PPO + 8 rules).

**Expected performance patterns** (based on problem structure):

| Scenario | Expected Top Rule |
|----------|------------------|
| Tight due dates | **EDD** or **SLACK** |
| Maximizing throughput | **SPT** |
| Balanced objectives | **CR** or **PPO** |

**PPO's advantage**: Unlike any single heuristic, PPO can **adapt its implicit rule** based on the current state — switching between urgency-driven and throughput-maximizing behavior within a single episode.

### 10.3 Key Observed Patterns

1. **PPO learns to interleave**: Rather than always dispatching the most urgent job, the agent learns to **pipeline** operations — dispatching less urgent jobs when urgent ones are blocked, keeping all machines busy.

2. **Dynamic switching**: The learned policy implicitly switches priorities based on slack values — behaving like EDD when jobs are near deadlines, like SPT when there is buffer time.

3. **Utilization vs tardiness tradeoff**: High utilization alone does not guarantee low tardiness; the agent learns that some idle time is acceptable if it prevents future conflicts.

---

## 11. Key Challenges & Design Decisions

### 11.1 Sparse vs Dense Rewards

**Challenge**: With only episode-end rewards, credit assignment over thousands of steps is intractable.

**Solution**: Dense per-step shaping rewards (`tardiness_step`, `utilization_bonus`) provide frequent gradient signal, while the `step_time_penalty` prevents the agent from accumulating shaping rewards indefinitely.

### 11.2 Action Masking

**Challenge**: With $N+1$ actions, many are invalid at each step (unarrived jobs, completed jobs, no eligible machine idle). Without masking, the agent wastes capacity learning to avoid these.

**Solution**: `MaskablePPO` with `get_action_mask()` ensures the policy only considers valid dispatches, dramatically accelerating learning.

### 11.3 Large Observation Space

**Challenge**: Real company data has 7,000+ jobs, making a 4N+M+1 observation vector massive (~28,000 dims).

**Solution**: The toy environment uses a fixed subset (50 jobs, 10 machines → 211 dims). For the real environment, jobs are batched or subsampled per episode. Tanh normalization ensures all features have consistent scale regardless of problem size.

### 11.4 Long Horizon Episodes

**Challenge**: Episodes can span 50,000+ timesteps (multi-day production schedules in minute resolution).

**Solution**: `gamma = 0.999` (near-undiscounted), large `n_steps = 8192` rollout buffer, and per-step shaping ensure the agent receives useful gradients throughout the horizon.

### 11.5 Real vs Synthetic Data

**Challenge**: Real company data has irregular job arrivals, unbalanced operation mixes, and missing/inconsistent eligibility data.

**Solution**: A dedicated preprocessing pipeline (`preprocess_company_real.py`) handles data cleaning, while a synthetic generator (`generate_dataset_final.py`) provides controlled experiments for ablation studies.

---

## 12. Conclusion & Future Work

### 12.1 Summary

This project successfully applies **Proximal Policy Optimization with action masking** to the **Flexible Dynamic Job Shop Scheduling Problem** on real manufacturing data. Key contributions:

1. **End-to-end RL pipeline** for real industrial data (Excel → RL environment → trained policy)
2. **Carefully engineered reward function** balancing makespan, tardiness, and utilization with normalized, dense shaping
3. **Comprehensive evaluation framework** comparing PPO against 8 classical dispatching heuristics
4. **Scalable environment design** supporting both toy (30 jobs) and large-scale (7,000+ jobs) instances

### 12.2 Technical Stack

| Layer | Technology |
|-------|-----------|
| RL Framework | `stable-baselines3 == 2.3.2` + `sb3_contrib` (MaskablePPO) |
| Deep Learning | `PyTorch 2.2.2` |
| Environment | `Gymnasium 0.29.1` |
| Data | `pandas 2.3.3`, `openpyxl 3.1.5` |
| Visualization | `matplotlib 3.7.5` |
| Logging | `TensorBoard 2.14.0` |

### 12.3 Future Work

| Direction | Description |
|-----------|-------------|
| **Graph Neural Networks** | Replace MLP with GNN to handle variable-size job sets natively |
| **Multi-agent RL** | Assign one agent per machine for decentralized scheduling |
| **Attention mechanisms** | Transformer-based policy to model job–machine interactions |
| **Online learning** | Continuously update policy as new production data arrives |
| **Constraint handling** | Explicit precedence constraints, setup times, maintenance windows |
| **Multi-objective Pareto** | Learn a Pareto front of schedules rather than a weighted sum |

---

## Appendix A: File Reference

| File | Location | Purpose |
|------|---------|---------|
| `env_company.py` | `Dynamic-Job-Shop-Scheduling-RL/` | Primary RL environment (`CompanyFJSSPEnv`) |
| `train.py` | `Dynamic-Job-Shop-Scheduling-RL/` | Training entry point |
| `evaluate.py` | `Dynamic-Job-Shop-Scheduling-RL/` | Benchmarking vs heuristics |
| `preprocess_company_real.py` | `Dynamic-Job-Shop-Scheduling-RL/` | Excel → CSV pipeline |
| `utils.py` | `Dynamic-Job-Shop-Scheduling-RL/` | Gantt + metric plots |
| `generate_dataset_final.py` | `D:\UROP/` | Synthetic dataset generator |
| `generate_toy_company_csv.py` | `Dynamic-Job-Shop-Scheduling-RL/` | Toy dataset generator |
| `main.py` | `Dynamic-Job-Shop-Scheduling-RL/` | Quick demo script |

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **JSSP** | Job Shop Scheduling Problem |
| **FJSSP** | Flexible JSSP (multiple eligible machines per op) |
| **DJSSP** | Dynamic JSSP (jobs arrive over time) |
| **PPO** | Proximal Policy Optimization (RL algorithm) |
| **GAE** | Generalized Advantage Estimation |
| **Makespan** ($C_{\max}$) | Time from start to last job completion |
| **Tardiness** ($T_j$) | $\max(0, C_j - D_j)$ — lateness past due date |
| **Utilization** | Fraction of machine-time actually used |
| **Action masking** | Zeroing out invalid actions in the policy |
| **Dispatching rule** | Greedy heuristic for ordering jobs (FIFO, EDD, etc.) |
| **Time scale** ($\tau$) | Normalization factor for observation and reward |

---

*Generated: March 16, 2026*
