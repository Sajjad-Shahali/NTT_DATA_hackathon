# Training Pipeline Fix: Dataset-Specific Training

Prepared on: 2026-03-20
Scope: Making `--n_jobs` and `--seed` CLI arguments work for PPO and DQN training.

---

## 1. Current Issue (Before Fix)

When running:
```bash
python experiments/run_experiment.py --config experiments/configs/v1_ppo_default.yaml --n_jobs 30 --seed 42
```

The pipeline crashed with:
```
FileNotFoundError: data/benchmark/n30/seed_42/bench_n30_s42_jobs.csv
```

**Root cause**: `build_dataset()` in `experiments/common/dataset.py` called `load_benchmark_instance()` from the old v3 generator (`generate_toy_company_csv.py`), which expected pre-generated benchmark bank files at `data/benchmark/n{N}/seed_{S}/bench_n{N}_s{S}_*.csv`. These files were never generated because the v3 generator requires a real Excel file (`data/dataset_scheduling.xlsx`) that may not be available, and the benchmark bank was never populated.

---

## 2. Files Modified

### `experiments/common/dataset.py` (the only file changed)

This is the shared dataset loader used by both PPO and DQN trainers:
- `experiments/ppo/trainer.py` line 38: `from experiments.common.dataset import build_dataset`
- `experiments/dqn/trainer.py` line 54: `from experiments.common.dataset import build_dataset`

No changes needed in the trainers, CLI, or config files — they already pass `n_jobs` and `seed` correctly.

---

## 3. Code Changes Made

### `experiments/common/dataset.py` — `build_dataset()` function

**Before (broken)**:
```python
if n_jobs is not None and seed is not None:
    from generate_toy_company_csv import load_benchmark_instance
    return load_benchmark_instance(
        n_jobs=n_jobs, seed=seed,
        out_base=_resolve_project_path("data/benchmark"),
        time_unit=cfg["dataset"]["time_unit"],
    )
```

**After (working)**:
```python
if n_jobs is not None and seed is not None:
    from generate_toy_synthetic import generate, CFG as _synth_cfg
    import copy

    out_dir = _resolve_project_path(f"data/generated/n{n_jobs}_s{seed:02d}")
    prefix  = f"syn_n{n_jobs}_s{seed:02d}"
    csv_check = os.path.join(out_dir, f"{prefix}_jobs.csv")

    if os.path.exists(csv_check):
        print(f"[dataset] Loading cached synthetic dataset: {out_dir}/{prefix}_*.csv")
        return pcr.load_company_dataset_from_csv(
            out_dir=out_dir, prefix=prefix,
            time_unit=cfg["dataset"]["time_unit"],
        )

    # Generate on-the-fly and cache
    print(f"[dataset] Generating synthetic dataset: n_jobs={n_jobs}, seed={seed}")
    synth_cfg = copy.deepcopy(_synth_cfg)
    dataset = generate(synth_cfg, n_jobs=n_jobs, seed=seed,
                       out=out_dir, prefix=prefix)
    return dataset
```

**Why each change was needed:**
1. **Replaced `generate_toy_company_csv` with `generate_toy_synthetic`** — the new generator is self-contained (no Excel dependency), produces correct utilization-calibrated datasets, and outputs the exact 3-file CSV schema the env expects.
2. **Generate-on-demand with caching** — first run generates CSVs to `data/generated/n{N}_s{SS}/`, subsequent runs load from cache. No need to pre-generate a benchmark bank.
3. **Deterministic naming** — `data/generated/n30_s42/syn_n30_s42_*.csv` makes it clear which configuration produced which files. Different (n_jobs, seed) pairs never collide.

---

## 4. Run Commands

### PPO training
```bash
# n=30, seed=42
python experiments/run_experiment.py --config experiments/configs/v1_ppo_default.yaml --n_jobs 30 --seed 42

# n=100, seed=0
python experiments/run_experiment.py --config experiments/configs/v1_ppo_default.yaml --n_jobs 100 --seed 0

# Smoke test (1000 timesteps)
python experiments/run_experiment.py --config experiments/configs/v1_ppo_default.yaml --n_jobs 30 --seed 42 --timesteps 1000
```

### DQN training
```bash
# DQN vanilla, n=30, seed=42
python experiments/run_experiment.py --config experiments/dqn/configs/dqn_vanilla.yaml --algo dqn --n_jobs 30 --seed 42

# DQN double, n=100, seed=0
python experiments/run_experiment.py --config experiments/dqn/configs/dqn_double.yaml --algo dqn --n_jobs 100 --seed 0
```

### Without --n_jobs/--seed (default toy dataset)
```bash
# Falls through to data/processed/toy_*.csv as before
python experiments/run_experiment.py --config experiments/configs/v1_ppo_default.yaml
```

### Regenerate a dataset (delete cache to force regeneration)
```bash
# Delete cached dataset
rm -rf data/generated/n30_s42/
# Next training run will regenerate automatically
```

### Generate dataset standalone (without training)
```bash
python generate_toy_synthetic.py --n-jobs 30 --seed 42 --out data/generated/n30_s42 --prefix syn_n30_s42
```

---

## 5. Validation / Smoke Test Results

### PPO n=30 seed=42 (1000 timesteps)
```
Dataset: 30 jobs, 7 machines
Max steps per episode: 20,000
[dataset] Loading cached synthetic dataset: data/generated/n30_s42/syn_n30_s42_*.csv
fps=205  total_timesteps=1000  ✓ no crash
```

### PPO n=100 seed=0 (1000 timesteps)
```
Dataset: 100 jobs, 7 machines
Max steps per episode: 20,000
[dataset] Loading cached synthetic dataset: data/generated/n100_s00/syn_n100_s00_*.csv
fps=150  total_timesteps=1000  ✓ no crash
```

### DQN vanilla n=30 seed=42 (500 timesteps)
```
Dataset: 30 jobs, 7 machines
obs_dim=128, action_dim=31
DQN network: vanilla  hidden=[256, 256]  params=106,783
[Calibration] Random-policy makespans: [14060, 14230, 14410, 14170, 14230]
✓ no crash
```

### Dataset generation stats
```
n=30 seed=42:  avg_job_work=2044 min  realized_util=0.676  avg_ops=5.4  ~5.2 simultaneous active jobs
n=100 seed=0:  avg_job_work=2042 min  realized_util=0.757  avg_ops=5.6  ~5.2 simultaneous active jobs
```

---

## 6. Remaining Notes

### Env seed bug — already fixed
`env_company.py` line 330 already uses `self._rng.choice(idle_elig)` for randomized machine tie-breaking. Different episode seeds → different machine assignments → different episode trajectories. This was fixed in the previous session.

### Dataset caching
Generated datasets are cached at `data/generated/n{N}_s{SS}/`. To regenerate with different generator parameters (e.g., different `target_util`), delete the cache directory first. The generator uses default `Config` values (target_util=0.75, due_tight=[1.3, 1.8], dur_clip=[10, 800]).

### Backward compatibility
- Running without `--n_jobs`/`--seed` still loads from `data/processed/toy_*.csv` as before.
- The old `generate_toy_company_csv.load_benchmark_instance()` path is no longer called from `build_dataset()`. The `data/benchmark/` directory and its code are effectively dead.

### Observation dimension scales with n_jobs
- n=30: obs_dim=128, action_dim=31
- n=100: obs_dim=408, action_dim=101
- Larger datasets need proportionally larger networks. The default [256, 256] may be undersized for n=100+.

### Recommended configurations for training
- **n=30, seed=42**: Good for development/debugging. ~5 simultaneous active jobs, fast episodes.
- **n=100, seed=0**: Realistic scale. ~5 simultaneous active jobs, longer episodes.
- **n=50, seed=0**: Middle ground (not yet generated — will auto-generate on first use).
