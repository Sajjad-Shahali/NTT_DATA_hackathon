# Project Handoff — Tire-Road Friction Identification
**Date:** 28 March 2026
**Event:** NTT DATA BEST Hackathon 2026 — Politecnico di Torino
**Authors:** s328433@studenti.polito.it · s340464@studenti.polito.it
**Repo:** https://github.com/Sajjad-Shahali/NTT_DATA_hackathon

---

## 1. What This Project Is

A real-time tire-road friction identification system using an Extremum Seeking Control (ESC) gradient algorithm. The core is a three-layer identifier (`ESCTwoPointID`) that estimates the Burckhardt friction model parameters `(c1, c2, c3)` from live wheel slip and friction measurements.

The original algorithm was written in MATLAB (`hackaton_id.m.txt`). We ported it to Python, validated it against MATLAB simulation data, built a surface classification pipeline, and wrote a preprint-style technical report.

---

## 2. Current State (as of 28 March 2026)

### Done
- Python port of MATLAB identifier — `hackaton_id.py` (`ESCTwoPointID` class)
- 48-simulation MATLAB dataset exported to CSV (`friction_data_full.csv`)
- Full preprocessing pipeline — 765k rows → 20,616 balanced rows
- Three ML models compared for offline μ(s) prediction (NLS, GP, NN)
- Surface classifier (unlabeled slip+mu → dry/wet/snow) — 100% F1 on simulation data
- Three-source robustness sweep (sensor noise, slip range, parameter drift)
- 18 plots generated in `reports/plots/`
- LaTeX preprint paper — `reports/friction_identification_report.tex`
- Four hackathon deliverable use cases + executive proposal in `deliverables/`
- MATLAB noisy export script — `export_for_python_noisy.m.txt`

### Pending / Not Yet Done
- Run `export_for_python_noisy.m.txt` in MATLAB → generate `friction_data_noisy_smu0.0100_ss0.0020_sp0.0500.csv`
- Feed noisy MATLAB data back through `eval_classification.py` (currently robustness tested with synthetic Python noise only)
- Compile `reports/friction_identification_report.tex` to PDF and verify layout
- Real-time streaming implementation (WebSocket / MQTT / ZeroMQ — not yet built)

---

## 3. Burckhardt Friction Model

The tire friction coefficient μ as a function of wheel slip s:

```
μ(s) = c1 · (1 − exp(−c2 · |s|)) − c3 · |s|
```

The gradient (dμ/d|s|):

```
μ'(s) = c1 · c2 · exp(−c2 · |s|) − c3
```

The optimal slip (where friction peaks):

```
s_opt = ln(c1 · c2 / c3) / c2        [valid only when c1·c2 > c3]
```

Peak friction:

```
μ_peak = μ(s_opt)
```

### Surface Parameters (c_matrix)

| Surface    | c1     | c2     | c3     | μ_peak | s_opt |
|------------|--------|--------|--------|--------|-------|
| Dry asphalt| 1.2801 | 23.99  | 0.5200 | ~1.17  | ~0.17 |
| Wet asphalt| 0.8570 | 33.822 | 0.3470 | ~0.80  | ~0.13 |
| Snow       | 0.1946 | 94.129 | 0.0646 | ~0.19  | ~0.06 |
| Ice        | 0.0500 | 306.40 | 0.0010 | ~0.05  | ~0.03 |

**Key insight:** c2 is the most discriminating parameter between surfaces (24 vs 34 vs 94 vs 306). The classifier exploits this.

---

## 4. Three-Layer ESC Identifier (hackaton_id.py)

Every timestep the identifier receives:
- `mu_measured` — friction coefficient from force estimation
- `s_measured` — wheel slip from encoder
- `s_probe` — ESC-perturbed slip setpoint
- `g_esc` — ESC gradient from LPF demodulator
- `a_esc` — ESC dither amplitude (known design parameter)
- `c1_prev, c2_prev, c3_prev` — previous estimates (warm-started)

### Layer 1 — Lookup Table (always active)

Maps observed buffer-peak friction to a coarse c2 estimate via linear interpolation:

```
μ_max → interp1([0.19, 0.40, 0.85, 1.15], [94.1, 33.8, 33.8, 23.99], μ_max) → c2_coarse
```

This is the fallback when Layers 2 and 3 cannot fire. It is always computed and used as the initial bracket for Brent.

### Layer 2 — ESC-Gradient Analytic (v3, key innovation)

Convert ESC gradient to the true friction curve gradient:

```
g_true = −g_esc · 2 / a_esc
```

Then solve analytically for c1 and c3 at the **current operating point** (not just at the peak):

```
c1 = (μ − g_true · |s|) / (1 − (1 + c2·|s|) · exp(−c2·|s|))
c3 = c1 · c2 · exp(−c2·|s|) − g_true
```

This is the v3 fix. Previous versions (v1/v2) assumed the operating point was at the friction peak (μ'=0), which fails when the ESC perturbation places the wheel away from the peak — especially on snow (sharp peak, c2=94).

### Layer 3 — Brent Root-Finding (fires when slip excitation is sufficient)

When `|s_probe − s_peak| > δ_s = 0.008`, two points are available on the friction curve. Solve for c2 by finding the root of:

```
f(c2) = μ_probe − [c1(c2)·(1 − exp(−c2·|s_probe|)) − c3(c2)·|s_probe|] = 0
```

Brent's method is used (superlinear convergence, guaranteed bracket). The bracket starts narrow around the LUT estimate; falls back to full [c2_min, c2_max] = [10, 150]. Results deviating >60% from the LUT estimate are rejected.

### debug_flag values

| Value | Meaning |
|-------|---------|
| 0 | Brent active + ESC-gradient analytic |
| 1 | Warmup — t < t_start, no update |
| 10 | LUT c2 + ESC-gradient analytic |
| 20 | LUT c2 + peak-based fallback (no g_esc) |

### Road Change Detection

A smoothed peak friction `μ_max_smooth` tracks surface state. If it changes by ±15%, all Brent history is flushed and the identifier reinitialises to the new surface.

---

## 5. Data Pipeline

### MATLAB Side (simulation → CSV)

```
run_validation_updated.m   →  48 simulations (8 scenarios × 3 brake distributions × 2 modes)
export_for_python.m        →  friction_data_full.csv   (~765k rows, 29 columns)
export_for_python_noisy.m  →  friction_data_noisy_*.csv (same schema + 3 noise columns)
```

**29 columns per row:**
- Identity: `test_no, case_no, wheel, mode, scenario, dist_name, road_1, road_2`
- Measured: `slip, mu_noisy, s_probe, g_esc, a_esc, s_hat, alpha, dither`
- Derived: `abs_active` (alpha > 0.001), `surface` (label string)
- Ground truth: `c1_true, c2_true, c3_true, mu_true, s_opt_true, mu_peak_true`
- Run metrics: `d_stop, t_stop, T_front, T_rear`

### Python Preprocessing (preprocess.py)

Raw 765k rows → 20,616 balanced rows via this filter chain:

| Step | Filter | Rows remaining |
|------|--------|---------------|
| Start | Raw CSV | ~765,000 |
| ABS filter | `abs_active == 1` | ~479,000 |
| Confidence filter | `alpha > 0.01` | ~340,000 |
| Slip range | `0.002 ≤ slip ≤ 0.55` | ~310,000 |
| Stride | Every 10th row per run×wheel (removes autocorrelation) | ~48,000 |
| Balance | Snow capped at 12,000; Dry and Wet capped at 6,000 each | **20,616** |

**Why Snow cap is 2× Dry/Wet:** Snow scenarios cover more simulation time (lower deceleration, longer stopping distance), so raw Snow rows dominate. The 12k/6k/6k split gives balanced training while retaining more of the harder class.

**Why stride=10:** The simulation timestep is ~1ms; at stride=1, consecutive rows are almost perfectly correlated. Stride=10 keeps 1 row per ~10ms window — enough to decorrelate while retaining the full slip range.

---

## 6. ML Pipeline

### Model 1 — Burckhardt NLS (predict_mu.py, evaluate_unlabeled.py, eval_classification.py)

Fits the Burckhardt model directly to (slip, mu) data using non-linear least squares:

```python
from scipy.optimize import curve_fit

def burckhardt(s, c1, c2, c3):
    return c1 * (1 - np.exp(-c2 * np.abs(s))) - c3 * np.abs(s)

# Three seed initialisations to avoid local minima:
seeds = [
    [1.28, 24.0, 0.52],   # dry asphalt prior
    [0.86, 34.0, 0.35],   # wet asphalt prior
    [0.19, 94.0, 0.06],   # snow prior
]
bounds = ([0.01, 5.0, 0.001], [2.0, 350.0, 1.0])

best_c, best_residual = None, np.inf
for seed in seeds:
    try:
        c, _ = curve_fit(burckhardt, slip, mu, p0=seed, bounds=bounds, maxfev=5000)
        res = np.sum((burckhardt(slip, *c) - mu)**2)
        if res < best_residual:
            best_c, best_residual = c, res
    except RuntimeError:
        pass
```

**Why three seeds:** Burckhardt is non-convex. A dry-asphalt initialisation will diverge on snow data. Three seeds covering the three canonical surfaces ensures convergence regardless of input surface.

### Model 2 — Gaussian Process (predict_mu.py)

Kernel: `ConstantKernel(1.0) × RBF(length_scale=0.1) + WhiteKernel(noise_level=1e-4)`

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

kernel = ConstantKernel(1.0) * RBF(0.1) + WhiteKernel(1e-4)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
gpr.fit(slip_train.reshape(-1,1), mu_train)
mu_pred, sigma = gpr.predict(slip_test.reshape(-1,1), return_std=True)
```

**Advantage over NLS:** Provides uncertainty bounds (σ per prediction). Better on sharp-peak surfaces (snow/ice) when NLS struggles with the narrow peak region.

### Model 3 — Neural Network (predict_mu.py)

Architecture: `MLP(hidden_layer_sizes=(64, 64, 32), activation='relu', max_iter=2000)`

```python
from sklearn.neural_network import MLPRegressor

nn = MLPRegressor(hidden_layer_sizes=(64, 64, 32), activation='relu',
                  solver='adam', max_iter=2000, random_state=42)
nn.fit(slip_train.reshape(-1,1), mu_train)
```

**Weakness:** Requires ~500+ training points to match NLS. At n_train=50 it is the worst performer (RMSE up to 0.048). Does not extrapolate physically outside the training slip range.

### RMSE Comparison (n_train=50, σ_μ=0.01)

| Surface     | Burckhardt NLS | Gaussian Process | Neural Network |
|-------------|---------------|-----------------|---------------|
| Dry asphalt | **0.0099**    | 0.0096          | 0.0129        |
| Wet asphalt | **0.0099**    | 0.0099          | 0.0104        |
| Snow        | 0.0099        | **0.0057**      | 0.0103        |
| Ice         | 0.0099        | **0.0056**      | 0.0102        |

**Verdict:** NLS is the default — physics-embedded, 3 parameters, data-efficient. GP wins on sharp-peak surfaces with sufficient data and when uncertainty bounds are needed.

---

## 7. Surface Classification Pipeline

### How It Works

Given a batch of 50 unlabeled (slip, mu) points:

1. **Fit Burckhardt NLS** to the batch → get `(c1_fit, c2_fit, c3_fit)`
2. **Normalise** the fitted parameters against the c_matrix reference values:
   ```python
   c_norm = np.array([c1_fit/C_REF[:,0], c2_fit/C_REF[:,1], c3_fit/C_REF[:,2]])
   ```
3. **Nearest-neighbour** in normalised (c1, c2, c3) space: Euclidean distance to each reference surface
4. **Classify** as the closest surface

### Why It Works on Simulation Data (F1=1.0)

This is a **two-level circular reasoning trap**:
- The simulation generates (slip, mu) data using the exact Burckhardt formula with the exact c_matrix values
- The classifier fits the Burckhardt formula to (slip, mu) and compares against the same c_matrix
- The fit must converge to the exact input parameters → classification cannot fail

**F1=1.0 is a software sanity check, not a classifier accuracy result.** The robustness sweeps (Section 8) are the scientifically meaningful evaluation.

### Batch Size = 50

Chosen empirically:
- Too small (< 20): NLS fit has high variance, especially on noisy data
- Too large (> 200): Includes surface transitions in real driving; reduces temporal resolution
- 50 points at stride=10 = ~500ms of data = one braking phase window

### Batch Construction (eval_classification.py)

**Critical:** batches must be drawn per-surface, not randomly from the full dataset.

```python
for surf in SURFACES:
    grp = df[df["_true"] == surf].reset_index(drop=True)
    n_batches = len(grp) // batch_size
    for i in range(n_batches):
        batch = grp.iloc[i*batch_size:(i+1)*batch_size][["slip","mu_noisy"]]
        predicted_surface, c_fit = classify(batch)
        results.append((surf, predicted_surface))
```

If you mix all surfaces randomly before batching, Snow rows (2,400) dominate over Dry/Wet rows (~850 each) → every batch is majority-Snow → classifier always predicts Snow → F1=0.10. This was a bug in the first version, now fixed.

---

## 8. Three-Source Noise Model

Used in `eval_robustness.py` and `export_for_python_noisy.m.txt` to simulate realistic automotive sensor conditions.

### Equations

**Source 1 — Per-run parameter drift** (temperature, tyre wear):
```
c1_run = c1_true · (1 + σ_p · N(0,1))     ← scalar, constant for whole braking event
c2_run = c2_true · (1 + σ_p · N(0,1))
c3_run = c3_true · (1 + σ_p · N(0,1))
```

**Source 2 — Slip encoder noise** (quantisation + wheel flex):
```
s_meas = |s_true| + σ_s · N(0,1)
s_meas = max(0, s_meas)                    ← physical: slip ≥ 0
```

**Source 3 — Friction measurement noise** (force estimation error):
```
μ_true  = c1_run · (1 − exp(−c2_run · s_meas)) − c3_run · s_meas
μ_noisy = μ_true + σ_μ · N(0,1)
μ_noisy = max(0, μ_noisy)                  ← physical: μ ≥ 0
```

### Noise Parameter Scenarios

| Scenario          | σ_μ    | σ_s    | σ_p  |
|-------------------|--------|--------|------|
| Baseline (ideal)  | 0.0003 | 0.0002 | 0.00 |
| Realistic vehicle | 0.0100 | 0.0020 | 0.05 |
| Noisy sensor      | 0.0300 | 0.0050 | 0.05 |
| Severe/worn tyre  | 0.0500 | 0.0100 | 0.15 |
| Extreme test      | 0.1000 | 0.0200 | 0.20 |

**Default used:** Realistic vehicle (σ_μ=0.01, σ_s=0.002, σ_p=0.05)

---

## 9. Robustness Sweep Design

Each sweep varies one parameter while holding the others at the realistic-vehicle baseline.
Each configuration: 15 batches × 3 surfaces × 200 points per batch → F1 computed as macro-average.

### Sweep A — Sensor Noise (σ_μ)
Values: [0.0003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

### Sweep B — Slip Range (s_max)
Values: [0.50, 0.30, 0.20, 0.10, 0.07, 0.05, 0.03]
Simulates restricted operating range (normal driving vs ABS braking).

### Sweep C — Parameter Drift (σ_p)
Values: [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

### Results Summary

| Sweep | Parameter | F1=0.90 threshold | F1 at nominal | Key finding |
|-------|-----------|-------------------|---------------|-------------|
| A | σ_μ | ~0.05 | 0.967 @ σ_μ=0.01 | Standard sensors sufficient |
| B | s_max | ~0.20 | 0.967 @ s_max=0.50 | **Dominant failure** — F1=0.58 at s<0.05 |
| C | σ_p | >0.20 | 0.967 @ σ_p=0.0 | Highly robust |

**Why slip range is the dominant failure mode:** When s_max < 0.05 (normal driving, no ABS), all three surface curves look nearly identical in the linear regime near s=0. The NLS fit cannot distinguish c2=24 from c2=94 from a handful of near-zero slip points. ABS excitation (s up to 0.20+) is a hard prerequisite.

---

## 10. File Map

```
NTT_DATA_hackathon/
│
├── CORE ALGORITHM
│   ├── hackaton_id.m.txt              Original MATLAB (do not edit)
│   ├── hackaton_id.py                 Python port — ESCTwoPointID class
│   └── src/
│       ├── burckhardt.py              Model math, SURFACES dict, s_opt / mu_peak
│       └── data_gen.py                Synthetic (slip, mu) CSV generation
│
├── DATA PIPELINE
│   ├── preprocess.py                  Raw CSV → prepared_friction.csv
│   ├── load_mat_data.py               Inspect / filter / prepare real CSV
│   ├── export_for_python.m.txt        MATLAB → friction_data_full.csv
│   └── export_for_python_noisy.m.txt  MATLAB → friction_data_noisy_*.csv (3-source noise)
│
├── EVALUATION
│   ├── predict_mu.py                  NLS vs GP vs NN comparison
│   ├── evaluate_unlabeled.py          Surface ID from raw (slip,mu) — plot 13
│   ├── eval_classification.py         F1/Precision/Recall — plot 14
│   └── eval_robustness.py             3-sweep robustness analysis — plots 15–18
│
├── VISUALISATION
│   ├── make_plots.py                  Plots 01–05 (presentation)
│   ├── plot_data_exploration.py       Plots 06–10 (EDA)
│   └── plot_burckhardt_vs_real.py     Plots 11–12 (c_matrix vs real scatter)
│
├── REPORTS
│   ├── reports/plots/                 All 18 generated PNGs
│   ├── reports/friction_identification_report.tex   LaTeX preprint
│   └── reports/project_handoff.md    This file
│
├── DELIVERABLES
│   ├── deliverables/UC_001_friction_identification.md
│   ├── deliverables/UC_002_requirements_traceability.md
│   ├── deliverables/UC_003_code_migration_review.md
│   ├── deliverables/UC_004_synthetic_test_data.md
│   ├── deliverables/prioritization_matrix.md
│   └── deliverables/executive_proposal.md
│
├── DATA
│   ├── data/raw/friction_data_full.csv          ← place MATLAB export here
│   ├── data/raw/prepared_friction.csv           ← generated by preprocess.py
│   └── data/raw/unlabeled_eval.csv              ← generated by eval_classification.py
│
└── CONFIG
    ├── requirements.txt
    ├── CLAUDE.md
    └── README.md
```

---

## 11. How to Reproduce Everything from Scratch

```bash
# 0. Activate environment
source .env/Scripts/activate

# 1. Generate presentation plots (no data needed — synthetic)
python make_plots.py

# 2. After MATLAB export — preprocess real data
#    Copy friction_data_full.csv → data/raw/
python preprocess.py
python load_mat_data.py --inspect

# 3. Train and compare models
python predict_mu.py --data data/raw/prepared_friction.csv --n-train 500 --save reports/plots/

# 4. EDA and model-vs-data plots
python plot_data_exploration.py
python plot_burckhardt_vs_real.py

# 5. Surface classification evaluation
python evaluate_unlabeled.py
python eval_classification.py --csv data/raw/prepared_friction.csv

# 6. Robustness analysis
python eval_robustness.py

# 7. Compile LaTeX report
cd reports
pdflatex friction_identification_report.tex
pdflatex friction_identification_report.tex   # twice for cross-references
```

---

## 12. Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Burckhardt NLS as classifier | Fits 3 params; surfaces well-separated in c2 (24 vs 34 vs 94); data-efficient |
| Nearest-neighbour in normalised (c1,c2,c3) | Simple, interpretable, no labelled training data needed |
| Three seed initialisations for NLS | Burckhardt is non-convex; single seed fails on cross-surface data |
| F1=1.0 is a sanity check | Circular reasoning: data generated by Burckhardt → fit Burckhardt → trivially perfect |
| Batch size = 50 | Empirical: <20 gives noisy fits; >200 spans surface transitions |
| Per-surface batch construction | Random mixing causes majority-Snow batches → biased F1=0.10 |
| Per-run scalar drift | One randn per c parameter per braking event — constant within run, varies between runs |
| MATLAB faithful port | Known Brent swap bug kept — same bug in MATLAB; fix only when MATLAB is corrected |

---

## 13. Known Bugs

| Bug | File | Line | Details |
|-----|------|------|---------|
| Brent swap | `hackaton_id.py` | ~270 | `a_b=b_b; b_b=c_br; c_br=a_b` loses old `a_b`. Fix: use temp variable. Same in MATLAB — do not fix unilaterally. |

---

## 14. LaTeX Report Compilation Notes

File: `reports/friction_identification_report.tex`

Packages required (all in TeX Live 2023+):
`natbib`, `algpseudocode`, `siunitx`, `cleveref`, `lineno`, `bm`, `booktabs`, `hyperref`

- Load `algorithm` before `cleveref` or `\crefname{algorithm}` will fail
- Compile **twice** (`pdflatex` × 2) to resolve `cleveref` cross-references
- Bibliography is hand-typed (`thebibliography`) — no `.bib` file needed
- Line numbers enabled via `\linenumbers` (before `\maketitle`) — remove for final version

---

## 15. Next Steps

1. **Run MATLAB noisy export** — `export_for_python_noisy.m.txt` with default params → copy CSV to `data/raw/` → re-run `eval_classification.py` to validate against real simulation noise
2. **Compile PDF** — `pdflatex` × 2 on `reports/friction_identification_report.tex`
3. **Real-time streaming** — implement FastAPI WebSocket or ZeroMQ wrapper around `ESCTwoPointID.identify()` for live demo
4. **Real experimental data** — validate on tire test-bench data (TU Delft flat-track, NHTSA braking datasets) to break the circular reasoning limitation
