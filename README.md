# NTT DATA Hackathon 2026 — Tire-Road Friction Identification

![Dashboard Demo](reports/demo-intro.png)

**Event:** NTT DATA BEST Hackathon 2026
**Date:** 28 March 2026 | Politecnico di Torino, Turin, Italy
**Challenge:** Applying Innovative AI Technologies (ML/AI/GenAI/Agentic AI) to R&D Engineering and Product Development Processes

---

## Overview

This project implements a **real-time tire-road friction identification system** using an Extremum Seeking Control (ESC) gradient-based identifier. The system continuously estimates the parameters of the Burckhardt tire friction model from live wheel slip and friction measurements, enabling optimal slip tracking for maximum braking and traction force.

The core algorithm was originally developed in MATLAB and has been ported to Python for integration into AI-augmented engineering workflows.

---

## The Problem

In automotive control systems (ABS, traction control, ESC), the controller must know the tire-road friction curve to operate at peak grip. This curve changes in real time as road conditions change (dry → wet → snow → ice). Identifying it online — without stopping the vehicle — is a classic and safety-critical challenge.

### The Burckhardt Friction Model

The tire friction coefficient μ as a function of wheel slip *s* follows:

```
μ(s) = c1 · (1 − exp(−c2 · |s|)) − c3 · |s|
```

| Parameter | Physical meaning |
|---|---|
| `c1` | Peak friction amplitude |
| `c2` | Curve sharpness (high on ice/snow, low on dry asphalt) |
| `c3` | Descending slope after peak |

The optimal slip — where friction is maximum — is:

```
s_opt = ln(c1 · c2 / c3) / c2
```

**Goal:** Identify `c1`, `c2`, `c3` in real time from noisy sensor data, without requiring the vehicle to drive across the friction peak.

---

## Algorithm: ESC_TwoPoint_ID (v3)

The identifier uses three hierarchical layers that run every timestep:

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1 — LUT                                              │
│  μ_max → interpolate → c2_coarse  (always active, fast)     │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2 — ESC-Gradient Analytic                            │
│  (s, μ, g_esc) → solve analytically → c1, c3               │
│  Uses real-time ESC gradient — no peak assumption needed    │
├─────────────────────────────────────────────────────────────┤
│  LAYER 3 — Brent Root-Finding                               │
│  Two-point (s_peak, μ_peak) + (s_probe, μ_probe) → c2_fine │
│  Active only when slip excitation is sufficient             │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1 — Lookup Table

A precomputed table maps observed peak friction to a coarse `c2` estimate:

| Surface | μ_max | c2 |
|---|---|---|
| Ice | 0.19 | 94.1 |
| Snow | 0.40 | 33.8 |
| Wet asphalt | 0.85 | 33.8 |
| Dry asphalt | 1.15 | 23.99 |

This provides an immediate, fallback-safe estimate at every timestep.

### Layer 2 — ESC-Gradient Analytic (v3 — key innovation)

Given the ESC gradient `g_esc` (from an outer ESC optimizer's LPF demodulator), the true gradient of the friction curve at the current operating point is:

```
g_true = −g_esc · 2 / a_esc
```

This allows solving for `c1` and `c3` analytically at **any** operating point — not just at the friction peak:

```
c1 = (μ − g_true · |s|) / (1 − (1 + c2·|s|) · exp(−c2·|s|))
c3 = c1 · c2 · exp(−c2·|s|) − g_true
```

**Why this matters (v3 vs v1/v2):** Previous versions assumed the buffer maximum equals the true friction peak (`dμ/ds = 0`). This fails whenever the ESC perturbation places the operating point away from the peak — especially on snow where the peak is sharp (`c2 = 94`). The v3 gradient method is accurate at any operating point, significantly improving parameter estimates and therefore the confidence metric `alpha`.

### Layer 3 — Brent Root-Finding

When the ESC has explored a sufficiently wide slip range (`|s_probe − s_peak| > fark_threshold`), a two-point root-finding problem is solved to refine `c2`:

```
find c2 such that: μ_probe = c1(c2)·(1 − exp(−c2·|s_probe|)) − c3(c2)·|s_probe|
```

Brent's method (superlinear convergence, guaranteed bracket) is used. A narrow bracket around the LUT estimate is tried first; the full `[c2_min, c2_max]` range is the fallback. Results deviating more than 60% from the LUT estimate are rejected as spurious.

### Road Change Detection

A smoothed peak friction `μ_max_smooth` tracks the running surface. If the buffer peak changes by more than ±15%, all Brent history is reset and the identifier adapts to the new surface.

### Fallback Chain

```
ESC gradient available?
    YES → ESC-Gradient Analytic (debug_flag = 0 or 10)
    NO  → Peak-based fallback (debug_flag = 20)

Brent bracket found?
    YES → refined c2 (debug_flag = 0)
    NO  → LUT c2     (debug_flag = 10 or 20)

t < t_start?
    YES → return without update (debug_flag = 1)
```

---

## Repository Structure

```
NTT_DATA_hackathon/
├── hackaton_id.m.txt        # Original MATLAB implementation
├── hackaton_id.py           # Python port — ESCTwoPointID class
├── predict_mu.py            # μ(s) prediction script (3 models compared)
├── src/
│   ├── burckhardt.py        # Burckhardt model math, surface presets, s_opt/mu_peak utils
│   └── data_gen.py          # Synthetic (slip, μ) dataset generation
├── data/
│   ├── raw/                 # Place real measurement CSVs here (slip, mu_noisy, surface)
│   └── synthetic/           # Auto-generated Burckhardt curves (CSV)
├── make_plots.py            # Smoke test + generate all 5 presentation plots
├── models/                  # Save fitted models here
├── reports/
│   └── plots/               # Auto-generated presentation plots (make_plots.py)
├── templates/               # Flask HTML templates (index.html, dashboard.html)
├── static/img/              # Profile photos served by Flask
├── demo/                    # Static HTML mockups (intro, dashboard previews)
├── rules/
│   ├── hackathon_case_study.md          # Full case study documentation
│   ├── use_case_template.md             # Use case definition template
│   ├── 2026-03-28_NTT_DATA_BEST_...pdf  # Original hackathon brief (password-protected)
│   └── 2026-03-26_UseCaseDefinition...  # Use case template (PowerPoint)
├── requirements.txt         # Python dependencies
├── CLAUDE.md                # AI assistant guidance
└── README.md                # This file
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Sajjad-Shahali/NTT_DATA_hackathon.git
cd NTT_DATA_hackathon

# Activate the existing virtual environment (Windows/Git Bash)
source .env/Scripts/activate

# Install all dependencies
pip install -r requirements.txt
```

Requires Python 3.8+ and the packages in `requirements.txt` (`numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`).

---

## Usage

### Class-based (recommended — one instance per wheel)

```python
import numpy as np
from hackaton_id import ESCTwoPointID

# Instantiate one identifier per wheel
identifier = ESCTwoPointID()

# Parameter vector (15 elements, 0-indexed)
params = np.array([
    8,       # [0]  buf_size
    0,       # [1]  (unused)
    0.008,   # [2]  fark_threshold
    0,       # [3]  (unused)
    0.5,     # [4]  t_start
    0.5,     # [5]  c1_min
    2.0,     # [6]  c1_max
    10.0,    # [7]  c2_min
    150.0,   # [8]  c2_max
    0.01,    # [9]  c3_min
    1.0,     # [10] c3_max
    0.8,     # [11] mu_buf_init
    0.05,    # [12] s_buf_init
    1e-4,    # [13] brent_tol
    20,      # [14] brent_iter
])

# Initial parameter estimates (e.g. dry asphalt)
c1, c2, c3 = 1.28, 23.99, 0.52

# Call every timestep
result = identifier.identify(
    mu_measured = 0.85,   # measured friction coefficient
    s_measured  = 0.12,   # measured wheel slip
    s_probe     = 0.10,   # ESC-perturbed slip setpoint
    c1_prev     = c1,
    c2_prev     = c2,
    c3_prev     = c3,
    t           = 1.0,    # current simulation time [s]
    params      = params,
    g_esc       = -0.03,  # ESC gradient estimate from LPF demodulator
    a_esc       = 0.02,   # ESC dither amplitude
)

(c1, c2, c3, s_opt, mu_opt, valid, debug_flag,
 best_fark, fark_now, best_s, best_mu,
 s_at_peak, mu_opt_est) = result

if valid:
    print(f"Identified: c1={c1:.3f}, c2={c2:.2f}, c3={c3:.4f}")
    print(f"Optimal slip: {s_opt:.4f}  |  Peak friction: {mu_opt_est:.3f}")
```

### Function wrapper (single-wheel convenience)

```python
from hackaton_id import ESC_TwoPoint_ID

result = ESC_TwoPoint_ID(mu_measured, s_measured, s_probe,
                          c1_prev, c2_prev, c3_prev,
                          t, params, g_esc, a_esc)
```

> **Note:** The function wrapper uses a module-level shared instance. For multi-wheel simulations always use `ESCTwoPointID()` directly.

---

## Real Data Pipeline

```bash
# 1. Export from MATLAB (run in MATLAB)
run('run_validation_updated.m')   # 48 simulations
run('export_for_python.m')        # → friction_data_full.csv

# 2. Copy CSV to project
cp friction_data_full.csv data/raw/

# 3. Preprocess
python preprocess.py              # → data/raw/prepared_friction.csv  (20,616 rows)

# 4. Train models on real data
python predict_mu.py --data data/raw/prepared_friction.csv --n-train 500 --save reports/plots/

# 5. Visualise real data
python plot_data_exploration.py   # plots 06–10
python plot_burckhardt_vs_real.py # plots 11–12

# 6. Evaluate surface identification (unlabeled)
python evaluate_unlabeled.py      # plot 13 — single-batch demo + 45-batch accuracy
python eval_classification.py     # plot 14 — F1 / Precision / Recall per class
```

### Preprocessing summary (`preprocess.py`)

Raw `friction_data_full.csv` (765k rows) → `prepared_friction.csv` (20,616 rows):

| Step | Filter | Rows remaining |
|---|---|---|
| Start | Raw | ~765,000 |
| Drop 12 columns | s_hat, road_1/2, dither, T_front/rear, … | same |
| ABS filter | `abs_active == 1` | ~479,000 |
| Confidence filter | `alpha > 0.01` | ~340,000 |
| Stride=10 | Thin autocorrelated time-series per run×wheel | ~48,000 |
| Balance | Snow cap=12,000 / Dry,Wet cap=6,000 each | **20,616** |



---

## Offline μ(s) Prediction — `predict_mu.py`

For offline analysis and model validation, `predict_mu.py` trains and compares three methods on synthetic or real measurement data.

### Quick start

```bash
# Generate synthetic data for all 4 surfaces and compare models
python predict_mu.py

# Single surface only
python predict_mu.py --surface dry

# Vary training set size (test data efficiency)
python predict_mu.py --n-train 20

# Use your own measurements (CSV: slip, mu_noisy, surface columns)
python predict_mu.py --data data/raw/my_measurements.csv

# Save plots to disk
python predict_mu.py --save plots/
```

### Methods compared

| Method | Description | Best for |
|---|---|---|
| **Burckhardt NLS** | `scipy.curve_fit` — fits c1, c2, c3 directly | Small datasets, interpretable params, physics known |
| **Gaussian Process** | `sklearn.GPR` — non-parametric with uncertainty bounds | When Burckhardt model may not fit; need confidence intervals |
| **Neural Network** | `sklearn.MLP` — 3-layer regressor | Large datasets (1000+ samples), multi-modal inputs |

### Typical RMSE results (50 training points, noise σ=0.01)

| Surface | Burckhardt NLS | Gaussian Process | Neural Network |
|---|---|---|---|
| Dry asphalt | **0.0099** | 0.0096 | 0.0129 |
| Wet asphalt | **0.0099** | 0.0099 | 0.0104 |
| Snow | 0.0099 | **0.0057** | 0.0103 |
| Ice | 0.0099 | **0.0056** | 0.0102 |

**Verdict:** Burckhardt NLS is the default choice — 3 parameters, physically interpretable, data-efficient. Gaussian Process wins on sharp-peak surfaces (snow/ice) with sufficient samples.

---

## Return Values

| Index | Name | Description |
|---|---|---|
| 0 | `c1_new` | Updated Burckhardt parameter c1 |
| 1 | `c2_new` | Updated Burckhardt parameter c2 |
| 2 | `c3_new` | Updated Burckhardt parameter c3 |
| 3 | `s_opt_out` | Optimal slip (at maximum friction), clamped to [0.01, 0.45] |
| 4 | `mu_opt_out` | Buffer peak friction (observed maximum) |
| 5 | `valid` | 1 if identification succeeded this step, 0 otherwise |
| 6 | `debug_flag` | Active layer: 0=Brent+grad, 1=warmup, 10=LUT+grad, 20=fallback |
| 7 | `best_fark_out` | Best slip excitation distance recorded |
| 8 | `fark_simdi_out` | Current slip excitation distance |
| 9 | `best_s_out` | Slip at best probe point |
| 10 | `best_mu_out` | Friction at best probe point |
| 11 | `s_at_mu_max_k` | Slip value corresponding to buffer peak |
| 12 | `mu_opt_estimated` | μ evaluated at `s_opt` using identified parameters |

---

## Required Input Data

To run the identifier, the following signals are needed at each timestep:

| Signal | Symbol | Source |
|---|---|---|
| Wheel slip | `s` | `(v_wheel − v_vehicle) / v_vehicle` |
| Friction coefficient | `μ` | `F_x / F_z` — longitudinal force / normal load |
| ESC gradient | `g_esc` | Output of LPF demodulator in outer ESC optimizer |
| ESC dither amplitude | `a_esc` | Known design parameter of the ESC excitation signal |
| ESC probe slip | `s_probe` | Current ESC-perturbed slip setpoint |

### Deriving the inputs

```
v_wheel   → ABS wheel speed sensor (standard in all modern vehicles)
v_vehicle → GPS / optical sensor / average of non-driven wheels
F_x       → Drivetrain torque model: F_x = (T_engine − I_wheel · α_wheel) / R_wheel
F_z       → Static load + accelerometer-based load transfer model
g_esc     → Demodulation: g_esc = LPF[ μ_measured · sin(ω·t) ] · (2/a)
```

For **offline curve fitting** only `(time, slip, friction)` triplets are needed.

---

## Hackathon Context

This implementation addresses **Use Case Area #2 — Requirements Engineering / Embedded Control** from the NTT DATA hackathon challenge V-Model, specifically the intersection of:

- **Area 6** — Assistants for SW Design and Development (automatic algorithm synthesis)
- **Area 7** — Quality Analysis (automatic parameter validation, model consistency checks)

### Challenge Goal

Each team must define an **Executive Proposal** of 4 AI use cases (2 Make + 2 Buy) targeting a chosen industry sector, evaluated on:

| Criterion | Weight |
|---|---|
| Use Case Definition | 25% |
| Market Tools Analysis | 25% |
| Executive Proposal quality | 35% |
| Prioritization Approach | 10% |
| Process Mapping | 5% |

See [rules/hackathon_case_study.md](rules/hackathon_case_study.md) for the full case study breakdown and [rules/use_case_template.md](rules/use_case_template.md) for the use case definition template.


### Presentation Plots

Run `python make_plots.py` to regenerate all plots to `reports/plots/`:

| File | Content | Slide use |
|---|---|---|
| `01_surfaces_overview.png` | All 4 Burckhardt curves with μ_peak & s_opt annotated | "The Problem" |
| `02_model_comparison.png` | 2×2 grid: GT vs NLS vs GP vs NN per surface | Model validation |
| `03_rmse_comparison.png` | Grouped bar chart: RMSE by method and surface | Accuracy summary |
| `04_data_efficiency.png` | RMSE vs training-set size (dry & ice) | Data efficiency argument |
| `05_identifier_convergence.png` | ESC identifier converging from wrong surface to ground truth | "Our solution" |

### Data Exploration Plots

Run `python plot_data_exploration.py` — requires `data/raw/friction_data_full.csv`:

| File | Content |
|---|---|
| `06_data_slip_distribution.png` | Slip histograms raw vs preprocessed per surface |
| `07_data_mu_vs_slip.png` | Scatter mu vs slip + Burckhardt ground-truth overlay |
| `08_data_alpha_distribution.png` | Identifier confidence (alpha) by ABS state + time-series |
| `09_data_scenario_coverage.png` | Sample counts per scenario before/after preprocessing |
| `10_data_real_vs_model.png` | Real simulation data vs Burckhardt ground truth per surface |

### c_matrix vs Real Data

Run `python plot_burckhardt_vs_real.py`:

| File | Content |
|---|---|
| `11_burckhardt_vs_real.png` | Per-surface: measured scatter + Burckhardt model curve |
| `12_burckhardt_vs_real_combined.png` | All 3 surfaces on one axes for side-by-side comparison |

### Unlabeled Evaluation & Classification Metrics

```bash
# Identify surface type from raw (slip, mu) — no label given
python evaluate_unlabeled.py              # 100% accuracy, 45 batches
python eval_classification.py             # F1 / Precision / Recall report
```

| File | Content |
|---|---|
| `13_unlabeled_evaluation.png` | Fitted curves, c1/c2 scatter, confusion matrix, RMSE dist. |
| `14_classification_metrics.png` | Per-class Precision / Recall / F1 bar chart + confusion matrix |

---

## Technical Reference

### Burckhardt Surface Parameters (typical values)

| Surface | c1 | c2 | c3 | μ_peak | s_opt |
|---|---|---|---|---|---|
| Dry asphalt | 1.2801 | 23.99 | 0.52 | ~0.85–1.0 | ~0.17 |
| Wet asphalt | 0.857 | 33.82 | 0.347 | ~0.60 | ~0.12 |
| Snow | 0.1946 | 94.13 | 0.0646 | ~0.18 | ~0.06 |
| Ice | 0.05 | 306.4 | 0.001 | ~0.05 | ~0.02 |

### Key Equations

```
# Friction curve
μ(s) = c1·(1 − exp(−c2·|s|)) − c3·|s|

# Gradient (dμ/d|s|)
μ'(s) = c1·c2·exp(−c2·|s|) − c3

# Optimal slip (peak of curve)
s_opt = ln(c1·c2 / c3) / c2

# Peak friction
μ_peak = μ(s_opt)

# Validity condition (peak exists)
c1·c2 > c3
```

---

## Known Issues

| Issue | Location | Details |
|---|---|---|
| Brent swap bug | `hackaton_id.py` line 265 / `hackaton_id.m.txt` line 220 | `a_b=b_b; b_b=c_br; c_br=a_b` duplicates old `b_b` — identical bug in both files; fix only when MATLAB source is corrected |
| Codex CLI auth | Dev tooling | `refresh_token_reused` 401 — fix with `codex logout && codex login` |

---

## Live Web Dashboard

A browser-based demo replays `data/raw/friction_data_full.csv` through the sliding-window Burckhardt NLS classifier, visualising per-wheel surface identification in real time.

**Features:**
- Three.js top-down car scene with split road (left/right surface coloring) and animated wheels
- Left / right data panels: surface badge, μ_peak bar, c₁/c₂/c₃ parameters, slip sparkline, Burckhardt μ(s) curve
- Live stats strip: classifications count, avg μ per side, surface change count
- Interactive noise robustness slider: drag σ_μ 0→0.10, see real F1 + Balanced Accuracy from actual model runs
- Professional intro page with 3-layer algorithm explanation, real Burckhardt curves with data scatter, actual prediction table

### Prerequisites

- `data/raw/friction_data_full.csv` must exist (export from MATLAB — see Real Data Pipeline above)
- Virtual environment activated

### Run

```bash
# 1. Activate environment
source .env/Scripts/activate

# 2. Install Flask (only needed once)
pip install flask

# 3. Start the server
python app.py

# 4. Open in browser
#    http://localhost:5000
```

The intro screen loads first. Scroll to explore the algorithm, real predictions, and noise robustness. Click **▶ Start Live Demo** to open the dashboard.

### Dashboard Controls

| Control | Description |
|---------|-------------|
| ▶ / ⏸ | Play / Pause the replay |
| ⟳ Restart | Reset to row 0, clear both side buffers |
| Speed ×0.5 → ×5 | Adjust replay cadence |
| Scenario selector | Filter CSV by `scenario` column |
| Noise slider | Add σ_μ to incoming data, watch F1 / BAC update live |

### Classification Metrics

The classifier is evaluated with **F1 macro** and **Balanced Accuracy (BAC)**. With equal batch counts per surface these are equivalent and neither inflates scores due to class imbalance. Real results from `data/raw/prepared_friction.csv`:

| σ_μ added | F1 macro | Bal. Accuracy | Condition |
|-----------|----------|---------------|-----------|
| 0.000 | **1.000** | **1.000** | Ideal (baseline) |
| 0.010 | 0.724 | 0.733 | Realistic vehicle |
| 0.050 | 0.528 | 0.533 | Degraded sensor |
| 0.100 | 0.436 | 0.433 | Extreme noise |

See [reports/project_handoff.md §16](reports/project_handoff.md) for full architecture details.

---

## Team

| Name | Role | Programme |
|------|------|-----------|
| **Sajjad Shahali** | Machine Learning Engineer | MSc — [Politecnico di Torino](https://www.polito.it/) |
| **Omer Ozkan** | Data Science & Engineer | MSc — [Politecnico di Torino](https://www.polito.it/) |
| **Kaan Sadik Aslan** | Mechatronic Engineering | MSc — [Politecnico di Torino](https://www.polito.it/) |
| **Berk Ali Demir** | Mechatronic Engineer | MSc — [Politecnico di Torino](https://www.polito.it/) |

---

## License

This project was developed for the NTT DATA BEST Hackathon 2026 at Politecnico di Torino.
