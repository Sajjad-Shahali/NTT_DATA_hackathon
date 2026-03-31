"""
make_plots.py — Generate all presentation plots for the NTT DATA Hackathon.

Saves 7 publication-quality figures to reports/plots/:
  01_surfaces_overview.png      — All 4 Burckhardt curves, μ_peak & s_opt annotated
  02_model_comparison.png       — 2×2 grid: Ground truth vs NLS vs GP vs NN per surface
  03_rmse_comparison.png        — Grouped bar chart: model accuracy by surface
  04_data_efficiency.png        — RMSE vs training-set size (shows data efficiency)
  05_identifier_convergence.png — Simulated ESC identifier converging to ground truth
  15_robustness_noise_mu.png    — Macro F1 vs σ_μ for 4 classifier variants (Monte Carlo)
  16_robustness_noise_slip.png  — Macro F1 vs σ_s for 4 classifier variants (Monte Carlo)

Usage
-----
  python make_plots.py
"""

import sys
import warnings
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# Make sure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.burckhardt import mu as bmu, gradient as bgrad, s_opt, mu_peak, SURFACES
from src.data_gen import generate_surface
from predict_mu import (
    BurckhardtFitter, GPFrictionPredictor, NNFrictionPredictor,
    train_and_evaluate, SURFACE_COLORS,
)
from hackaton_id import ESCTwoPointID

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
SAVE_DIR = Path("reports/plots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.labelsize":     11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "figure.dpi":         150,
    "savefig.dpi":        180,
    "savefig.bbox":       "tight",
})

SURFACE_KEYS   = ["dry", "wet", "snow", "ice"]
SURFACE_LABELS = [SURFACES[k].name for k in SURFACE_KEYS]
COLORS         = [SURFACE_COLORS[lbl] for lbl in SURFACE_LABELS]


# ===========================================================================
# PLOT 1 — All 4 Burckhardt Curves on One Axes
# ===========================================================================

def plot_01_surfaces_overview():
    """
    Single clean figure showing all 4 Burckhardt curves.
    Annotates μ_peak and s_opt for each surface.
    Ideal for 'The Problem' presentation slide.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    s = np.linspace(0.001, 0.50, 600)

    for key, label, color in zip(SURFACE_KEYS, SURFACE_LABELS, COLORS):
        p = SURFACES[key]
        mu_vals = bmu(s, p.c1, p.c2, p.c3)
        s_op    = s_opt(p.c1, p.c2, p.c3)
        mu_op   = mu_peak(p.c1, p.c2, p.c3)

        ax.plot(s, mu_vals, lw=2.5, color=color, label=label)

        # Peak marker
        ax.plot(s_op, mu_op, "o", ms=8, color=color, zorder=6)

        # Annotation offset varies per curve to avoid overlap
        offsets = {"dry": (0.03, 0.04), "wet": (0.03, 0.03),
                   "snow": (0.005, 0.02), "ice": (0.003, 0.012)}
        dx, dy = offsets.get(key, (0.02, 0.02))
        ax.annotate(
            f"μ_peak={mu_op:.3f}\ns_opt={s_op:.3f}",
            xy=(s_op, mu_op),
            xytext=(s_op + dx, mu_op + dy),
            fontsize=8.5, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
        )

    ax.set_xlabel("Wheel slip ratio  s", fontsize=12)
    ax.set_ylabel("Friction coefficient  μ(s)", fontsize=12)
    ax.set_title("Burckhardt Tire-Road Friction Model — All Surfaces", fontsize=14)
    ax.set_xlim(0, 0.52)
    ax.set_ylim(-0.02, 1.45)
    ax.legend(fontsize=11, loc="upper right")

    out = SAVE_DIR / "01_surfaces_overview.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================================
# PLOT 2 — 2×2 Model Comparison (GT vs NLS vs GP vs NN)
# ===========================================================================

def plot_02_model_comparison(all_results):
    """
    2×2 grid — one panel per surface.
    Each panel: ground-truth curve, noisy train/test scatter, and all 3 model fits.
    """
    s_dense = np.linspace(0.001, 0.50, 500)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, res, color in zip(axes, all_results, COLORS):
        # Ground truth
        if res["gt_c1"] is not None:
            mu_gt = bmu(s_dense, res["gt_c1"], res["gt_c2"], res["gt_c3"])
            ax.plot(s_dense, mu_gt, color=color, lw=2.5,
                    label="Ground truth", zorder=5)

        # Training scatter
        ax.scatter(res["slip_train"], res["mu_train"],
                   s=22, color=color, alpha=0.55,
                   label=f"Train (n={len(res['slip_train'])})", zorder=4)

        # Test scatter (×)
        ax.scatter(res["slip_test"], res["mu_test"],
                   s=16, color=color, marker="x", alpha=0.65,
                   linewidths=0.8,
                   label=f"Test  (n={len(res['slip_test'])})", zorder=4)

        # Burckhardt NLS
        bf = res["burckhardt"]
        if bf.c1 is not None:
            mu_bf = bf.predict(s_dense)
            ax.plot(s_dense, mu_bf, color="black", lw=1.8, ls="--",
                    label=f"Burckhardt NLS  RMSE={bf.rmse:.4f}", zorder=6)
            s_op = s_opt(bf.c1, bf.c2, bf.c3)
            ax.axvline(s_op, color="black", lw=0.8, ls=":", alpha=0.5)

        # GP ±2σ
        gp = res["gp"]
        try:
            mu_gp, sig_gp = gp.predict(s_dense)
            ax.plot(s_dense, mu_gp, color="#E67E22", lw=1.6, ls="-.",
                    label=f"Gaussian Process  RMSE={gp.rmse:.4f}", zorder=5)
            ax.fill_between(s_dense,
                            mu_gp - 2 * sig_gp, mu_gp + 2 * sig_gp,
                            color="#E67E22", alpha=0.10, label="GP ±2σ")
        except Exception:
            pass

        # Neural Network
        nn = res["nn"]
        try:
            mu_nn = nn.predict(s_dense)
            ax.plot(s_dense, mu_nn, color="#9B59B6", lw=1.4, ls=":",
                    label=f"Neural Network  RMSE={nn.rmse:.4f}", zorder=4)
        except Exception:
            pass

        ax.set_title(res["surface"], color=color)
        ax.set_xlabel("Slip  s")
        ax.set_ylabel("μ(s)")
        ax.set_xlim(0, 0.52)
        ax.set_ylim(bottom=-0.01)
        ax.legend(fontsize=7.5, loc="upper right")

    fig.suptitle(
        "μ(s) Prediction: Burckhardt NLS  vs  Gaussian Process  vs  Neural Network\n"
        "(50 training points, σ_noise = 0.01)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    out = SAVE_DIR / "02_model_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================================
# PLOT 3 — RMSE Bar Chart
# ===========================================================================

def plot_03_rmse_bar(all_results):
    """Grouped bar chart comparing model RMSE across all surfaces."""
    surfaces  = [r["surface"] for r in all_results]
    methods   = ["Burckhardt NLS", "Gaussian Process", "Neural Network"]
    keys      = ["burckhardt", "gp", "nn"]
    bar_colors = ["#2C3E50", "#E67E22", "#9B59B6"]

    rmse_table = np.zeros((len(methods), len(surfaces)))
    for j, res in enumerate(all_results):
        for i, key in enumerate(keys):
            m = res[key]
            if m is not None and m.rmse is not None:
                rmse_table[i, j] = m.rmse

    x     = np.arange(len(surfaces))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (method, c) in enumerate(zip(methods, bar_colors)):
        bars = ax.bar(x + i * width, rmse_table[i], width,
                      label=method, color=c, alpha=0.87,
                      edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, rmse_table[i]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0002,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=8, color=c, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(surfaces, fontsize=12)
    ax.set_ylabel("Test RMSE  (friction coefficient)", fontsize=12)
    ax.set_title("Model Accuracy by Road Surface  (50 train / ~250 test)", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(float(rmse_table.max()) * 1.4, 1e-3))

    out = SAVE_DIR / "03_rmse_comparison.png"
    plt.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================================
# PLOT 4 — Data Efficiency: RMSE vs n_train
# ===========================================================================

def plot_04_data_efficiency():
    """
    Shows how fast each method converges as training data grows.
    Physics (Burckhardt NLS) wins at small n; gap closes at large n.
    Key message: physics-informed ML is essential when data is scarce.
    """
    n_train_values = [5, 10, 20, 50, 100, 200]
    # Use two contrasting surfaces: dry (easy) and ice (sharp peak, hard)
    test_keys  = ["dry", "ice"]
    test_labels = ["Dry asphalt", "Ice"]
    test_colors = [SURFACE_COLORS["Dry asphalt"], SURFACE_COLORS["Ice"]]

    methods   = ["Burckhardt NLS", "Gaussian Process", "Neural Network"]
    model_ls  = ["-", "--", ":"]
    model_mk  = ["o", "s", "^"]
    model_col = ["#2C3E50", "#E67E22", "#9B59B6"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, key, label, surf_color in zip(axes, test_keys, test_labels, test_colors):
        rmse_per_method = {m: [] for m in methods}

        for n in n_train_values:
            df = generate_surface(key, n_samples=max(300, n + 100),
                                  noise_std=0.01, seed=7)
            try:
                res = train_and_evaluate(df, SURFACES[key].name, n_train=n)
                rmse_per_method["Burckhardt NLS"].append(res["burckhardt"].rmse or np.nan)
                rmse_per_method["Gaussian Process"].append(res["gp"].rmse or np.nan)
                rmse_per_method["Neural Network"].append(res["nn"].rmse or np.nan)
            except Exception:
                for m in methods:
                    rmse_per_method[m].append(np.nan)

        for method, ls, mk, mc in zip(methods, model_ls, model_mk, model_col):
            vals = np.array(rmse_per_method[method], dtype=float)
            ax.plot(n_train_values, vals,
                    ls=ls, marker=mk, ms=7,
                    color=mc, lw=2, label=method)

        ax.set_xlabel("Number of training samples", fontsize=12)
        ax.set_ylabel("Test RMSE", fontsize=12)
        ax.set_title(label, color=surf_color, fontsize=13)
        ax.set_xticks(n_train_values)
        ax.legend(fontsize=10)

    fig.suptitle(
        "Data Efficiency: RMSE vs Training Set Size\n"
        "Physics-informed NLS converges fastest — critical when test data is scarce",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    out = SAVE_DIR / "04_data_efficiency.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================================
# PLOT 5 — ESC Identifier Convergence Simulation
# ===========================================================================

def plot_05_identifier_convergence():
    """
    Simulates the ESC_TwoPoint_ID running in real-time.
    Shows c1, c2, c3 estimates converging toward ground truth as
    the ESC probe explores the slip range.

    Setup:
      - True surface: Dry asphalt  (c1=1.2801, c2=23.99, c3=0.52)
      - Initial guess: Wet asphalt (c1=0.857,  c2=33.82, c3=0.347)
      - ESC dither sweeps slip from 0.01 to 0.30 over T timesteps
    """
    np.random.seed(0)

    # Ground truth (dry asphalt)
    C1_TRUE, C2_TRUE, C3_TRUE = 1.2801, 23.99, 0.52
    # Starting guess (wet asphalt — wrong surface)
    c1, c2, c3 = 0.857, 33.82, 0.347

    # ESC parameters
    a_esc   = 0.02
    T       = 300       # timesteps
    dt      = 0.01      # s per timestep  (100 Hz)
    t_start = 0.5       # warmup period

    params = np.array([
        8,       # buf_size
        0,       # (unused)
        0.008,   # fark_threshold
        0,       # (unused)
        t_start, # t_start
        0.3,     # c1_min
        2.0,     # c1_max
        10.0,    # c2_min
        350.0,   # c2_max
        0.001,   # c3_min
        1.0,     # c3_max
        0.8,     # mu_buf_init
        0.05,    # s_buf_init
        1e-4,    # brent_tol
        20,      # brent_iter
    ])

    identifier = ESCTwoPointID()

    c1_hist, c2_hist, c3_hist   = [], [], []
    s_opt_hist, mu_peak_hist     = [], []
    valid_hist, flag_hist        = [], []
    t_hist                       = []

    for k in range(T):
        t = k * dt

        # ESC probe: slow sweep from s=0.01 to s=0.30 and back
        period = T * dt / 2.0
        s_probe = 0.01 + 0.29 * abs(np.sin(np.pi * t / period))

        # True friction and gradient at s_probe
        mu_true   = float(bmu(s_probe, C1_TRUE, C2_TRUE, C3_TRUE))
        g_true    = float(bgrad(s_probe, C1_TRUE, C2_TRUE, C3_TRUE))

        # ESC gradient estimate: g_esc = -g_true * a_esc / 2  (+noise)
        g_esc = -g_true * a_esc / 2.0 + np.random.normal(0, 0.0005)

        # Noisy friction measurement
        mu_meas  = mu_true + np.random.normal(0, 0.01)
        s_meas   = s_probe + np.random.normal(0, 0.002)

        result = identifier.identify(
            mu_measured=mu_meas,
            s_measured=s_meas,
            s_probe=s_probe,
            c1_prev=c1,
            c2_prev=c2,
            c3_prev=c3,
            t=t,
            params=params,
            g_esc=g_esc,
            a_esc=a_esc,
        )

        c1, c2, c3 = result[0], result[1], result[2]
        c1_hist.append(c1)
        c2_hist.append(c2)
        c3_hist.append(c3)
        s_opt_hist.append(result[3])
        mu_peak_hist.append(result[12])
        valid_hist.append(result[5])
        flag_hist.append(result[6])
        t_hist.append(t)

    t_arr   = np.array(t_hist)
    c1_arr  = np.array(c1_hist)
    c2_arr  = np.array(c2_hist)
    c3_arr  = np.array(c3_hist)
    sop_arr = np.array(s_opt_hist)
    mup_arr = np.array(mu_peak_hist)
    val_arr = np.array(valid_hist)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panel_data = [
        (axes[0, 0], c1_arr,  C1_TRUE, "c1 (amplitude)",       "#E74C3C"),
        (axes[0, 1], c2_arr,  C2_TRUE, "c2 (sharpness)",        "#3498DB"),
        (axes[0, 2], c3_arr,  C3_TRUE, "c3 (descending slope)", "#2ECC71"),
        (axes[1, 0], sop_arr, float(np.log(C1_TRUE * C2_TRUE / C3_TRUE) / C2_TRUE),
                              "s_opt (optimal slip)",    "#E67E22"),
        (axes[1, 1], mup_arr, float(bmu(
            np.log(C1_TRUE * C2_TRUE / C3_TRUE) / C2_TRUE,
            C1_TRUE, C2_TRUE, C3_TRUE)),
                              "μ_peak (peak friction)",  "#9B59B6"),
    ]

    for ax, est, gt, title, color in panel_data:
        ax.plot(t_arr, est, color=color, lw=1.5, label="Identified", alpha=0.85)
        ax.axhline(gt, color="black", lw=1.2, ls="--", label=f"True = {gt:.4f}")
        # Shade valid identification points
        ax.fill_between(t_arr, ax.get_ylim()[0] if ax.get_ylim()[0] > -1 else 0,
                        np.where(val_arr == 1, gt, np.nan),
                        alpha=0.0)
        ax.set_title(title, color=color)
        ax.set_xlabel("Time  t  [s]")
        ax.legend(fontsize=9)

    # 6th panel: layer activity (debug flag)
    ax6 = axes[1, 2]
    flag_arr = np.array(flag_hist)
    flag_colors = {0: "#27AE60", 1: "#BDC3C7", 10: "#E67E22", 20: "#E74C3C"}
    flag_labels = {0: "Brent+Gradient", 1: "Warmup",
                   10: "LUT+Gradient", 20: "Fallback"}
    for flag_val, fc in flag_colors.items():
        mask = flag_arr == flag_val
        if mask.any():
            ax6.scatter(t_arr[mask], np.full(mask.sum(), flag_val),
                        c=fc, s=8, label=flag_labels[flag_val], zorder=3)
    ax6.set_yticks([0, 1, 10, 20])
    ax6.set_yticklabels(["Brent+Grad", "Warmup", "LUT+Grad", "Fallback"], fontsize=9)
    ax6.set_xlabel("Time  t  [s]")
    ax6.set_title("Active Identification Layer")
    ax6.legend(fontsize=8, loc="right")

    fig.suptitle(
        "ESC_TwoPoint_ID — Real-Time Convergence Simulation\n"
        "Initial guess: Wet asphalt  →  True surface: Dry asphalt",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    out = SAVE_DIR / "05_identifier_convergence.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================================
# PLOTS 15 & 16 — Classifier Robustness: Macro F1 vs Sensor Noise
# ===========================================================================

# Prototypes used for classification (same as app.py + index.html)
_ROB_PROTOTYPES = {
    "Dry asphalt": np.array([1.2801, 23.99, 0.52]),
    "Wet asphalt": np.array([0.857,  33.82, 0.347]),
    "Snow":        np.array([0.1946, 94.13, 0.0646]),
}
_ROB_SURF_KEYS  = ["dry", "wet", "snow"]
_ROB_SURF_LBLS  = ["Dry asphalt", "Wet asphalt", "Snow"]
_ROB_NORM_MIN   = np.array([0.1946, 23.99, 0.0646])
_ROB_NORM_RNG   = np.array([1.2801 - 0.1946, 94.13 - 23.99, 0.52 - 0.0646])
_ROB_SIGMA      = [0.000, 0.005, 0.010, 0.020, 0.030, 0.050, 0.070, 0.100]
_ROB_BATCHES    = 60   # batches per (sigma × surface)
_ROB_SEEDS      = [[1.28, 24.0, 0.52], [0.86, 34.0, 0.35], [0.19, 94.0, 0.065]]

_ROB_VARIANTS = {
    "Baseline  (batch=50, 3D NN)":       {"batch": 50,  "mode": "3d", "vote": 1},
    "Large batch ×2  (batch=100, 3D NN)": {"batch": 100, "mode": "3d", "vote": 1},
    "c₂-only  (batch=50)":              {"batch": 50,  "mode": "c2", "vote": 1},
    "Majority vote ×3  (batch=50, 3D NN)":{"batch": 50,  "mode": "3d", "vote": 3},
}

_ROB_VARIANT_COLORS = ["#2C3E50", "#E74C3C", "#27AE60", "#9B59B6"]
_ROB_VARIANT_LS     = ["-", "--", "-.", ":"]


def _rob_burckhardt(s, c1, c2, c3):
    return c1 * (1.0 - np.exp(-c2 * np.abs(s))) - c3 * np.abs(s)


def _rob_fit(slips, mus):
    """NLS Burckhardt fit with 3 seeds. Returns (c1, c2, c3) or None."""
    best_params, best_resid = None, np.inf
    for seed in _ROB_SEEDS:
        try:
            popt, _ = curve_fit(
                _rob_burckhardt, slips, mus,
                p0=seed,
                bounds=([0.0, 0.0, 0.0], [3.0, 400.0, 3.0]),
                maxfev=800,
            )
            resid = float(np.mean((_rob_burckhardt(slips, *popt) - mus) ** 2))
            if resid < best_resid:
                best_resid = resid
                best_params = popt
        except Exception:
            pass
    return best_params


def _rob_classify_3d(c1, c2, c3):
    """Nearest-neighbour in normalised (c1,c2,c3) space."""
    vec = (np.array([c1, c2, c3]) - _ROB_NORM_MIN) / _ROB_NORM_RNG
    best_label, best_dist = None, np.inf
    for label, proto in _ROB_PROTOTYPES.items():
        d = float(np.linalg.norm(vec - (proto - _ROB_NORM_MIN) / _ROB_NORM_RNG))
        if d < best_dist:
            best_dist, best_label = d, label
    return best_label


def _rob_classify_c2(c2):
    """Classify using c2 distance alone."""
    c2_refs = {"Dry asphalt": 23.99, "Wet asphalt": 33.82, "Snow": 94.13}
    c2_range = 94.13 - 23.99
    best_label, best_dist = None, np.inf
    for label, ref in c2_refs.items():
        d = abs(c2 - ref) / c2_range
        if d < best_dist:
            best_dist, best_label = d, label
    return best_label


def _run_rob_mc(noise_axis, rng):
    """
    Monte Carlo robustness experiment.
    noise_axis: 'mu' or 'slip'
    Returns: dict  variant_name -> list[macro_f1]  (one entry per sigma level)
    """
    all_f1 = {name: [] for name in _ROB_VARIANTS}

    for sigma in _ROB_SIGMA:
        for name, cfg in _ROB_VARIANTS.items():
            batch_size   = cfg["batch"]
            classify_mode = cfg["mode"]
            vote_window  = cfg["vote"]

            tp = {lbl: 0 for lbl in _ROB_SURF_LBLS}
            fp = {lbl: 0 for lbl in _ROB_SURF_LBLS}
            fn = {lbl: 0 for lbl in _ROB_SURF_LBLS}

            # Per-surface vote buffers: majority vote is temporal smoothing within
            # a single surface stream — must not mix votes across different surfaces.
            vote_bufs = {lbl: [] for lbl in _ROB_SURF_LBLS}

            for _ in range(_ROB_BATCHES):
                for sk, true_lbl in zip(_ROB_SURF_KEYS, _ROB_SURF_LBLS):
                    p     = SURFACES[sk]
                    slips = rng.uniform(0.01, 0.35, batch_size)
                    mu_clean = bmu(slips, p.c1, p.c2, p.c3)

                    if noise_axis == "mu":
                        mus = mu_clean + rng.normal(0, max(sigma, 1e-9), batch_size)
                        fit_slips = slips
                    else:
                        fit_slips = np.clip(
                            slips + rng.normal(0, max(sigma, 1e-9), batch_size),
                            0.001, 0.50,
                        )
                        mus = mu_clean   # no mu noise when sweeping slip noise

                    params = _rob_fit(fit_slips, mus)
                    if params is None:
                        fn[true_lbl] += 1
                        continue

                    c1_, c2_, c3_ = params

                    if classify_mode == "c2":
                        pred = _rob_classify_c2(c2_)
                    else:
                        raw = _rob_classify_3d(c1_, c2_, c3_)
                        if vote_window > 1:
                            vote_bufs[true_lbl].append(raw)
                            pred = Counter(
                                vote_bufs[true_lbl][-vote_window:]
                            ).most_common(1)[0][0]
                        else:
                            pred = raw

                    if pred == true_lbl:
                        tp[true_lbl] += 1
                    else:
                        fp[pred]     += 1
                        fn[true_lbl] += 1

            # Macro F1
            f1s = []
            for lbl in _ROB_SURF_LBLS:
                prec = tp[lbl] / max(tp[lbl] + fp[lbl], 1)
                rec  = tp[lbl] / max(tp[lbl] + fn[lbl], 1)
                f1   = 2 * prec * rec / max(prec + rec, 1e-9)
                f1s.append(f1)
            all_f1[name].append(float(np.mean(f1s)))

    return all_f1


def _draw_rob_plot(all_f1, noise_axis, out_path):
    """Shared drawing routine for plots 15 and 16."""
    sigma_arr = np.array(_ROB_SIGMA)
    fig, ax = plt.subplots(figsize=(10, 6))

    for (name, f1_vals), color, ls in zip(
        all_f1.items(), _ROB_VARIANT_COLORS, _ROB_VARIANT_LS
    ):
        ax.plot(sigma_arr, f1_vals,
                color=color, ls=ls, lw=2.2,
                marker="o", ms=6, label=name)

    ax.set_xlabel(
        "Friction noise  σ_μ" if noise_axis == "mu" else "Slip noise  σ_s",
        fontsize=12,
    )
    ax.set_ylabel("Macro F1  (3 surfaces)", fontsize=12)
    ax.set_title(
        f"Classifier Robustness vs {'Friction Noise  σ_μ' if noise_axis == 'mu' else 'Slip Noise  σ_s'}\n"
        f"Monte Carlo ({_ROB_BATCHES} batches × {len(_ROB_SIGMA)} σ levels × 3 surfaces)",
        fontsize=13,
    )
    ax.set_xlim(-0.002, 0.105)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xticks(_ROB_SIGMA)
    ax.axhline(1.0, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.legend(fontsize=10, loc="lower left")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_15_16_robustness():
    """
    Two-panel robustness comparison:
      15 — Macro F1 vs σ_μ (friction noise)
      16 — Macro F1 vs σ_s (slip noise)
    Each plot shows 4 classifier variants:
      Baseline, Large batch×2, c2-only, Majority vote×3.
    """
    rng = np.random.default_rng(99)

    print("     Running Monte Carlo for σ_μ sweep …")
    f1_mu   = _run_rob_mc("mu",   rng)
    _draw_rob_plot(f1_mu,   "mu",   SAVE_DIR / "15_robustness_noise_mu.png")

    print("     Running Monte Carlo for σ_s sweep …")
    f1_slip = _run_rob_mc("slip", rng)
    _draw_rob_plot(f1_slip, "slip", SAVE_DIR / "16_robustness_noise_slip.png")


# ===========================================================================
# MAIN — smoke test + generate all plots
# ===========================================================================

def main():
    print("=" * 60)
    print("  make_plots.py — Generating presentation plots")
    print(f"  Output directory: {SAVE_DIR.resolve()}")
    print("=" * 60)

    # ── Generate synthetic data once (shared across plots 2 & 3) ─────────
    print("\n[1/7]  Surfaces overview ...")
    plot_01_surfaces_overview()

    print("\n[2/7]  Generating training data + fitting all models ...")
    all_results = []
    for i, key in enumerate(SURFACE_KEYS):
        df = generate_surface(key, n_samples=300, noise_std=0.01, seed=42 + i)
        res = train_and_evaluate(df, SURFACES[key].name, n_train=50)
        all_results.append(res)

    print("\n[2/7]  Model comparison (2×2) ...")
    plot_02_model_comparison(all_results)

    print("\n[3/7]  RMSE bar chart ...")
    plot_03_rmse_bar(all_results)

    print("\n[4/7]  Data efficiency ...")
    plot_04_data_efficiency()

    print("\n[5/7]  Identifier convergence simulation ...")
    plot_05_identifier_convergence()

    print("\n[6-7/7]  Robustness comparison (Monte Carlo, ~60 s) ...")
    plot_15_16_robustness()

    print("\n" + "=" * 60)
    print(f"  All 7 plots saved to:  {SAVE_DIR.resolve()}")
    print("=" * 60)

    # ── Smoke-test summary ───────────────────────────────────────────────
    saved = sorted(SAVE_DIR.glob("*.png"))
    assert len(saved) >= 7, f"Expected ≥7 plots, found {len(saved)}"
    for p in saved:
        sz = p.stat().st_size
        assert sz > 20_000, f"{p.name} seems too small ({sz} bytes) — rendering may have failed"
        print(f"  OK  {p.name}  ({sz/1024:.0f} kB)")

    print("\n  SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
