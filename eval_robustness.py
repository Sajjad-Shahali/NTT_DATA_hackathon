"""
eval_robustness.py — Robustness evaluation under realistic noise.

Replaces the ideal noise-free formula:
    mu_res = c1*(1 - exp(-c2*s_res)) - c3*s_res

With a three-source realistic noise model:
    s_meas  = s_true  + N(0, sigma_s)            # slip sensor noise
    mu_meas = mu_true + N(0, sigma_mu)            # friction sensor noise
    c_i     = c_i_nom * (1 + N(0, sigma_param))  # parameter drift (wear/temp)

Runs three independent sweeps to isolate each failure mode:
  A) Noise sweep     — sigma_mu from 0.0003 (current) to 0.10
  B) Slip-range sweep — max slip from 0.50 down to 0.02 (partial excitation)
  C) Parameter-drift  — c1/c2/c3 perturbed ±5%..±20% per batch

Saves:
  reports/plots/15_robustness_noise_sweep.png
  reports/plots/16_robustness_slip_range.png
  reports/plots/17_robustness_param_drift.png

Usage
-----
  python eval_robustness.py
  python eval_robustness.py --n-batches 30 --batch-size 50
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from src.burckhardt import mu as bmu, s_opt, mu_peak

# ---------------------------------------------------------------------------
# c_matrix database
# ---------------------------------------------------------------------------
C_MATRIX = {
    "Dry asphalt": np.array([1.2801,  23.990,  0.5200]),
    "Wet asphalt": np.array([0.8570,  33.822,  0.3470]),
    "Snow":        np.array([0.1946,  94.129,  0.0646]),
}
SURFACES = list(C_MATRIX.keys())
COLORS   = {"Dry asphalt": "#E74C3C", "Wet asphalt": "#3498DB", "Snow": "#95A5A6"}

_C_RANGES = np.array([
    C_MATRIX["Dry asphalt"][0] - C_MATRIX["Snow"][0],
    C_MATRIX["Snow"][1]        - C_MATRIX["Dry asphalt"][1],
    C_MATRIX["Dry asphalt"][2] - C_MATRIX["Snow"][2],
])

SAVE_DIR = Path("reports/plots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "savefig.dpi": 180, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Realistic noisy Burckhardt generator
# ---------------------------------------------------------------------------
def generate_noisy_batch(
    surface_key: str,
    n_points: int,
    sigma_mu: float    = 0.0003,   # friction measurement noise
    sigma_s: float     = 0.002,    # slip measurement noise
    sigma_param: float = 0.0,      # relative parameter drift (0.05 = ±5%)
    slip_max: float    = 0.50,     # max slip in batch (limits excitation range)
    slip_min: float    = 0.001,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Generate one batch of (slip_meas, mu_meas) for a given surface.

    Noise model:
        s_true  ~ Uniform(slip_min, slip_max)  sqrt-spaced (denser near 0)
        c_i_actual = c_i_nominal * (1 + N(0, sigma_param))   per-batch drift
        mu_true = Burckhardt(s_true, c1_actual, c2_actual, c3_actual)
        s_meas  = s_true  + N(0, sigma_s)
        mu_meas = mu_true + N(0, sigma_mu)

    Returns
    -------
    s_meas, mu_meas, (c1_actual, c2_actual, c3_actual)
    """
    if rng is None:
        rng = np.random.default_rng()

    c_nom = C_MATRIX[surface_key]

    # Per-batch parameter drift (temperature, tyre wear)
    if sigma_param > 0:
        drift = rng.normal(0, sigma_param, 3)
        c_actual = c_nom * (1.0 + drift)
        # Enforce physical constraints
        c_actual = np.clip(c_actual, [0.01, 5.0, 0.001], [2.0, 350.0, 1.0])
    else:
        c_actual = c_nom.copy()

    c1, c2, c3 = c_actual

    # Slip sampling — sqrt-spaced to give more points near s=0
    u = rng.uniform(0, 1, n_points)
    s_true = slip_min + (slip_max - slip_min) * np.sqrt(u)

    # True friction
    mu_true = bmu(s_true, c1, c2, c3)

    # Add noise
    s_meas  = s_true  + rng.normal(0, sigma_s,  n_points)
    mu_meas = mu_true + rng.normal(0, sigma_mu, n_points)

    # Physical clip
    s_meas  = np.clip(s_meas,  slip_min, slip_max + 0.05)
    mu_meas = np.clip(mu_meas, 0.0, None)

    return s_meas, mu_meas, c_actual


# ---------------------------------------------------------------------------
# Burckhardt NLS fit + nearest-neighbour classifier
# ---------------------------------------------------------------------------
def fit_and_classify(slip: np.ndarray, mu_data: np.ndarray):
    """Returns (predicted_surface, fitted_params) or (None, None) on failure."""
    def model(s, c1, c2, c3):
        return c1 * (1.0 - np.exp(-c2 * np.abs(s))) - c3 * np.abs(s)

    lo = [0.01,  5.0, 0.001]
    hi = [2.00, 350.0, 1.0]
    seeds = [
        [1.2, 24.0, 0.50],
        [0.85, 34.0, 0.35],
        [0.20, 94.0, 0.06],
    ]
    best_params, best_cost = None, np.inf
    for p0 in seeds:
        try:
            popt, _ = curve_fit(model, slip, mu_data,
                                p0=p0, bounds=(lo, hi), maxfev=5000)
            cost = np.sqrt(np.mean((model(slip, *popt) - mu_data) ** 2))
            if cost < best_cost:
                best_cost, best_params = cost, popt
        except Exception:
            continue

    if best_params is None:
        return None, None

    params_norm = best_params / _C_RANGES
    best_name, best_dist = None, np.inf
    for name, ref in C_MATRIX.items():
        d = np.linalg.norm(params_norm - ref / _C_RANGES)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name, best_params


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
def run_sweep(
    param_values,
    param_name: str,
    get_kwargs,          # callable: param_value → dict of generate_noisy_batch kwargs
    n_batches: int,
    batch_size: int,
    seed: int = 0,
):
    """
    For each value in param_values, run n_batches × 3 surfaces,
    return dict: value → {surface → (precision, recall, f1, n_correct, n_total)}
    and macro_f1 list.
    """
    rng = np.random.default_rng(seed)
    results = {}
    macro_f1s = []
    per_surface_f1 = {s: [] for s in SURFACES}

    for val in param_values:
        kwargs = get_kwargs(val)
        y_true, y_pred = [], []

        for surf in SURFACES:
            for _ in range(n_batches):
                s_b, mu_b, _ = generate_noisy_batch(
                    surf, batch_size, rng=rng, **kwargs)
                pred, _ = fit_and_classify(s_b, mu_b)
                if pred is None:
                    pred = "FAIL"
                y_true.append(surf)
                y_pred.append(pred)

        macro = f1_score(y_true, y_pred, labels=SURFACES,
                         average="macro", zero_division=0)
        macro_f1s.append(macro)

        for surf in SURFACES:
            f1_s = f1_score(y_true, y_pred, labels=[surf],
                            average="macro", zero_division=0)
            per_surface_f1[surf].append(f1_s)

        results[val] = {"y_true": y_true, "y_pred": y_pred, "macro_f1": macro}
        print(f"  {param_name}={val:<8}  macro_F1={macro:.3f}  "
              + "  ".join(f"{s[:3]}={per_surface_f1[s][-1]:.2f}"
                          for s in SURFACES))

    return results, macro_f1s, per_surface_f1


# ---------------------------------------------------------------------------
# SWEEP A — Measurement noise on mu
# ---------------------------------------------------------------------------
def sweep_noise(n_batches, batch_size):
    sigma_values = [0.0003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

    print(f"\n{'='*60}")
    print("  SWEEP A — sigma_mu noise on friction measurement")
    print(f"  (slip range = 0 to 0.50, batch={batch_size}, n_batches={n_batches}/surface)")
    print(f"{'='*60}")

    results, macro_f1s, per_f1 = run_sweep(
        param_values=sigma_values,
        param_name="sigma_mu",
        get_kwargs=lambda v: {"sigma_mu": v, "sigma_s": 0.002,
                              "sigma_param": 0.0, "slip_max": 0.50},
        n_batches=n_batches,
        batch_size=batch_size,
    )
    return sigma_values, macro_f1s, per_f1


# ---------------------------------------------------------------------------
# SWEEP B — Partial slip range (limited excitation = real driving)
# ---------------------------------------------------------------------------
def sweep_slip_range(n_batches, batch_size):
    slip_maxes = [0.50, 0.30, 0.20, 0.10, 0.07, 0.05, 0.03]

    print(f"\n{'='*60}")
    print("  SWEEP B — Maximum slip in batch (limited excitation)")
    print(f"  (sigma_mu=0.01, batch={batch_size}, n_batches={n_batches}/surface)")
    print(f"{'='*60}")

    results, macro_f1s, per_f1 = run_sweep(
        param_values=slip_maxes,
        param_name="slip_max",
        get_kwargs=lambda v: {"sigma_mu": 0.01, "sigma_s": 0.002,
                              "sigma_param": 0.0, "slip_max": v},
        n_batches=n_batches,
        batch_size=batch_size,
    )
    return slip_maxes, macro_f1s, per_f1


# ---------------------------------------------------------------------------
# SWEEP C — Parameter drift (temperature / tyre wear)
# ---------------------------------------------------------------------------
def sweep_param_drift(n_batches, batch_size):
    drift_values = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]

    print(f"\n{'='*60}")
    print("  SWEEP C — Per-batch parameter drift (c1/c2/c3 vary ±sigma_param)")
    print(f"  (sigma_mu=0.01, slip_max=0.50, batch={batch_size})")
    print(f"{'='*60}")

    results, macro_f1s, per_f1 = run_sweep(
        param_values=drift_values,
        param_name="sigma_p",
        get_kwargs=lambda v: {"sigma_mu": 0.01, "sigma_s": 0.002,
                              "sigma_param": v, "slip_max": 0.50},
        n_batches=n_batches,
        batch_size=batch_size,
    )
    return drift_values, macro_f1s, per_f1


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _annotate_threshold(ax, x_vals, y_vals, threshold=0.90):
    """Draw vertical line at the first x where F1 drops below threshold."""
    for i, (x, y) in enumerate(zip(x_vals, y_vals)):
        if y < threshold:
            ax.axvline(x, color="red", lw=1.2, ls="--", alpha=0.7,
                       label=f"F1 < {threshold} at x={x}")
            ax.text(x * 1.02, threshold + 0.01, f"F1<{threshold}\n@{x}",
                    color="red", fontsize=8)
            break


def plot_sweep_A(sigma_vals, macro_f1s, per_f1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: macro F1
    ax = axes[0]
    ax.plot(sigma_vals, macro_f1s, "k-o", lw=2, ms=7, label="Macro F1")
    ax.axhline(1.0,  color="grey",  lw=0.8, ls=":", alpha=0.5)
    ax.axhline(0.90, color="orange", lw=1.0, ls="--", alpha=0.8,
               label="F1 = 0.90")
    _annotate_threshold(ax, sigma_vals, macro_f1s)
    ax.set_xlabel("sigma_mu — friction measurement noise (std)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Noise Robustness\n(full slip range 0–0.50)")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xscale("log")

    # Right: per-surface F1
    ax2 = axes[1]
    for surf in SURFACES:
        ax2.plot(sigma_vals, per_f1[surf], "o-",
                 color=COLORS[surf], lw=2, ms=6, label=surf)
    ax2.axhline(0.90, color="orange", lw=1, ls="--", alpha=0.7)
    ax2.set_xlabel("sigma_mu — friction measurement noise (std)")
    ax2.set_ylabel("Per-class F1")
    ax2.set_title("Per-Surface F1 vs Noise Level")
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.05, 1.10)
    ax2.set_xscale("log")

    fig.suptitle(
        "SWEEP A — Effect of Friction Measurement Noise\n"
        "Current simulation noise = 0.0003  |  Realistic sensor noise = 0.01–0.05",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = SAVE_DIR / "15_robustness_noise_sweep.png"
    fig.savefig(out); plt.close(fig)
    print(f"\n  Saved: {out}")
    return out


def plot_sweep_B(slip_vals, macro_f1s, per_f1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(slip_vals, macro_f1s, "k-o", lw=2, ms=7, label="Macro F1")
    ax.axhline(0.90, color="orange", lw=1, ls="--", alpha=0.8, label="F1 = 0.90")
    _annotate_threshold(ax, slip_vals[::-1], macro_f1s[::-1])
    ax.set_xlabel("Max slip in batch (slip_max)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Slip Range Robustness\n(sigma_mu=0.01)")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.10)
    ax.axvspan(0, 0.05, alpha=0.08, color="red",
               label="Normal driving zone (s<0.05)")
    ax.text(0.025, 0.15, "Normal\ndriving", ha="center", fontsize=8, color="red")

    ax2 = axes[1]
    for surf in SURFACES:
        ax2.plot(slip_vals, per_f1[surf], "o-",
                 color=COLORS[surf], lw=2, ms=6, label=surf)
    ax2.axhline(0.90, color="orange", lw=1, ls="--", alpha=0.7)
    ax2.axvspan(0, 0.05, alpha=0.08, color="red")
    ax2.set_xlabel("Max slip in batch (slip_max)")
    ax2.set_ylabel("Per-class F1")
    ax2.set_title("Per-Surface F1 vs Slip Range")
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.05, 1.10)

    fig.suptitle(
        "SWEEP B — Effect of Slip Range Limitation\n"
        "Real driving rarely exceeds s=0.05 (red zone = near-zero info about peak)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = SAVE_DIR / "16_robustness_slip_range.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_sweep_C(drift_vals, macro_f1s, per_f1):
    drift_pct = [v * 100 for v in drift_vals]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(drift_pct, macro_f1s, "k-o", lw=2, ms=7, label="Macro F1")
    ax.axhline(0.90, color="orange", lw=1, ls="--", alpha=0.8, label="F1 = 0.90")
    _annotate_threshold(ax, drift_pct, macro_f1s)
    ax.set_xlabel("Parameter drift sigma (% of nominal c1/c2/c3)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Parameter Drift Robustness\n(sigma_mu=0.01, slip_max=0.50)")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.10)

    ax2 = axes[1]
    for surf in SURFACES:
        ax2.plot(drift_pct, per_f1[surf], "o-",
                 color=COLORS[surf], lw=2, ms=6, label=surf)
    ax2.axhline(0.90, color="orange", lw=1, ls="--", alpha=0.7)
    ax2.set_xlabel("Parameter drift sigma (%)")
    ax2.set_ylabel("Per-class F1")
    ax2.set_title("Per-Surface F1 vs Parameter Drift")
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.05, 1.10)

    fig.suptitle(
        "SWEEP C — Effect of Parameter Drift (tyre wear / temperature)\n"
        "c1, c2, c3 perturbed per-batch by N(0, sigma_param * c_nominal)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = SAVE_DIR / "17_robustness_param_drift.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Summary plot — all three sweeps on one figure
# ---------------------------------------------------------------------------
def plot_summary(
    sigma_vals, macro_A,
    slip_vals,  macro_B,
    drift_vals, macro_C,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # A
    axes[0].plot(sigma_vals, macro_A, "k-o", lw=2, ms=7)
    axes[0].axhline(0.90, color="orange", lw=1.2, ls="--", label="F1=0.90")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("sigma_mu (log scale)")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("A: Sensor Noise")
    axes[0].set_ylim(-0.05, 1.10)
    axes[0].axvline(0.0003, color="green", lw=1, ls=":", label="current sim")
    axes[0].axvline(0.01,   color="red",   lw=1, ls=":", label="realistic")
    axes[0].legend(fontsize=8)

    # B
    axes[1].plot(slip_vals, macro_B, "k-o", lw=2, ms=7)
    axes[1].axhline(0.90, color="orange", lw=1.2, ls="--")
    axes[1].axvspan(0, 0.05, alpha=0.08, color="red")
    axes[1].set_xlabel("slip_max")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("B: Slip Range Limitation")
    axes[1].set_ylim(-0.05, 1.10)

    # C
    drift_pct = [v * 100 for v in drift_vals]
    axes[2].plot(drift_pct, macro_C, "k-o", lw=2, ms=7)
    axes[2].axhline(0.90, color="orange", lw=1.2, ls="--")
    axes[2].set_xlabel("Parameter drift (%)")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title("C: Parameter Drift")
    axes[2].set_ylim(-0.05, 1.10)

    fig.suptitle(
        "Robustness Summary — Where Does the Classifier Break?\n"
        "Burckhardt NLS + Nearest-Neighbour Classifier",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    out = SAVE_DIR / "18_robustness_summary.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-batches",  type=int, default=20,
                        help="Batches per surface per sweep point (default 20)")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Points per batch (default 50)")
    args = parser.parse_args()

    print("=" * 60)
    print("  eval_robustness.py — Realistic Noise Robustness Evaluation")
    print("=" * 60)
    print(f"\n  Noise model:")
    print(f"    mu_meas = Burckhardt(s_true, c_actual) + N(0, sigma_mu)")
    print(f"    s_meas  = s_true + N(0, sigma_s=0.002)")
    print(f"    c_actual = c_nominal * (1 + N(0, sigma_param))  [sweep C only]")
    print(f"\n  n_batches={args.n_batches}/surface  batch_size={args.batch_size}")

    sigma_vals, macro_A, per_A = sweep_noise(args.n_batches, args.batch_size)
    slip_vals,  macro_B, per_B = sweep_slip_range(args.n_batches, args.batch_size)
    drift_vals, macro_C, per_C = sweep_param_drift(args.n_batches, args.batch_size)

    print(f"\n[Plots]")
    o1 = plot_sweep_A(sigma_vals, macro_A, per_A)
    o2 = plot_sweep_B(slip_vals,  macro_B, per_B)
    o3 = plot_sweep_C(drift_vals, macro_C, per_C)
    o4 = plot_summary(sigma_vals, macro_A, slip_vals, macro_B, drift_vals, macro_C)

    # Smoke test
    print(f"\n--- Smoke test ---")
    ok = True
    for p in [o1, o2, o3, o4]:
        sz = Path(p).stat().st_size
        if sz < 20_000:
            print(f"  FAIL  {Path(p).name}  ({sz} bytes)")
            ok = False
        else:
            print(f"  OK    {Path(p).name}  ({sz // 1024} kB)")

    if ok:
        print("\n  SMOKE TEST PASSED")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  ROBUSTNESS SUMMARY")
    print(f"{'='*60}")
    print(f"\n  A) Sensor noise (sigma_mu) vs Macro F1:")
    for v, f in zip(sigma_vals, macro_A):
        bar = "#" * int(f * 20)
        tag = " <- current sim" if v == 0.0003 else " <- realistic" if v == 0.01 else ""
        print(f"    sigma={v:.4f}  F1={f:.3f}  {bar}{tag}")

    print(f"\n  B) Slip range (slip_max) vs Macro F1:")
    for v, f in zip(slip_vals, macro_B):
        bar = "#" * int(f * 20)
        tag = " <- normal driving" if v <= 0.05 else ""
        print(f"    slip_max={v:.2f}   F1={f:.3f}  {bar}{tag}")

    print(f"\n  C) Parameter drift (%) vs Macro F1:")
    for v, f in zip(drift_vals, macro_C):
        bar = "#" * int(f * 20)
        print(f"    drift={v*100:4.0f}%    F1={f:.3f}  {bar}")


if __name__ == "__main__":
    main()
