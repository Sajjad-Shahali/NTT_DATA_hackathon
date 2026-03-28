"""
evaluate_unlabeled.py — Surface identification from unlabeled (slip, mu) data.

Given a batch of (slip, mu) measurements with NO surface label:
  1. Fits Burckhardt NLS  →  estimates c1, c2, c3
  2. Classifies surface   →  nearest neighbour in normalised parameter space
  3. Reports              →  predicted surface, confidence, mu_peak, s_opt
  4. Compares             →  predicted vs true label (accuracy over many batches)
  5. Saves                →  reports/plots/13_unlabeled_evaluation.png

Usage
-----
  python evaluate_unlabeled.py
  python evaluate_unlabeled.py --n-samples 30   # points per batch
  python evaluate_unlabeled.py --n-batches 20   # batches per surface
  python evaluate_unlabeled.py --csv data/raw/prepared_friction.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, str(Path(__file__).parent))
from src.burckhardt import mu as bmu, s_opt, mu_peak

# ---------------------------------------------------------------------------
# Known surfaces — the c_matrix "database"
# ---------------------------------------------------------------------------
C_MATRIX = {
    "Dry asphalt": np.array([1.2801,  23.990,  0.5200]),
    "Wet asphalt": np.array([0.8570,  33.822,  0.3470]),
    "Snow":        np.array([0.1946,  94.129,  0.0646]),
}

COLORS = {
    "Dry asphalt": "#E74C3C",
    "Wet asphalt": "#3498DB",
    "Snow":        "#95A5A6",
}

SAVE_DIR = Path("reports/plots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "savefig.dpi": 180, "savefig.bbox": "tight",
})

# Normalisation bounds for parameter space distance
_C_RANGES = np.array([
    1.2801 - 0.1946,   # c1 range
    94.129 - 23.990,   # c2 range
    0.5200 - 0.0646,   # c3 range
])


# ---------------------------------------------------------------------------
# Burckhardt NLS fit (unlabeled — no surface assumed)
# ---------------------------------------------------------------------------
def fit_burckhardt(slip: np.ndarray, mu_data: np.ndarray):
    """
    Fit c1, c2, c3 from raw (slip, mu) pairs.
    Returns (c1, c2, c3) or None if fit fails.
    """
    def model(s, c1, c2, c3):
        return c1 * (1.0 - np.exp(-c2 * np.abs(s))) - c3 * np.abs(s)

    # Wide bounds covering all known surfaces
    lo = [0.01,  10.0, 0.001]
    hi = [1.60, 310.0, 0.700]

    # Multiple starting points to avoid local minima
    best_params, best_cost = None, np.inf
    seeds = [
        [1.2, 24.0, 0.50],   # dry-ish
        [0.85, 34.0, 0.35],  # wet-ish
        [0.20, 94.0, 0.06],  # snow-ish
    ]
    for p0 in seeds:
        try:
            popt, _ = curve_fit(
                model, slip, mu_data,
                p0=p0, bounds=(lo, hi),
                maxfev=5000,
            )
            resid = np.sqrt(np.mean((model(slip, *popt) - mu_data) ** 2))
            if resid < best_cost:
                best_cost = resid
                best_params = popt
        except Exception:
            continue

    return best_params, best_cost


# ---------------------------------------------------------------------------
# Surface classification — nearest neighbour in normalised parameter space
# ---------------------------------------------------------------------------
def classify_surface(params: np.ndarray):
    """
    Returns (predicted_surface, confidence_dict) where confidence is
    1 - normalised_distance to each surface.
    """
    params_norm = params / _C_RANGES

    distances = {}
    for name, ref in C_MATRIX.items():
        ref_norm = ref / _C_RANGES
        distances[name] = float(np.linalg.norm(params_norm - ref_norm))

    sorted_names = sorted(distances, key=distances.get)
    best   = sorted_names[0]
    second = distances[sorted_names[1]]

    # Confidence: how much better than the runner-up (0..1)
    d_best = distances[best]
    total  = sum(distances.values())
    # Softmax-like score
    inv = {k: 1.0 / (v + 1e-9) for k, v in distances.items()}
    inv_sum = sum(inv.values())
    confidence = {k: v / inv_sum for k, v in inv.items()}

    return best, confidence, distances


# ---------------------------------------------------------------------------
# Single-batch demo (verbose)
# ---------------------------------------------------------------------------
def identify_batch(slip: np.ndarray, mu_data: np.ndarray,
                   true_surface: str = None):
    """
    Identify one batch of unlabeled (slip, mu) data.
    Prints a report and returns the result dict.
    """
    params, rmse = fit_burckhardt(slip, mu_data)
    if params is None:
        print("  [FAIL] Burckhardt fit did not converge.")
        return None

    c1, c2, c3 = params
    predicted, confidence, distances = classify_surface(params)
    sop = s_opt(c1, c2, c3)
    mup = mu_peak(c1, c2, c3)
    correct = (predicted == true_surface) if true_surface else None

    print(f"\n  Fitted parameters : c1={c1:.4f}  c2={c2:.2f}  c3={c3:.4f}")
    print(f"  Fit RMSE          : {rmse:.5f}")
    print(f"  Predicted surface : {predicted}  {'[CORRECT]' if correct else '[WRONG] true='+str(true_surface) if correct is False else ''}")
    print(f"  mu_peak           : {mup:.4f}")
    print(f"  s_opt             : {sop:.4f}")
    print(f"  Confidence        : " +
          "  ".join(f"{k}: {v:.1%}" for k, v in confidence.items()))
    print(f"  Distances         : " +
          "  ".join(f"{k}: {v:.4f}" for k, v in distances.items()))

    return {
        "params": params, "rmse": rmse,
        "predicted": predicted, "true": true_surface,
        "correct": correct,
        "confidence": confidence, "distances": distances,
        "sop": sop, "mup": mup,
    }


# ---------------------------------------------------------------------------
# Batch evaluation over many random draws
# ---------------------------------------------------------------------------
def batch_evaluation(df: pd.DataFrame, n_samples: int, n_batches: int, seed: int = 0):
    """
    Repeatedly draw n_samples from each surface (unlabeled),
    run the identifier, and collect accuracy stats.
    """
    rng = np.random.default_rng(seed)
    surfaces = list(C_MATRIX.keys())
    results = []

    print(f"\n{'='*60}")
    print(f"  Batch evaluation: {n_batches} batches x {n_samples} pts per surface")
    print(f"{'='*60}")

    for surf in surfaces:
        grp = df[df["surface"] == surf]
        correct_count = 0

        for i in range(n_batches):
            idx = rng.choice(len(grp), size=n_samples, replace=False)
            batch = grp.iloc[idx]
            res = identify_batch(
                batch["slip"].values,
                batch["mu_noisy"].values,
                true_surface=surf,
            )
            if res and res["correct"]:
                correct_count += 1
            if res:
                results.append(res)

        acc = correct_count / n_batches
        print(f"\n  {surf:20s}  accuracy = {acc:.0%}  ({correct_count}/{n_batches})")

    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_results(df: pd.DataFrame, results: list):
    """
    4-panel figure:
      [0] Fitted curves vs real data (one example per surface)
      [1] c1/c2/c3 scatter — fitted vs ground truth
      [2] Confusion matrix
      [3] RMSE distribution per surface
    """
    s_dense = np.linspace(0.001, 0.50, 500)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)
    ax_curves  = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_params  = fig.add_subplot(gs[1, 0])
    ax_cm      = fig.add_subplot(gs[1, 1])
    ax_rmse    = fig.add_subplot(gs[1, 2])

    surfaces = list(C_MATRIX.keys())
    rng = np.random.default_rng(99)

    # ── Panel row 0: one example fit per surface ──────────────────────────
    for ax, surf in zip(ax_curves, surfaces):
        color = COLORS[surf]
        grp = df[df["surface"] == surf]
        idx = rng.choice(len(grp), size=60, replace=False)
        batch = grp.iloc[idx]
        slip_b = batch["slip"].values
        mu_b   = batch["mu_noisy"].values

        # Real scatter
        ax.scatter(slip_b, mu_b, s=15, alpha=0.6, color=color,
                   label="Measured (unlabeled)")

        # Ground-truth model
        ref = C_MATRIX[surf]
        ax.plot(s_dense, bmu(s_dense, *ref), color="black",
                lw=2, ls="--", label="True model")

        # NLS fit from unlabeled data
        params, rmse = fit_burckhardt(slip_b, mu_b)
        if params is not None:
            c1, c2, c3 = params
            predicted, _, _ = classify_surface(params)
            ax.plot(s_dense, bmu(s_dense, c1, c2, c3),
                    color=color, lw=2, ls="-",
                    label=f"NLS fit -> {predicted}\nRMSE={rmse:.5f}")
            sop = s_opt(c1, c2, c3)
            mup = mu_peak(c1, c2, c3)
            ax.plot(sop, mup, "^", ms=9, color=color, zorder=6)

        ax.set_title(f"True: {surf}", color=color)
        ax.set_xlabel("Slip  s")
        ax.set_ylabel("mu(s)")
        ax.set_xlim(0, 0.52)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    # ── Panel [1,0]: c1 vs c2 scatter (fitted vs reference) ──────────────
    for surf, ref in C_MATRIX.items():
        color = COLORS[surf]
        ax_params.scatter(ref[0], ref[1], s=200, color=color,
                          marker="*", zorder=6, label=f"{surf} (true)")
    for res in results:
        if res["params"] is not None:
            c = COLORS.get(res["true"], "#555")
            marker = "o" if res["correct"] else "x"
            ax_params.scatter(res["params"][0], res["params"][1],
                              s=30, alpha=0.5, color=c, marker=marker, zorder=3)
    ax_params.set_xlabel("c1 (amplitude)")
    ax_params.set_ylabel("c2 (sharpness)")
    ax_params.set_title("Fitted c1 vs c2\n(* = true, o = correct, x = wrong)")
    ax_params.legend(fontsize=8)

    # ── Panel [1,1]: Confusion matrix ─────────────────────────────────────
    y_true = [r["true"]      for r in results if r["true"] is not None]
    y_pred = [r["predicted"] for r in results if r["true"] is not None]
    if y_true:
        cm = confusion_matrix(y_true, y_pred, labels=surfaces)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Dry", "Wet", "Snow"])
        disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
        ax_cm.set_title("Surface Classification\nConfusion Matrix")

    # ── Panel [1,2]: RMSE distribution per surface ────────────────────────
    for surf in surfaces:
        rmses = [r["rmse"] for r in results
                 if r["true"] == surf and r["rmse"] is not None]
        if rmses:
            ax_rmse.boxplot(rmses,
                            positions=[surfaces.index(surf)],
                            widths=0.5,
                            patch_artist=True,
                            boxprops=dict(facecolor=COLORS[surf], alpha=0.6),
                            medianprops=dict(color="black", lw=2),
                            whiskerprops=dict(color=COLORS[surf]),
                            capprops=dict(color=COLORS[surf]),
                            flierprops=dict(marker=".", color=COLORS[surf]),
                            )
    ax_rmse.set_xticks(range(len(surfaces)))
    ax_rmse.set_xticklabels(["Dry", "Wet", "Snow"], fontsize=10)
    ax_rmse.set_ylabel("Fit RMSE")
    ax_rmse.set_title("NLS Fit RMSE\nper Surface (distribution)")

    overall_acc = (sum(1 for r in results if r["correct"]) / len(results)
                   if results else 0)

    fig.suptitle(
        f"Unlabeled Surface Identification — Overall Accuracy: {overall_acc:.1%}\n"
        f"(n_samples per batch shown in per-surface subplots)",
        fontsize=14, y=1.01,
    )

    out = SAVE_DIR / "13_unlabeled_evaluation.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"\n  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="data/raw/prepared_friction.csv")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Points per unlabeled batch")
    parser.add_argument("--n-batches", type=int, default=15,
                        help="Batches per surface")
    args = parser.parse_args()

    print("=" * 60)
    print("  evaluate_unlabeled.py — Road Surface Identification")
    print("=" * 60)

    # Load data
    csv = args.csv
    if not Path(csv).exists():
        sys.exit(f"[ERROR] {csv} not found. Run preprocess.py first.")

    df = pd.read_csv(csv, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    # Quality filter
    if "abs_active" in df.columns:
        df = df[df["abs_active"] == 1]
    if "alpha" in df.columns:
        df = df[df["alpha"] > 0.01]
    print(f"  After quality filter: {len(df):,} rows")
    print(f"  Surfaces: {df['surface'].value_counts().to_dict()}")

    # ── Quick single-surface demo ─────────────────────────────────────────
    print("\n--- Single-batch demo (50 pts, true surface hidden from model) ---")
    for surf in C_MATRIX:
        grp = df[df["surface"] == surf]
        batch = grp.sample(50, random_state=42)
        identify_batch(batch["slip"].values, batch["mu_noisy"].values,
                       true_surface=surf)

    # ── Full batch evaluation ─────────────────────────────────────────────
    results = batch_evaluation(df, args.n_samples, args.n_batches)

    overall_acc = sum(1 for r in results if r["correct"]) / len(results)
    print(f"\n{'='*60}")
    print(f"  OVERALL ACCURACY:  {overall_acc:.1%}  "
          f"({sum(1 for r in results if r['correct'])}/{len(results)} batches)")
    print(f"{'='*60}")

    # ── Plot ──────────────────────────────────────────────────────────────
    out = plot_results(df, results)

    # ── Smoke test ────────────────────────────────────────────────────────
    p = Path(out)
    assert p.exists() and p.stat().st_size > 20_000, \
        f"Plot too small or missing: {p}"
    print(f"  OK  {p.name}  ({p.stat().st_size // 1024} kB)")
    print("  SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
