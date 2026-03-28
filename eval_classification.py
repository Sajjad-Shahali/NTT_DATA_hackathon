"""
eval_classification.py — Generate unlabeled evaluation set and score it.

Steps
-----
1. Split prepared_friction.csv into train / eval (80/20 stratified by surface)
2. Strip surface labels from eval set  →  unlabeled_eval.csv
3. Run Burckhardt NLS + nearest-neighbour classifier on every batch
4. Compute Precision, Recall, F1 (per class + macro + weighted)
5. Save confusion matrix + per-class bar chart to reports/plots/

Usage
-----
  python eval_classification.py
  python eval_classification.py --batch-size 30
  python eval_classification.py --csv data/raw/prepared_friction.csv
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from src.burckhardt import mu as bmu, s_opt, mu_peak

# ---------------------------------------------------------------------------
# c_matrix "database"
# ---------------------------------------------------------------------------
C_MATRIX = {
    "Dry asphalt": np.array([1.2801,  23.990,  0.5200]),
    "Wet asphalt": np.array([0.8570,  33.822,  0.3470]),
    "Snow":        np.array([0.1946,  94.129,  0.0646]),
}
SURFACES = list(C_MATRIX.keys())

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

_C_RANGES = np.array([
    C_MATRIX["Dry asphalt"][0] - C_MATRIX["Snow"][0],   # c1
    C_MATRIX["Snow"][1]        - C_MATRIX["Dry asphalt"][1],  # c2
    C_MATRIX["Dry asphalt"][2] - C_MATRIX["Snow"][2],   # c3
])


# ---------------------------------------------------------------------------
# Burckhardt NLS fit
# ---------------------------------------------------------------------------
def fit_burckhardt(slip: np.ndarray, mu_data: np.ndarray):
    def model(s, c1, c2, c3):
        return c1 * (1.0 - np.exp(-c2 * np.abs(s))) - c3 * np.abs(s)

    lo = [0.01,  10.0, 0.001]
    hi = [1.60, 310.0, 0.700]
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
                best_cost = cost
                best_params = popt
        except Exception:
            continue
    return best_params, best_cost


# ---------------------------------------------------------------------------
# Nearest-neighbour classifier
# ---------------------------------------------------------------------------
def classify(params: np.ndarray):
    params_norm = params / _C_RANGES
    best_name, best_dist = None, np.inf
    for name, ref in C_MATRIX.items():
        d = np.linalg.norm(params_norm - ref / _C_RANGES)
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


# ---------------------------------------------------------------------------
# Step 1 — Build eval set
# ---------------------------------------------------------------------------
def build_eval_set(csv_path: str, eval_frac: float = 0.20, seed: int = 42):
    df = pd.read_csv(csv_path, low_memory=False)
    if "abs_active" in df.columns:
        df = df[df["abs_active"] == 1]
    if "alpha" in df.columns:
        df = df[df["alpha"] > 0.01]

    train_parts, eval_parts = [], []
    for surf, grp in df.groupby("surface"):
        tr, ev = train_test_split(grp, test_size=eval_frac,
                                  random_state=seed, shuffle=True)
        train_parts.append(tr)
        eval_parts.append(ev)

    train_df = pd.concat(train_parts, ignore_index=True)
    eval_df  = pd.concat(eval_parts,  ignore_index=True)

    # Save the surface label separately, then strip it
    eval_labels = eval_df["surface"].copy()
    unlabeled   = eval_df[["slip", "mu_noisy"]].copy()

    return train_df, unlabeled, eval_labels


# ---------------------------------------------------------------------------
# Step 2 — Run classifier on batches
# ---------------------------------------------------------------------------
def run_classifier(unlabeled: pd.DataFrame,
                   true_labels: pd.Series,
                   batch_size: int,
                   seed: int = 0):
    """
    For each surface, draw batches of `batch_size` rows from that surface only.
    The model receives ONLY slip + mu_noisy — label is hidden.
    Returns y_true, y_pred lists (one entry per batch).
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = [], []
    failed = 0
    total_batches = 0

    # Re-attach true labels temporarily so we can group by surface
    df = unlabeled.copy()
    df["_true"] = true_labels.values

    for surf in SURFACES:
        grp = df[df["_true"] == surf].reset_index(drop=True)
        n_batches = len(grp) // batch_size
        total_batches += n_batches
        idx = rng.permutation(len(grp))

        for i in range(n_batches):
            batch_idx = idx[i * batch_size:(i + 1) * batch_size]
            # Feed ONLY slip + mu_noisy — no surface label
            slip_b = grp["slip"].iloc[batch_idx].values
            mu_b   = grp["mu_noisy"].iloc[batch_idx].values

            params, _ = fit_burckhardt(slip_b, mu_b)
            if params is None:
                failed += 1
                continue

            predicted = classify(params)
            y_true.append(surf)
            y_pred.append(predicted)

    print(f"  Eval rows : {len(unlabeled):,}  |  batch_size={batch_size}")
    print(f"  Batches   : {total_batches}  "
          f"({', '.join(f'{s}: {(true_labels==s).sum()//batch_size}' for s in SURFACES)})")
    if failed:
        print(f"  [WARN] {failed} batches failed to converge and were skipped")

    return y_true, y_pred


# ---------------------------------------------------------------------------
# Step 3 — Compute metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    report = classification_report(
        y_true, y_pred,
        labels=SURFACES,
        target_names=SURFACES,
        digits=4,
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=SURFACES,
        zero_division=0,
    )

    # Macro and weighted
    _, _, f1_macro,    _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro",    zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "precision": dict(zip(SURFACES, precision)),
        "recall":    dict(zip(SURFACES, recall)),
        "f1":        dict(zip(SURFACES, f1)),
        "support":   dict(zip(SURFACES, support)),
        "f1_macro":    float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
    return report, metrics


# ---------------------------------------------------------------------------
# Step 4 — Plots
# ---------------------------------------------------------------------------
def plot_metrics(y_true, y_pred, metrics, n_eval, batch_size):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── [0] Confusion matrix ──────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=SURFACES)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Dry", "Wet", "Snow"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(
        f"Confusion Matrix\n"
        f"(n_eval={n_eval:,} rows, batch={batch_size} pts)"
    )

    # ── [1] Per-class Precision / Recall / F1 bar chart ───────────────────
    ax = axes[1]
    x = np.arange(len(SURFACES))
    w = 0.25
    bars_p = ax.bar(x - w,   [metrics["precision"][s] for s in SURFACES],
                    w, label="Precision", color="#2980B9", alpha=0.85)
    bars_r = ax.bar(x,       [metrics["recall"][s]    for s in SURFACES],
                    w, label="Recall",    color="#27AE60", alpha=0.85)
    bars_f = ax.bar(x + w,   [metrics["f1"][s]        for s in SURFACES],
                    w, label="F1",        color="#E67E22", alpha=0.85)

    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(["Dry", "Wet", "Snow"], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics\n(Precision / Recall / F1)")
    ax.legend(fontsize=10)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.5)

    # Macro / weighted F1 annotation
    ax.annotate(
        f"Macro F1    = {metrics['f1_macro']:.4f}\n"
        f"Weighted F1 = {metrics['f1_weighted']:.4f}",
        xy=(0.98, 0.05), xycoords="axes fraction",
        ha="right", fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9),
    )

    # ── [2] Support (batch count per class) ───────────────────────────────
    ax2 = axes[2]
    supports = [metrics["support"][s] for s in SURFACES]
    colors   = [COLORS[s] for s in SURFACES]
    bars = ax2.bar(["Dry", "Wet", "Snow"], supports,
                   color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, supports):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of batches")
    ax2.set_title("Eval Batches per Surface\n(support)")

    fig.suptitle(
        f"Unlabeled Surface Classification — Evaluation Report\n"
        f"Burckhardt NLS + Nearest-Neighbour Classifier",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    out = SAVE_DIR / "14_classification_metrics.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="data/raw/prepared_friction.csv")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Slip/mu points per classification batch (default 50)")
    parser.add_argument("--eval-frac",  type=float, default=0.20,
                        help="Fraction held out for evaluation (default 0.20)")
    args = parser.parse_args()

    print("=" * 60)
    print("  eval_classification.py")
    print("=" * 60)

    if not Path(args.csv).exists():
        sys.exit(f"[ERROR] {args.csv} not found. Run preprocess.py first.")

    # ── 1. Build eval set ─────────────────────────────────────────────────
    print(f"\n[1/4] Building eval set  (eval_frac={args.eval_frac}) ...")
    train_df, unlabeled, true_labels = build_eval_set(
        args.csv, eval_frac=args.eval_frac)

    print(f"  Train rows : {len(train_df):,}")
    print(f"  Eval rows  : {len(unlabeled):,}  (labels stripped)")
    for surf in SURFACES:
        n = (true_labels == surf).sum()
        print(f"    {surf:<22s}  {n:,}")

    # Save unlabeled eval CSV
    unlabeled_path = Path("data/raw/unlabeled_eval.csv")
    unlabeled.to_csv(unlabeled_path, index=False)
    print(f"\n  Unlabeled eval set saved: {unlabeled_path}")
    print(f"  Columns: {list(unlabeled.columns)}  (surface label removed)")

    # ── 2. Classify ───────────────────────────────────────────────────────
    print(f"\n[2/4] Running classifier (batch_size={args.batch_size}) ...")
    y_true, y_pred = run_classifier(
        unlabeled, true_labels, batch_size=args.batch_size)

    print(f"  Total batches classified: {len(y_true)}")
    print(f"  Correct predictions     : {sum(t==p for t,p in zip(y_true,y_pred))}")

    # ── 3. Metrics ────────────────────────────────────────────────────────
    print(f"\n[3/4] Computing metrics ...")
    report, metrics = compute_metrics(y_true, y_pred)

    print(f"\n{'='*60}")
    print(f"  CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(report)
    print(f"  Macro    F1 : {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1 : {metrics['f1_weighted']:.4f}")
    print(f"\n  Per-class summary:")
    print(f"  {'Surface':<22}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Batches':>8}")
    print(f"  {'-'*60}")
    for surf in SURFACES:
        print(f"  {surf:<22}  "
              f"{metrics['precision'][surf]:>10.4f}  "
              f"{metrics['recall'][surf]:>8.4f}  "
              f"{metrics['f1'][surf]:>8.4f}  "
              f"{metrics['support'][surf]:>8}")

    # ── 4. Plot ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving plots ...")
    out = plot_metrics(y_true, y_pred, metrics,
                       n_eval=len(unlabeled), batch_size=args.batch_size)

    # ── Smoke test ────────────────────────────────────────────────────────
    p = Path(out)
    assert p.exists() and p.stat().st_size > 20_000, \
        f"Plot missing or too small: {p}"
    print(f"  OK  {p.name}  ({p.stat().st_size // 1024} kB)")
    print("\n  SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
