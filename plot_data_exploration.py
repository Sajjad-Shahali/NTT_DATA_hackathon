"""
plot_data_exploration.py — EDA plots for the real simulation dataset.

Generates 5 plots saved to reports/plots/:
  06_data_slip_distribution.png   -- slip histogram by surface + mode
  07_data_mu_vs_slip.png          -- scatter mu vs slip, colour = surface
  08_data_alpha_distribution.png  -- identifier confidence distribution
  09_data_scenario_coverage.png   -- sample counts per scenario & surface
  10_data_real_vs_model.png       -- real data overlaid on Burckhardt ground truth

Usage:  python plot_data_exploration.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.burckhardt import mu as bmu, s_opt, mu_peak, SURFACES

SAVE_DIR = Path("reports/plots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.dpi": 180, "savefig.bbox": "tight",
})

COLORS = {
    "Dry asphalt": "#E74C3C",
    "Wet asphalt": "#3498DB",
    "Snow":        "#95A5A6",
}


def load():
    raw  = pd.read_csv("data/raw/friction_data_full.csv", low_memory=False)
    prep = pd.read_csv("data/raw/prepared_friction.csv",  low_memory=False)
    print(f"  Raw  : {len(raw):,} rows")
    print(f"  Prepped: {len(prep):,} rows")
    return raw, prep


# ===========================================================================
# PLOT 06 — Slip distributions: raw vs prepared, split by surface
# ===========================================================================
def plot_06_slip_distribution(raw, prep):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    surfaces = ["Dry asphalt", "Wet asphalt", "Snow"]

    for ax, surf in zip(axes, surfaces):
        color = COLORS[surf]
        r = raw[raw["surface"] == surf]["slip"]
        p = prep[prep["surface"] == surf]["slip"]

        bins = np.linspace(0, 0.50, 50)
        ax.hist(r, bins=bins, alpha=0.4, color=color,
                label=f"Raw (n={len(r):,})", density=True)
        ax.hist(p, bins=bins, alpha=0.85, color=color, histtype="step",
                linewidth=2, label=f"Prepared (n={len(p):,})", density=True)

        # Mark s_opt
        key = {"Dry asphalt": "dry", "Wet asphalt": "wet", "Snow": "snow"}[surf]
        p_surf = SURFACES[key]
        sop = s_opt(p_surf.c1, p_surf.c2, p_surf.c3)
        ax.axvline(sop, color="black", lw=1.5, ls="--",
                   label=f"s_opt_true={sop:.3f}")

        ax.set_title(surf, color=color)
        ax.set_xlabel("Slip  s")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    fig.suptitle(
        "Slip Distribution: Raw vs Preprocessed\n"
        "(abs_active filter removes low-slip free-rolling rows)",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    out = SAVE_DIR / "06_data_slip_distribution.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================================
# PLOT 07 — mu vs slip scatter coloured by surface
# ===========================================================================
def plot_07_mu_vs_slip(prep):
    fig, ax = plt.subplots(figsize=(10, 6))
    s_dense = np.linspace(0.001, 0.50, 500)

    for surf, grp in prep.groupby("surface"):
        color = COLORS.get(surf, "#333")
        # Thin scatter to avoid overplotting
        sample = grp.sample(min(len(grp), 2000), random_state=0)
        ax.scatter(sample["slip"], sample["mu_noisy"],
                   s=6, alpha=0.25, color=color, label=f"{surf} (data)")

    # Overlay ground-truth curves
    for key, label in [("dry","Dry asphalt"),("wet","Wet asphalt"),("snow","Snow")]:
        p = SURFACES[key]
        mu_gt = bmu(s_dense, p.c1, p.c2, p.c3)
        ax.plot(s_dense, mu_gt, color=COLORS[label], lw=2.5,
                ls="--", label=f"{label} (Burckhardt true)")
        sop = s_opt(p.c1, p.c2, p.c3)
        mup = mu_peak(p.c1, p.c2, p.c3)
        ax.plot(sop, mup, "o", ms=9, color=COLORS[label], zorder=6)

    ax.set_xlabel("Slip  s")
    ax.set_ylabel("Friction coefficient  mu")
    ax.set_title("Real Simulation Data: mu vs slip\n(scatter = measured, dashed = Burckhardt ground truth)")
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(0, 0.52)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    out = SAVE_DIR / "07_data_mu_vs_slip.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================================
# PLOT 08 — Alpha distribution (identifier confidence)
# ===========================================================================
def plot_08_alpha_distribution(raw):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overall alpha histogram coloured by abs_active
    ax = axes[0]
    for val, label, color in [(0, "abs_active=0\n(ABS off)", "#BDC3C7"),
                               (1, "abs_active=1\n(ABS on)",  "#27AE60")]:
        sub = raw[raw["abs_active"] == val]["alpha"]
        ax.hist(sub, bins=60, alpha=0.7, color=color,
                label=f"{label}  n={len(sub):,}", density=True)
    ax.set_xlabel("alpha (identifier confidence)")
    ax.set_ylabel("Density")
    ax.set_title("Alpha Distribution by ABS State")
    ax.legend(fontsize=9)

    # Right: alpha over time for one representative run
    ax2 = axes[1]
    fw_runs = raw[(raw["mode"] == "Framework") & (raw["scenario"] == "Snow High") &
                  (raw["wheel"] == "FL")]
    if not fw_runs.empty:
        run_id = fw_runs["test_no"].iloc[0]
        run = fw_runs[fw_runs["test_no"] == run_id].sort_values("time")
        ax2.plot(run["time"], run["alpha"], color="#27AE60", lw=1.5, label="alpha")
        ax2.fill_between(run["time"], 0, run["alpha"], alpha=0.15, color="#27AE60")
        ax2.axhline(0.01, color="red", lw=1, ls="--", label="filter threshold (0.01)")
        ax2.axhline(0.50, color="orange", lw=1, ls=":", label="high confidence (0.50)")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("alpha")
        ax2.set_title(f"Alpha Over Time — Snow High, FL wheel (test {run_id})")
        ax2.legend(fontsize=9)
        ax2.set_ylim(-0.05, 1.1)

    fig.suptitle("Identifier Confidence (alpha) Analysis", fontsize=13, y=1.01)
    plt.tight_layout()
    out = SAVE_DIR / "08_data_alpha_distribution.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================================
# PLOT 09 — Sample counts per scenario and surface
# ===========================================================================
def plot_09_scenario_coverage(raw, prep):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, df, title in [
        (axes[0], raw[raw["abs_active"] == 1], "Raw (abs_active==1)"),
        (axes[1], prep, "Prepared (after all filters)"),
    ]:
        counts = (df.groupby(["scenario", "surface"])
                    .size()
                    .reset_index(name="n")
                    .sort_values(["scenario", "surface"]))

        scens = counts["scenario"].unique()
        surfs = ["Dry asphalt", "Wet asphalt", "Snow"]
        x = np.arange(len(scens))
        width = 0.28

        for i, surf in enumerate(surfs):
            sub = counts[counts["surface"] == surf].set_index("scenario")
            vals = [sub.loc[s, "n"] if s in sub.index else 0 for s in scens]
            bars = ax.bar(x + i * width, vals, width,
                          label=surf, color=COLORS[surf], alpha=0.85,
                          edgecolor="white")
            for bar, v in zip(bars, vals):
                if v > 100:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                            f"{v:,}", ha="center", va="bottom", fontsize=6.5,
                            rotation=45)

        ax.set_xticks(x + width)
        ax.set_xticklabels(scens, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Sample count")
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle("Sample Coverage by Scenario and Surface", fontsize=13, y=1.01)
    plt.tight_layout()
    out = SAVE_DIR / "09_data_scenario_coverage.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================================
# PLOT 10 — Real data vs Burckhardt ground truth (per surface, full curve)
# ===========================================================================
def plot_10_real_vs_model(prep):
    s_dense = np.linspace(0.001, 0.50, 500)
    surfaces = [("dry", "Dry asphalt"), ("wet", "Wet asphalt"), ("snow", "Snow")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (key, surf_label) in zip(axes, surfaces):
        color = COLORS[surf_label]
        grp = prep[prep["surface"] == surf_label]
        p   = SURFACES[key]

        # Scatter — actual measurements
        sample = grp.sample(min(len(grp), 3000), random_state=1)
        ax.scatter(sample["slip"], sample["mu_noisy"],
                   s=8, alpha=0.35, color=color,
                   label=f"Measured  (n={len(grp):,})")

        # Ground-truth Burckhardt curve
        mu_gt = bmu(s_dense, p.c1, p.c2, p.c3)
        ax.plot(s_dense, mu_gt, color="black", lw=2.5, ls="--",
                label=f"Burckhardt true\nc1={p.c1}, c2={p.c2}, c3={p.c3}")

        # Residual annotation
        if "mu_true" in grp.columns:
            resid = (grp["mu_noisy"] - grp["mu_true"]).abs()
            ax.annotate(
                f"Noise floor\nmax|mu_meas - mu_true| = {resid.max():.5f}",
                xy=(0.35, 0.10), xycoords="axes fraction",
                fontsize=8, color="grey",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )

        sop = s_opt(p.c1, p.c2, p.c3)
        mup = mu_peak(p.c1, p.c2, p.c3)
        ax.axvline(sop, color="black", lw=1, ls=":", alpha=0.5)
        ax.plot(sop, mup, "k^", ms=8, zorder=6,
                label=f"s_opt={sop:.3f}, mu_peak={mup:.3f}")

        ax.set_title(surf_label, color=color)
        ax.set_xlabel("Slip  s")
        ax.set_ylabel("mu(s)")
        ax.set_xlim(0, 0.52)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Real Simulation Data vs Burckhardt Ground Truth\n"
        "(near-zero noise confirms clean simulation data)",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    out = SAVE_DIR / "10_data_real_vs_model.png"
    fig.savefig(out); plt.close(fig)
    print(f"  Saved: {out}")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("="*55)
    print("  plot_data_exploration.py")
    print(f"  Output: {SAVE_DIR.resolve()}")
    print("="*55)

    raw, prep = load()

    print("\n[06] Slip distributions ...")
    plot_06_slip_distribution(raw, prep)

    print("[07] mu vs slip scatter ...")
    plot_07_mu_vs_slip(prep)

    print("[08] Alpha (confidence) distribution ...")
    plot_08_alpha_distribution(raw)

    print("[09] Scenario coverage ...")
    plot_09_scenario_coverage(raw, prep)

    print("[10] Real data vs Burckhardt ground truth ...")
    plot_10_real_vs_model(prep)

    saved = sorted(SAVE_DIR.glob("0[6-9]_*.png")) + sorted(SAVE_DIR.glob("10_*.png"))
    print(f"\nAll {len(saved)} EDA plots saved:")
    for p in saved:
        print(f"  OK  {p.name}  ({p.stat().st_size//1024} kB)")


if __name__ == "__main__":
    main()
