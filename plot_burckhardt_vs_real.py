"""
plot_burckhardt_vs_real.py

Plots the Burckhardt model curves (from the given c_matrix) against
the real simulation measurements for Dry, Wet, and Snow surfaces.

c_matrix (from MATLAB):
    Dry      Wet      Snow
c1  1.2801   0.857    0.1946
c2  23.99    33.822   94.129
c3  0.52     0.347    0.0646

Usage:  python plot_burckhardt_vs_real.py
        python plot_burckhardt_vs_real.py --csv data/raw/friction_data_full.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.burckhardt import mu as bmu, s_opt, mu_peak

# ---------------------------------------------------------------------------
# Exact c_matrix from MATLAB (user-provided)
# ---------------------------------------------------------------------------
C_MATRIX = {
    "Dry asphalt": dict(c1=1.2801,  c2=23.990,  c3=0.5200),
    "Wet asphalt": dict(c1=0.8570,  c2=33.822,  c3=0.3470),
    "Snow":        dict(c1=0.1946,  c2=94.129,  c3=0.0646),
}

COLORS = {
    "Dry asphalt": "#E74C3C",
    "Wet asphalt": "#3498DB",
    "Snow":        "#95A5A6",
}

SAVE_DIR = Path("reports/plots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
})


def load_data(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        print(f"  [WARN] {csv_path} not found — model curves only (no real data overlay)")
        return None
    df = pd.read_csv(p, low_memory=False)
    print(f"  Loaded {len(df):,} rows from {p.name}")

    # Normalise column names
    if "slip" not in df.columns and "slip_abs" in df.columns:
        df["slip"] = df["slip_abs"]
    if "mu_noisy" not in df.columns and "mu_measured" in df.columns:
        df["mu_noisy"] = df["mu_measured"]

    # Keep ABS-active, high-confidence rows if columns are present
    before = len(df)
    if "abs_active" in df.columns:
        df = df[df["abs_active"] == 1]
    if "alpha" in df.columns:
        df = df[df["alpha"] > 0.01]
    if len(df) < before:
        print(f"  After quality filters: {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# PLOT A — 3 subplots, one per surface
# ---------------------------------------------------------------------------
def plot_per_surface(df, out_name="11_burckhardt_vs_real.png"):
    s_dense = np.linspace(0.001, 0.50, 600)
    surfaces = list(C_MATRIX.keys())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, surf in zip(axes, surfaces):
        p = C_MATRIX[surf]
        color = COLORS[surf]

        # ── Real data scatter ──────────────────────────────────────────────
        n_real = 0
        if df is not None and "surface" in df.columns:
            grp = df[df["surface"] == surf]
            if not grp.empty:
                sample = grp.sample(min(len(grp), 3000), random_state=0)
                ax.scatter(
                    sample["slip"], sample["mu_noisy"],
                    s=6, alpha=0.30, color=color,
                    label=f"Measured  (n={len(grp):,})",
                    zorder=3,
                )
                n_real = len(grp)

        # ── Burckhardt model curve ─────────────────────────────────────────
        mu_model = bmu(s_dense, p["c1"], p["c2"], p["c3"])
        ax.plot(
            s_dense, mu_model,
            color="black", lw=2.5, ls="--", zorder=5,
            label=(
                f"Burckhardt model\n"
                f"c1={p['c1']}, c2={p['c2']}, c3={p['c3']}"
            ),
        )

        # ── Peak & s_opt annotation ────────────────────────────────────────
        sop = s_opt(p["c1"], p["c2"], p["c3"])
        mup = mu_peak(p["c1"], p["c2"], p["c3"])
        ax.plot(sop, mup, "k^", ms=9, zorder=6,
                label=f"s_opt={sop:.3f}  mu_peak={mup:.3f}")
        ax.axvline(sop, color="black", lw=0.8, ls=":", alpha=0.4)

        # ── Residual annotation (if mu_true available) ─────────────────────
        if df is not None and "mu_true" in df.columns and n_real > 0:
            grp = df[df["surface"] == surf]
            resid = (grp["mu_noisy"] - grp["mu_true"]).abs()
            ax.annotate(
                f"Noise: max|meas-true|={resid.max():.5f}",
                xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=8, color="grey",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
            )

        ax.set_title(surf, color=color)
        ax.set_xlabel("Slip  s")
        ax.set_ylabel("Friction coefficient  mu(s)")
        ax.set_xlim(0, 0.52)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")

    data_note = "(scatter = simulation measurements)" if df is not None else "(model only — no data loaded)"
    fig.suptitle(
        f"Burckhardt c_matrix vs Real Simulation Data\n{data_note}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = SAVE_DIR / out_name
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# PLOT B — All 3 surfaces on one axes (easier for side-by-side comparison)
# ---------------------------------------------------------------------------
def plot_combined(df, out_name="12_burckhardt_vs_real_combined.png"):
    s_dense = np.linspace(0.001, 0.50, 600)

    fig, ax = plt.subplots(figsize=(10, 6))

    for surf, p in C_MATRIX.items():
        color = COLORS[surf]

        # Real data
        if df is not None and "surface" in df.columns:
            grp = df[df["surface"] == surf]
            if not grp.empty:
                sample = grp.sample(min(len(grp), 2000), random_state=1)
                ax.scatter(
                    sample["slip"], sample["mu_noisy"],
                    s=5, alpha=0.20, color=color, zorder=3,
                )

        # Model curve
        mu_model = bmu(s_dense, p["c1"], p["c2"], p["c3"])
        sop = s_opt(p["c1"], p["c2"], p["c3"])
        mup = mu_peak(p["c1"], p["c2"], p["c3"])

        ax.plot(s_dense, mu_model, color=color, lw=2.5, ls="--", zorder=5,
                label=f"{surf}  (mu_peak={mup:.3f}, s_opt={sop:.3f})")
        ax.plot(sop, mup, "o", ms=8, color=color, zorder=6)

    ax.set_xlabel("Slip  s", fontsize=12)
    ax.set_ylabel("Friction coefficient  mu(s)", fontsize=12)
    ax.set_title(
        "Burckhardt c_matrix — All Surfaces\n"
        "(dashed = model, dots = measured data)",
        fontsize=13,
    )
    ax.set_xlim(0, 0.52)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    out = SAVE_DIR / out_name
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke_test(paths):
    print("\n--- Smoke test ---")
    passed = True
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  FAIL  {p.name} — file not found")
            passed = False
        elif p.stat().st_size < 20_000:
            print(f"  FAIL  {p.name} — too small ({p.stat().st_size} bytes)")
            passed = False
        else:
            print(f"  OK    {p.name}  ({p.stat().st_size // 1024} kB)")
    if passed:
        print("  SMOKE TEST PASSED")
    else:
        print("  SMOKE TEST FAILED")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/raw/prepared_friction.csv",
        help="Path to real data CSV (default: data/raw/prepared_friction.csv). "
             "Falls back to friction_data_full.csv if prepared not found.",
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  plot_burckhardt_vs_real.py")
    print(f"  Output: {SAVE_DIR.resolve()}")
    print("=" * 55)

    # Try prepared first, then raw
    csv = args.csv
    if not Path(csv).exists():
        fallback = "data/raw/friction_data_full.csv"
        if Path(fallback).exists():
            print(f"  {csv} not found — using {fallback}")
            csv = fallback

    df = load_data(csv)

    print("\n[A] Per-surface subplots ...")
    p1 = plot_per_surface(df)

    print("[B] Combined all-surface plot ...")
    p2 = plot_combined(df)

    smoke_test([p1, p2])


if __name__ == "__main__":
    main()
