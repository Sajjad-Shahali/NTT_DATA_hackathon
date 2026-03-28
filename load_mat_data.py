"""
load_mat_data.py — Load, inspect, filter, and prepare the full simulation CSV
for training friction models with predict_mu.py.

The CSV is produced by export_for_python.m and contains all signals from
every simulation run (48 runs × 2 wheels × ~15 000 timesteps each).

Available columns
-----------------
Identity:   test_no, case_no, wheel, mode, scenario, dist_name, road_1, road_2, surface
Time:       time
Measured:   slip, mu_noisy, s_probe, g_esc, a_esc, s_hat, alpha, dither
Vehicle:    v_cog, abs_active
Ground truth: c1_true, c2_true, c3_true, mu_true, s_opt_true, mu_peak_true
Run metrics:  d_stop, t_stop, T_front, T_rear

Usage
-----
  # Show what is in the CSV (no file written)
  python load_mat_data.py --inspect

  # Export minimal CSV for predict_mu.py (default — Framework runs only)
  python load_mat_data.py

  # Include NoABS runs as well
  python load_mat_data.py --mode all

  # Use only timesteps where ABS was active (cleanest slip tracking)
  python load_mat_data.py --abs-only

  # Thin the data (keep every Nth row — avoids autocorrelation in training)
  python load_mat_data.py --stride 10

  # Keep specific scenarios only
  python load_mat_data.py --scenarios "Snow High" "Wet High"

  # Then train
  python predict_mu.py --data data/raw/prepared_friction.csv --n-train 500 --save plots/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT  = "data/raw/real_friction_data.csv"
DEFAULT_OUTPUT = "data/raw/prepared_friction.csv"

REQUIRED_COLS = {"slip", "mu_noisy", "surface"}

# Columns that must be numeric (can be NaN but not strings)
NUMERIC_COLS = [
    "slip", "mu_noisy", "s_probe", "g_esc", "a_esc", "s_hat",
    "alpha", "dither", "v_cog", "abs_active",
    "c1_true", "c2_true", "c3_true", "mu_true", "s_opt_true", "mu_peak_true",
    "d_stop", "t_stop", "T_front", "T_rear", "time",
]

SURFACE_LABELS = {"Dry asphalt", "Wet asphalt", "Snow"}


# ===========================================================================
# Load
# ===========================================================================

def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"\nERROR: File not found: {p.resolve()}")
        print("\nGenerate it with MATLAB:")
        print("  1. run('run_validation_updated.m')")
        print("  2. run('export_for_python.m')")
        print(f"  3. Copy friction_data_full.csv → {p.resolve()}")
        sys.exit(1)

    if p.stat().st_size == 0:
        print(f"\nERROR: {p} is empty (0 bytes).")
        print("The MATLAB simulation has not been run yet.")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded: {p.resolve()}")
    print(f"  Shape:  {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ===========================================================================
# Inspect
# ===========================================================================

COLUMN_DESCRIPTIONS = {
    "test_no":       "simulation run index (1–48)",
    "case_no":       "scenario+distribution case index (1–24)",
    "wheel":         "FL or RL",
    "mode":          "Framework (ABS active) or NoABS",
    "scenario":      "e.g. Snow High, Dry->Snow",
    "dist_name":     "brake distribution: 60F/40R etc.",
    "road_1":        "surface index before t_mu  (1=Dry, 2=Wet, 3=Snow)",
    "road_2":        "surface index after t_mu",
    "surface":       "active surface label at this timestep",
    "time":          "simulation time [s]",
    "slip":          "|s_measured|, positive, range 0–1  ← PRIMARY INPUT",
    "mu_noisy":      "measured friction coefficient (noisy)  ← PRIMARY TARGET",
    "s_probe":       "ESC-perturbed slip setpoint sent to wheel controller",
    "g_esc":         "ESC gradient estimate from LPF demodulator  ← identifier input",
    "a_esc":         "ESC dither amplitude (design parameter)     ← identifier input",
    "s_hat":         "ESC reference slip (optimizer output)",
    "alpha":         "identifier confidence metric  0=uncertain, 1=confident",
    "dither":        "ESC perturbation signal (raw)",
    "v_cog":         "vehicle longitudinal speed [m/s]",
    "abs_active":    "1 when ABS was controlling this wheel, 0 otherwise",
    "c1_true":       "Burckhardt c1 for the active surface  ← ground truth",
    "c2_true":       "Burckhardt c2 for the active surface  ← ground truth",
    "c3_true":       "Burckhardt c3 for the active surface  ← ground truth",
    "mu_true":       "noiseless mu(slip) from ground-truth params  ← ground truth",
    "s_opt_true":    "optimal slip = ln(c1*c2/c3)/c2  ← ground truth",
    "mu_peak_true":  "peak friction = mu(s_opt_true)  ← ground truth",
    "d_stop":        "stopping distance for this run [m]",
    "t_stop":        "stopping time for this run [s]",
    "T_front":       "front brake torque per wheel [Nm]",
    "T_rear":        "rear brake torque per wheel [Nm]",
}

USE_CASE_GUIDE = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  USE CASE GUIDE — which columns to use for each task                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Task                      Required columns                                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Offline μ(s) curve fit    slip, mu_noisy, surface                          ║
║  (Burckhardt NLS / GP / NN)                                                 ║
║                                                                             ║
║  Identifier validation     slip, mu_noisy, s_probe, g_esc, a_esc, surface  ║
║  (test ESCTwoPointID)      + c1_true, c2_true, c3_true for error metrics   ║
║                                                                             ║
║  Confidence-weighted fit   slip, mu_noisy, surface, alpha                  ║
║  (weight train samples)    (high-alpha rows = identifier was confident)     ║
║                                                                             ║
║  ABS-only analysis         filter: abs_active == 1, then use above         ║
║  (cleanest slip data)                                                       ║
║                                                                             ║
║  Stopping distance model   d_stop, scenario, mode, dist_name, surface      ║
║  (Framework vs NoABS)      (one row per run — use groupby test_no first)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def inspect(df: pd.DataFrame):
    print("\n" + "="*72)
    print("  COLUMN INVENTORY")
    print("="*72)
    for col in df.columns:
        desc = COLUMN_DESCRIPTIONS.get(col, "")
        if col in df.select_dtypes(include="number").columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                info = f"  [{vals.min():.4g} … {vals.max():.4g}]  nan={df[col].isna().sum()}"
            else:
                info = "  (all NaN)"
        else:
            uniq = df[col].dropna().unique()
            info = "  values: " + str(sorted(uniq.tolist())[:8])
        print(f"  {col:<20s}  {info}")
        if desc:
            print(f"  {'':20s}  → {desc}")

    print("\n" + "="*72)
    print("  DATASET BREAKDOWN")
    print("="*72)

    if "surface" in df.columns:
        print("\n  By surface:")
        for surf, grp in df.groupby("surface"):
            print(f"    {surf:<22s}  n={len(grp):8,}  "
                  f"slip=[{grp['slip'].min():.3f},{grp['slip'].max():.3f}]  "
                  f"mu=[{grp['mu_noisy'].min():.3f},{grp['mu_noisy'].max():.3f}]")

    if "mode" in df.columns:
        print("\n  By mode:")
        for mode, grp in df.groupby("mode"):
            print(f"    {mode:<15s}  n={len(grp):8,}")

    if "scenario" in df.columns:
        print("\n  By scenario:")
        for scen, grp in df.groupby("scenario"):
            surfs = grp["surface"].unique().tolist() if "surface" in grp else []
            print(f"    {scen:<16s}  n={len(grp):8,}  surfaces={surfs}")

    if "wheel" in df.columns:
        print("\n  By wheel:")
        for wh, grp in df.groupby("wheel"):
            print(f"    {wh:<6s}  n={len(grp):8,}")

    if "abs_active" in df.columns:
        n_abs = (df["abs_active"] == 1).sum()
        print(f"\n  ABS active rows : {n_abs:,}  ({100*n_abs/len(df):.1f}%)")

    if "alpha" in df.columns:
        a = df["alpha"].dropna()
        print(f"  Alpha range     : [{a.min():.4f} … {a.max():.4f}]  "
              f"median={a.median():.4f}")

    print(USE_CASE_GUIDE)


# ===========================================================================
# Filter & prepare
# ===========================================================================

def prepare(df: pd.DataFrame,
            mode_filter: str = "Framework",
            abs_only: bool = False,
            stride: int = 1,
            scenarios: list = None,
            surfaces: list = None,
            min_alpha: float = None,
            min_v_cog: float = None) -> pd.DataFrame:
    """
    Apply filters and return a clean DataFrame ready for predict_mu.py.

    Parameters
    ----------
    mode_filter : 'Framework', 'NoABS', or 'all'
    abs_only    : keep only rows where abs_active == 1
    stride      : keep every Nth row (reduces autocorrelation)
    scenarios   : list of scenario names to keep, e.g. ['Snow High', 'Wet High']
    surfaces    : list of surface labels to keep
    min_alpha   : discard rows with alpha < threshold (e.g. 0.3)
    min_v_cog   : discard rows with v_cog below this speed [m/s]
    """
    n0 = len(df)
    log = []

    # Mode filter
    if mode_filter != "all" and "mode" in df.columns:
        df = df[df["mode"] == mode_filter]
        log.append(f"mode={mode_filter}: {len(df):,} rows")

    # Scenario filter
    if scenarios and "scenario" in df.columns:
        df = df[df["scenario"].isin(scenarios)]
        log.append(f"scenarios={scenarios}: {len(df):,} rows")

    # Surface filter
    if surfaces and "surface" in df.columns:
        df = df[df["surface"].isin(surfaces)]
        log.append(f"surfaces={surfaces}: {len(df):,} rows")

    # ABS active only
    if abs_only and "abs_active" in df.columns:
        df = df[df["abs_active"] == 1]
        log.append(f"abs_active==1: {len(df):,} rows")

    # Min confidence
    if min_alpha is not None and "alpha" in df.columns:
        df = df[df["alpha"].fillna(0) >= min_alpha]
        log.append(f"alpha>={min_alpha}: {len(df):,} rows")

    # Min vehicle speed (exclude near-stop lockup)
    if min_v_cog is not None and "v_cog" in df.columns:
        df = df[df["v_cog"].fillna(0) >= min_v_cog]
        log.append(f"v_cog>={min_v_cog}: {len(df):,} rows")

    # Slip and mu sanity
    df = df[(df["slip"] >= 0.005) & (df["slip"] <= 0.50) &
            (df["mu_noisy"] >= 0.0) & (df["mu_noisy"] <= 2.0) &
            df["slip"].notna() & df["mu_noisy"].notna()]
    log.append(f"after quality filter: {len(df):,} rows")

    # Stride (thin)
    if stride > 1:
        df = df.iloc[::stride].copy()
        log.append(f"stride={stride}: {len(df):,} rows")

    print(f"\n  Filter chain ({n0:,} → {len(df):,} rows):")
    for l in log:
        print(f"    {l}")

    return df.reset_index(drop=True)


# ===========================================================================
# Summary
# ===========================================================================

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  Prepared dataset ready for training")
    print(f"{'='*60}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {list(df.columns)}")
    print()
    if "surface" in df.columns:
        for surf, grp in df.groupby("surface"):
            print(f"  {surf:<22s}  n={len(grp):7,}  "
                  f"slip=[{grp['slip'].min():.3f},{grp['slip'].max():.3f}]  "
                  f"mu=[{grp['mu_noisy'].min():.3f},{grp['mu_noisy'].max():.3f}]")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inspect and prepare simulation CSV for friction model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv",        default=DEFAULT_INPUT,
                        help=f"Input CSV path (default: {DEFAULT_INPUT})")
    parser.add_argument("--out",        default=DEFAULT_OUTPUT,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--inspect",    action="store_true",
                        help="Print column inventory and dataset breakdown only")
    parser.add_argument("--mode",       default="Framework",
                        choices=["Framework", "NoABS", "all"],
                        help="Simulation mode to keep (default: Framework)")
    parser.add_argument("--abs-only",   action="store_true",
                        help="Keep only timesteps where ABS was active")
    parser.add_argument("--stride",     type=int, default=10,
                        help="Keep every Nth row (reduces autocorrelation, default: 10)")
    parser.add_argument("--scenarios",  nargs="+", default=None,
                        metavar="SCEN",
                        help="Keep only these scenarios, e.g.: 'Snow High' 'Wet High'")
    parser.add_argument("--surfaces",   nargs="+", default=None,
                        metavar="SURF",
                        help="Keep only these surfaces, e.g.: 'Dry asphalt' 'Snow'")
    parser.add_argument("--min-alpha",  type=float, default=None,
                        help="Discard rows with identifier confidence below this value")
    parser.add_argument("--min-speed",  type=float, default=2.0,
                        help="Discard rows with v_cog below this speed m/s (default: 2.0)")
    args = parser.parse_args()

    print(f"\nLoading: {args.csv}")
    df = load_csv(args.csv)

    if args.inspect:
        inspect(df)
        return

    df_out = prepare(
        df,
        mode_filter=args.mode,
        abs_only=args.abs_only,
        stride=args.stride,
        scenarios=args.scenarios,
        surfaces=args.surfaces,
        min_alpha=args.min_alpha,
        min_v_cog=args.min_speed,
    )

    if df_out.empty:
        print("\nERROR: No rows survived the filters. Relax your filter settings.")
        sys.exit(1)

    print_summary(df_out)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"\n  Saved: {out.resolve()}")

    print(f"""
Next steps:
  # Train all 3 models on this data
  python predict_mu.py --data {args.out}

  # Train with more samples and save plots
  python predict_mu.py --data {args.out} --n-train 500 --save plots/

  # Train on one surface only
  python predict_mu.py --data {args.out} --surface dry
""")


if __name__ == "__main__":
    main()
