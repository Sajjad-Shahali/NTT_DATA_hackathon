"""
preprocess.py — Preprocessing pipeline for friction_data_full.csv

Prints a column-by-column keep/drop decision, applies all filters,
balances surfaces, and saves a training-ready CSV.

Usage
-----
  python preprocess.py                          # default settings
  python preprocess.py --no-balance            # keep natural imbalance
  python preprocess.py --alpha-min 0.5         # stricter confidence filter
  python preprocess.py --stride 5              # less thinning
  python preprocess.py --out data/raw/train.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column decisions — shown to user
# ---------------------------------------------------------------------------
COLUMN_GUIDE = """
+--------------------------------------------------------------------------+
|  COLUMN DECISIONS                                                        |
+--------------------------------------------------------------------------+
|  KEEP -- model features                                                  |
|    slip          primary input  (|s_measured|, ABS-excited slip)         |
|    mu_noisy      primary target (friction coefficient)                   |
|    surface       surface label  (per-surface model training)             |
|                                                                          |
|  KEEP -- ground truth (validation only, not training features)           |
|    c1_true / c2_true / c3_true   true Burckhardt params                 |
|    mu_true       noiseless friction  (~= mu_noisy, diff < 0.0003)       |
|    s_opt_true    true optimal slip                                       |
|    mu_peak_true  true peak friction                                      |
|                                                                          |
|  DROP -- all NaN                                                         |
|    s_hat         100% NaN -- not logged in this simulation               |
|                                                                          |
|  DROP -- redundant / not friction features                               |
|    road_1 / road_2   same info as surface label                         |
|    case_no           redundant with test_no + scenario + dist_name      |
|    dither            raw ESC perturbation waveform, not a feature        |
|    d_stop / t_stop   run-level metrics, same value for whole run        |
|    T_front / T_rear  brake torque = scenario setup, not friction signal  |
|                                                                          |
|  DROP -- used for FILTERING only (not model features)                    |
|    abs_active    filter: keep == 1 only (ABS excited slip)              |
|    alpha         filter: keep > 0.01 (skip identifier warmup)           |
|    time          not a friction feature (Burckhardt is time-invariant)  |
|    test_no / wheel / mode / scenario / dist_name   metadata only        |
|                                                                          |
|  DROP -- mostly zero (6.4% non-trivial rows)                             |
|    g_esc         ESC gradient -- near-zero when ESC not perturbing      |
|    a_esc / s_probe   ESC design params -- constant or near-constant     |
+--------------------------------------------------------------------------+
"""

TRAIN_COLS  = ["slip", "mu_noisy", "surface"]
VALIDT_COLS = ["c1_true", "c2_true", "c3_true",
               "mu_true", "s_opt_true", "mu_peak_true"]
META_COLS   = ["test_no", "scenario", "dist_name", "wheel", "mode"]


def run(input_csv: str,
        output_csv: str,
        alpha_min: float = 0.01,
        stride: int = 10,
        balance: bool = True,
        snow_cap: int = 12000,
        other_cap: int = 6000):

    print(COLUMN_GUIDE)

    # ── 1. Load ────────────────────────────────────────────────────────────
    print(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"  Raw shape: {df.shape[0]:,} rows × {df.shape[1]} cols\n")

    # ── 2. Drop useless columns ────────────────────────────────────────────
    drop_cols = ["s_hat", "road_1", "road_2", "case_no",
                 "dither", "d_stop", "t_stop", "T_front", "T_rear",
                 "g_esc", "a_esc", "s_probe"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    print(f"  After dropping {len(drop_cols)} columns: {df.shape[1]} cols remaining")

    # ── 3. Filter: ABS-active rows only ───────────────────────────────────
    before = len(df)
    df = df[df["abs_active"] == 1].copy()
    print(f"  After abs_active==1 filter: {len(df):,} rows  "
          f"(dropped {before-len(df):,} free-rolling rows)")

    # ── 4. Filter: identifier confidence ──────────────────────────────────
    before = len(df)
    df = df[df["alpha"] > alpha_min].copy()
    print(f"  After alpha>{alpha_min} filter: {len(df):,} rows  "
          f"(dropped {before-len(df):,} warmup rows)")

    # ── 5. Thin autocorrelated time-series (stride within each run+wheel) ─
    before = len(df)
    df = (df.groupby(["test_no", "wheel"], group_keys=False)
            .apply(lambda g: g.iloc[::stride])
            .reset_index(drop=True))
    print(f"  After stride={stride} within each run×wheel: {len(df):,} rows  "
          f"(dropped {before-len(df):,} autocorrelated rows)")

    # ── 6. Show surface distribution before balancing ──────────────────────
    print(f"\n  Surface distribution before balancing:")
    for surf, n in df["surface"].value_counts().items():
        print(f"    {surf:<22s}  {n:6,}  ({100*n/len(df):.1f}%)")

    # ── 7. Balance surfaces ────────────────────────────────────────────────
    if balance:
        rng = np.random.default_rng(42)
        parts = []
        caps = {"Snow": snow_cap, "Dry asphalt": other_cap, "Wet asphalt": other_cap}
        for surf, grp in df.groupby("surface"):
            cap = caps.get(surf, other_cap)
            if len(grp) > cap:
                # Stratified sample: keep diversity across scenarios
                grp = grp.groupby("scenario", group_keys=False).apply(
                    lambda g: g.sample(
                        min(len(g), max(1, int(cap * len(g) / len(
                            df[df["surface"] == surf])))),
                        random_state=42
                    )
                ).reset_index(drop=True)
                # If still over cap, random sample
                if len(grp) > cap:
                    grp = grp.sample(cap, random_state=42)
            parts.append(grp)
        df = pd.concat(parts, ignore_index=True)
        print(f"\n  Surface distribution after balancing (Snow cap={snow_cap:,}, others cap={other_cap:,}):")
        for surf, n in df["surface"].value_counts().items():
            print(f"    {surf:<22s}  {n:6,}  ({100*n/len(df):.1f}%)")

    # ── 8. Final sanity checks ─────────────────────────────────────────────
    print(f"\n  Slip range:  [{df['slip'].min():.4f}, {df['slip'].max():.4f}]  "
          f"mean={df['slip'].mean():.4f}")
    print(f"  mu range:   [{df['mu_noisy'].min():.4f}, {df['mu_noisy'].max():.4f}]  "
          f"mean={df['mu_noisy'].mean():.4f}")
    print(f"  NaN check:  {df[TRAIN_COLS].isnull().sum().to_dict()}")

    # ── 9. Keep only relevant columns in output ────────────────────────────
    out_cols = TRAIN_COLS + VALIDT_COLS + META_COLS
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols].reset_index(drop=True)

    # ── 10. Save ───────────────────────────────────────────────────────────
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  PREPROCESSED DATASET")
    print(f"{'='*60}")
    print(f"  Rows    : {len(df_out):,}")
    print(f"  Columns : {list(df_out.columns)}")
    print(f"  Saved   : {output_csv}")
    print(f"{'='*60}")

    return df_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="data/raw/friction_data_full.csv")
    parser.add_argument("--out",        default="data/raw/prepared_friction.csv")
    parser.add_argument("--alpha-min",  type=float, default=0.01)
    parser.add_argument("--stride",     type=int,   default=10)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--snow-cap",   type=int,   default=12000)
    parser.add_argument("--other-cap",  type=int,   default=6000)
    args = parser.parse_args()

    run(input_csv=args.input,
        output_csv=args.out,
        alpha_min=args.alpha_min,
        stride=args.stride,
        balance=not args.no_balance,
        snow_cap=args.snow_cap,
        other_cap=args.other_cap)


if __name__ == "__main__":
    main()
