"""
predict_mu.py — Predict μ (friction coefficient) as a function of slip.

Three approaches compared:
  1. Burckhardt Nonlinear Least Squares  (physics-based, interpretable)
  2. Gaussian Process Regression         (non-parametric, gives uncertainty)
  3. Neural Network (MLP)                (data-driven, black-box)

Usage
-----
  # Generate synthetic data + train + plot
  python predict_mu.py

  # Use your own CSV (columns: slip, mu_noisy, surface)
  python predict_mu.py --data data/raw/my_measurements.csv

  # Single surface only
  python predict_mu.py --surface dry

  # Save plots
  python predict_mu.py --save plots/
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from src.burckhardt import mu as bmu, s_opt, mu_peak, SURFACES
from src.data_gen import generate_all_surfaces, generate_surface

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Surface colours for consistent plotting
# ---------------------------------------------------------------------------
SURFACE_COLORS = {
    "Dry asphalt": "#E74C3C",
    "Wet asphalt": "#3498DB",
    "Snow":        "#95A5A6",
    "Ice":         "#2ECC71",
}


# ===========================================================================
# 1. BURCKHARDT NONLINEAR LEAST SQUARES
# ===========================================================================

def _burckhardt_callable(s, c1, c2, c3):
    """Scalar-safe Burckhardt model for scipy curve_fit."""
    return c1 * (1.0 - np.exp(-c2 * np.abs(s))) - c3 * np.abs(s)


class BurckhardtFitter:
    """
    Fits the Burckhardt model to (slip, mu) data via nonlinear least squares.

    Physics knowledge is embedded in the model structure — only 3 parameters
    need to be identified from data, making this extremely data-efficient.
    """

    # Parameter bounds: [c1_min, c2_min, c3_min] to [c1_max, c2_max, c3_max]
    # Tightened to physically realistic surface range (ice → dry asphalt)
    BOUNDS = ([0.01,  23.0, 0.001],
              [1.50, 310.0, 0.600])

    def __init__(self):
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.covariance = None
        self.rmse_train = None   # in-sample RMSE
        self.rmse       = None   # test RMSE (set externally by train_and_evaluate)

    def fit(self, slip: np.ndarray, mu_data: np.ndarray,
            p0=(1.0, 30.0, 0.3)) -> "BurckhardtFitter":
        """
        Fit c1, c2, c3 to the measured (slip, mu) pairs.

        Parameters
        ----------
        slip    : 1-D array of wheel slip values
        mu_data : 1-D array of measured friction coefficients
        p0      : initial guess (c1, c2, c3) — use LUT estimates if available

        Returns self for chaining.
        """
        popt, pcov = curve_fit(
            _burckhardt_callable,
            slip, mu_data,
            p0=p0,
            bounds=self.BOUNDS,
            maxfev=10_000,
        )
        self.c1, self.c2, self.c3 = popt
        self.covariance = pcov
        mu_pred = _burckhardt_callable(slip, *popt)
        self.rmse_train = float(np.sqrt(mean_squared_error(mu_data, mu_pred)))
        self.rmse = self.rmse_train  # overwritten with test RMSE by train_and_evaluate
        return self

    def predict(self, slip: np.ndarray) -> np.ndarray:
        """Evaluate the fitted Burckhardt curve."""
        if self.c1 is None:
            raise RuntimeError("Call fit() before predict().")
        return _burckhardt_callable(slip, self.c1, self.c2, self.c3)

    def report(self) -> str:
        if self.c1 is None:
            return "Not fitted."
        s = s_opt(self.c1, self.c2, self.c3)
        mp = mu_peak(self.c1, self.c2, self.c3)
        test_str = f"  test_RMSE={self.rmse:.5f}" if self.rmse != self.rmse_train else ""
        return (f"  c1={self.c1:.4f}  c2={self.c2:.2f}  c3={self.c3:.4f}  "
                f"s_opt={s:.4f}  mu_peak={mp:.4f}  train_RMSE={self.rmse_train:.5f}{test_str}")


# ===========================================================================
# 2. GAUSSIAN PROCESS REGRESSION
# ===========================================================================

class GPFrictionPredictor:
    """
    Non-parametric friction predictor with uncertainty bounds.

    Kernel = ConstantKernel × RBF + WhiteKernel (noise floor).
    Best used when Burckhardt model fit is poor or when confidence
    intervals are needed for safety-critical decisions.
    """

    def __init__(self, n_restarts: int = 5):
        kernel = (
            ConstantKernel(1.0, (1e-3, 10.0))
            * RBF(length_scale=0.05, length_scale_bounds=(1e-3, 1.0))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 0.1))
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
        )
        self.rmse = None

    def fit(self, slip: np.ndarray, mu_data: np.ndarray) -> "GPFrictionPredictor":
        self.gp.fit(slip.reshape(-1, 1), mu_data)
        mu_pred, _ = self.gp.predict(slip.reshape(-1, 1), return_std=True)
        self.rmse_train = float(np.sqrt(mean_squared_error(mu_data, mu_pred)))
        self.rmse = self.rmse_train  # overwritten with test RMSE by train_and_evaluate
        return self

    def predict(self, slip: np.ndarray):
        """Returns (mu_mean, mu_std)."""
        mean, std = self.gp.predict(slip.reshape(-1, 1), return_std=True)
        return mean, std


# ===========================================================================
# 3. NEURAL NETWORK (MLP)
# ===========================================================================

class NNFrictionPredictor:
    """
    MLP regressor — included for comparison with physics-based methods.

    Note: typically underperforms Burckhardt fitting on small datasets
    due to overfitting risk. Requires 10× more data for equivalent accuracy.
    """

    def __init__(self):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate_init=1e-3,
        )
        self.rmse_train = None
        self.rmse       = None   # overwritten with test RMSE by train_and_evaluate

    def fit(self, slip: np.ndarray, mu_data: np.ndarray) -> "NNFrictionPredictor":
        X = self.scaler_x.fit_transform(slip.reshape(-1, 1))
        y = self.scaler_y.fit_transform(mu_data.reshape(-1, 1)).ravel()
        self.model.fit(X, y)
        mu_pred = self.predict(slip)
        self.rmse_train = float(np.sqrt(mean_squared_error(mu_data, mu_pred)))
        self.rmse = self.rmse_train  # overwritten with test RMSE by train_and_evaluate
        return self

    def predict(self, slip: np.ndarray) -> np.ndarray:
        X = self.scaler_x.transform(slip.reshape(-1, 1))
        y_scaled = self.model.predict(X)
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


# ===========================================================================
# TRAINING & EVALUATION
# ===========================================================================

def train_and_evaluate(df_surface: pd.DataFrame, surface_name: str,
                       n_train: int = 50) -> dict:
    """
    Train all three models on n_train points and evaluate on the full curve.
    Returns dict with fitted models and metrics.
    """
    slip_all = df_surface["slip"].values
    mu_all   = df_surface["mu_noisy"].values

    # Shuffle and split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(slip_all))
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    slip_train = slip_all[train_idx]
    mu_train   = mu_all[train_idx]
    slip_test  = slip_all[test_idx]
    mu_test    = mu_all[test_idx]

    # Sort training data for display
    sort_order = np.argsort(slip_train)
    slip_train_s = slip_train[sort_order]
    mu_train_s   = mu_train[sort_order]

    results = {
        "surface": surface_name,
        "slip_train": slip_train_s,
        "mu_train":   mu_train_s,
        "slip_test":  slip_test,
        "mu_test":    mu_test,
        "slip_all":   slip_all,
        "mu_all":     mu_all,
    }

    # Ground truth parameters — only available for synthetic data
    row = df_surface.iloc[0]
    results["gt_c1"] = row["c1"] if "c1" in df_surface.columns else None
    results["gt_c2"] = row["c2"] if "c2" in df_surface.columns else None
    results["gt_c3"] = row["c3"] if "c3" in df_surface.columns else None

    print(f"\n{'='*60}")
    print(f"  Surface: {surface_name}  |  train={n_train}  test={len(test_idx)}")
    if results["gt_c1"] is not None:
        print(f"  Ground truth: c1={results['gt_c1']:.4f}  c2={results['gt_c2']:.2f}  c3={results['gt_c3']:.4f}")
    print(f"{'='*60}")

    # 1. Burckhardt fitting
    bf = BurckhardtFitter()
    try:
        bf.fit(slip_train, mu_train)
        mu_test_pred = bf.predict(slip_test)
        bf.rmse = float(np.sqrt(mean_squared_error(mu_test, mu_test_pred)))
        print(f"  [Burckhardt NLS]  {bf.report()}")
    except Exception as e:
        print(f"  [Burckhardt NLS]  FAILED: {e}")
    results["burckhardt"] = bf

    # 2. Gaussian Process
    gp = GPFrictionPredictor()
    try:
        gp.fit(slip_train, mu_train)
        mu_test_pred, _ = gp.predict(slip_test)
        gp.rmse = float(np.sqrt(mean_squared_error(mu_test, mu_test_pred)))
        print(f"  [Gaussian Process] train_RMSE={gp.rmse_train:.5f}  test_RMSE={gp.rmse:.5f}")
    except Exception as e:
        print(f"  [Gaussian Process] FAILED: {e}")
    results["gp"] = gp

    # 3. Neural Network
    nn = NNFrictionPredictor()
    try:
        nn.fit(slip_train, mu_train)
        mu_test_pred = nn.predict(slip_test)
        nn.rmse = float(np.sqrt(mean_squared_error(mu_test, mu_test_pred)))
        print(f"  [Neural Network]   train_RMSE={nn.rmse_train:.5f}  test_RMSE={nn.rmse:.5f}")
    except Exception as e:
        print(f"  [Neural Network]   FAILED: {e}")
    results["nn"] = nn

    return results


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_results(all_results: list, save_dir: str = None):
    """
    4-panel figure: one subplot per surface.
    Each panel shows: data, ground truth, Burckhardt fit, GP (±2σ), NN.
    """
    n_surfaces = len(all_results)
    fig = plt.figure(figsize=(7 * n_surfaces, 6))
    gs  = gridspec.GridSpec(1, n_surfaces, figure=fig, wspace=0.35)

    s_dense = np.linspace(0.001, 0.50, 500)

    for col, res in enumerate(all_results):
        ax = fig.add_subplot(gs[0, col])
        color = SURFACE_COLORS.get(res["surface"], "#333333")

        # Ground truth curve (only for synthetic data where params are known)
        if res["gt_c1"] is not None:
            mu_gt = bmu(s_dense, res["gt_c1"], res["gt_c2"], res["gt_c3"])
            ax.plot(s_dense, mu_gt,
                    color=color, lw=2.5, label="Ground truth", zorder=5)

        # Training data scatter
        ax.scatter(res["slip_train"], res["mu_train"],
                   s=18, color=color, alpha=0.5,
                   label=f"Train (n={len(res['slip_train'])})",
                   zorder=4)

        # Test data scatter (held-out — shows generalisation)
        ax.scatter(res["slip_test"], res["mu_test"],
                   s=14, color=color, marker="x", alpha=0.7, linewidths=0.8,
                   label=f"Test (n={len(res['slip_test'])})",
                   zorder=4)

        # Burckhardt NLS fit
        bf = res["burckhardt"]
        if bf.c1 is not None:
            mu_bf = bf.predict(s_dense)
            ax.plot(s_dense, mu_bf,
                    color="black", lw=1.8, ls="--",
                    label=f"Burckhardt NLS  test RMSE={bf.rmse:.4f}",
                    zorder=6)
            # Mark s_opt
            s_op = s_opt(bf.c1, bf.c2, bf.c3)
            ax.axvline(s_op, color="black", lw=0.8, ls=":", alpha=0.6)
            ax.annotate(f"s_opt={s_op:.3f}",
                        xy=(s_op, bf.predict(np.array([s_op]))[0]),
                        xytext=(s_op + 0.03, bf.predict(np.array([s_op]))[0] * 0.85),
                        fontsize=7, color="black",
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

        # Gaussian Process ±2σ band
        gp = res["gp"]
        try:
            mu_gp, sigma_gp = gp.predict(s_dense)
            ax.plot(s_dense, mu_gp,
                    color="#E67E22", lw=1.6, ls="-.",
                    label=f"GP  test RMSE={gp.rmse:.4f}", zorder=5)
            ax.fill_between(s_dense,
                            mu_gp - 2 * sigma_gp,
                            mu_gp + 2 * sigma_gp,
                            color="#E67E22", alpha=0.12,
                            label="GP ±2σ")
        except Exception:
            pass

        # Neural Network (s_dense is already sorted — argsort is a no-op)
        nn = res["nn"]
        try:
            mu_nn = nn.predict(s_dense)
            ax.plot(s_dense, mu_nn,
                    color="#9B59B6", lw=1.4, ls=":",
                    label=f"Neural Net  test RMSE={nn.rmse:.4f}", zorder=4)
        except Exception:
            pass

        ax.set_title(res["surface"], fontsize=12, fontweight="bold",
                     color=color)
        ax.set_xlabel("Slip ratio  s", fontsize=10)
        ax.set_ylabel("Friction coefficient  μ", fontsize=10)
        ax.set_xlim(0, 0.52)
        ax.set_ylim(bottom=-0.01)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25, lw=0.5)

    fig.suptitle(
        "μ(s) Prediction — Burckhardt NLS  vs  Gaussian Process  vs  Neural Network",
        fontsize=13, fontweight="bold", y=1.02
    )

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / "mu_prediction.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {out}")

    plt.tight_layout()
    plt.show()


def plot_model_comparison_summary(all_results: list, save_dir: str = None):
    """Bar chart comparing RMSE across methods and surfaces."""
    surfaces = [r["surface"] for r in all_results]
    methods  = ["Burckhardt NLS", "Gaussian Process", "Neural Network"]
    keys     = ["burckhardt", "gp", "nn"]
    colors_m = ["#2C3E50", "#E67E22", "#9B59B6"]

    rmse_table = np.zeros((len(methods), len(surfaces)))
    for j, res in enumerate(all_results):
        for i, key in enumerate(keys):
            model = res[key]
            if model.rmse is not None:
                rmse_table[i, j] = model.rmse

    x = np.arange(len(surfaces))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (method, color) in enumerate(zip(methods, colors_m)):
        bars = ax.bar(x + i * width, rmse_table[i], width,
                      label=method, color=color, alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, rmse_table[i]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.0003,
                        f"{val:.4f}", ha="center", va="bottom",
                        fontsize=7.5, color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(surfaces, fontsize=11)
    ax.set_ylabel("RMSE  (friction coefficient)", fontsize=11)
    ax.set_title("Model Accuracy Comparison by Road Surface", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.set_ylim(0, max(float(rmse_table.max()) * 1.3, 1e-4))

    if save_dir:
        out = Path(save_dir) / "rmse_comparison.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out}")

    plt.tight_layout()
    plt.show()


# ===========================================================================
# DATA REQUIREMENTS PRINTOUT
# ===========================================================================

def print_data_requirements():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           DATA REQUIREMENTS TO PREDICT μ(s)                     ║
╠══════════════════════════════════════════════════════════════════╣
║  MINIMUM (curve fitting, 3 params):                              ║
║    • 10–20 (slip, μ) pairs per road surface                     ║
║    • Slip range must cover [0.01, 0.45]                          ║
║    • Noise < 5% of peak μ                                        ║
║                                                                  ║
║  RECOMMENDED (robust identification):                            ║
║    • 200+ samples per surface                                    ║
║    • Multiple temperatures, loads, tire compounds                ║
║                                                                  ║
║  SIGNALS NEEDED per timestep:                                    ║
║    slip s    = (v_wheel − v_vehicle) / v_vehicle                 ║
║    μ        = F_x / F_z   (longitudinal force / normal load)    ║
║    F_x from : torque model  T_engine − I_wheel·α / R_wheel      ║
║    F_z from : static weight + accelerometer load transfer        ║
║    v_vehicle: GPS / optical / non-driven wheel average           ║
║    v_wheel  : ABS wheel speed sensor                             ║
║                                                                  ║
║  FOR REAL-TIME ESC IDENTIFIER (hackaton_id.py):                  ║
║    + g_esc  : ESC gradient from LPF demodulator                  ║
║    + a_esc  : ESC dither amplitude (design param)                ║
║    + s_probe: ESC-perturbed slip setpoint                        ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Predict μ(s) — tire-road friction curve identification"
    )
    parser.add_argument("--data",    type=str,  default=None,
                        help="Path to CSV with columns: slip, mu_noisy, surface")
    parser.add_argument("--surface", type=str,  default=None,
                        choices=list(SURFACES.keys()),
                        help="Train on one surface only")
    parser.add_argument("--n-train", type=int,  default=50,
                        help="Number of training points per surface (default: 50)")
    parser.add_argument("--noise",   type=float, default=0.01,
                        help="Measurement noise std (default: 0.01)")
    parser.add_argument("--save",    type=str,  default=None,
                        help="Directory to save plots")
    args = parser.parse_args()

    print_data_requirements()

    # ---- Load or generate data ----------------------------------------
    if args.data:
        print(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        surface_col = "surface" if "surface" in df.columns else None
        surfaces_in_data = df[surface_col].unique() if surface_col else ["Unknown"]
        all_results = []
        for surf in surfaces_in_data:
            subset = df[df[surface_col] == surf] if surface_col else df
            r = train_and_evaluate(subset, surf, n_train=args.n_train)
            all_results.append(r)
    else:
        print("No --data provided. Generating synthetic dataset...")
        surface_keys = [args.surface] if args.surface else list(SURFACES.keys())
        all_results = []
        for i, key in enumerate(surface_keys):
            df_s = generate_surface(key, n_samples=300, noise_std=args.noise,
                                    seed=42 + i)
            r = train_and_evaluate(df_s, SURFACES[key].name,
                                   n_train=args.n_train)
            all_results.append(r)

    # ---- Plots -------------------------------------------------------
    plot_results(all_results, save_dir=args.save)
    plot_model_comparison_summary(all_results, save_dir=args.save)

    # ---- Final summary ----------------------------------------------
    print("\n" + "="*60)
    print("  VERDICT: Best approach for this problem")
    print("="*60)
    print("""
  Burckhardt Nonlinear Least Squares wins because:
  1. Only 3 parameters to identify — tiny data requirement
  2. Physics model prevents overfitting by design
  3. Parameters c1,c2,c3 are physically interpretable
  4. s_opt is computed analytically from fitted params
  5. Works in real-time when paired with ESC (hackaton_id.py)

  Use Gaussian Process when:
  - Road surface doesn't fit Burckhardt well (e.g. mixed surfaces)
  - Safety-critical and uncertainty bounds are needed

  Use Neural Network when:
  - 1000+ samples available per surface
  - Multi-modal inputs (temperature, speed, tire type)
  - Combined slip (longitudinal + lateral) is important
""")


if __name__ == "__main__":
    main()
