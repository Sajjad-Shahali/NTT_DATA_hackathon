"""
data_gen.py — Synthetic dataset generation for Burckhardt model training/testing.

Generates (slip, mu) pairs from known surface parameters with configurable noise.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from src.burckhardt import mu as burckhardt_mu, SURFACES, SurfaceParams
from typing import Optional


def generate_surface(
    surface_key: str,
    n_samples: int = 200,
    slip_range: tuple = (0.001, 0.50),
    noise_std: float = 0.01,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic (slip, mu) dataset for one road surface.

    Parameters
    ----------
    surface_key : one of 'dry', 'wet', 'snow', 'ice'
    n_samples   : number of measurement points
    slip_range  : (s_min, s_max) range of slip ratios
    noise_std   : standard deviation of additive Gaussian measurement noise
    seed        : random seed for reproducibility

    Returns
    -------
    DataFrame with columns: slip, mu_true, mu_noisy, surface, c1, c2, c3
    """
    rng = np.random.default_rng(seed)
    params = SURFACES[surface_key]

    # Sample slip values — denser near 0 where curve is steep
    u = rng.uniform(0, 1, n_samples)
    s_min, s_max = slip_range
    # Square-root spacing gives more samples near 0
    slip = s_min + (s_max - s_min) * np.sqrt(u)

    mu_true  = burckhardt_mu(slip, params.c1, params.c2, params.c3)
    mu_noisy = mu_true + rng.normal(0, noise_std, n_samples)
    # Clip to physical range [0, ∞)
    mu_noisy = np.clip(mu_noisy, 0.0, None)

    return pd.DataFrame({
        "slip":    slip,
        "mu_true": mu_true,
        "mu_noisy": mu_noisy,
        "surface": params.name,
        "c1": params.c1,
        "c2": params.c2,
        "c3": params.c3,
    })


def generate_all_surfaces(
    n_samples: int = 200,
    noise_std: float = 0.01,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate dataset for all four standard surfaces and optionally save to CSV.

    Returns combined DataFrame.
    """
    frames = []
    for i, key in enumerate(SURFACES):
        df = generate_surface(key, n_samples=n_samples,
                              noise_std=noise_std, seed=seed + i)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if save_dir is not None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path / "synthetic_friction_data.csv", index=False)
        print(f"Saved {len(combined)} samples to {path / 'synthetic_friction_data.csv'}")

    return combined


def load_or_generate(csv_path: str, **kwargs) -> pd.DataFrame:
    """Load from CSV if it exists, otherwise generate and save."""
    path = Path(csv_path)
    if path.exists():
        print(f"Loading existing data from {csv_path}")
        return pd.read_csv(csv_path)
    print("No data found — generating synthetic dataset.")
    return generate_all_surfaces(save_dir=str(path.parent), **kwargs)
