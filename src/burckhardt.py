"""
burckhardt.py — Burckhardt tire friction model utilities.

Model:  μ(s) = c1·(1 − exp(−c2·|s|)) − c3·|s|

Parameters by surface (typical values from literature):
    Surface         c1      c2      c3
    Dry asphalt   1.2801  23.99   0.5200
    Wet asphalt   0.8570  33.82   0.3470
    Snow          0.1946  94.13   0.0646
    Ice           0.0500 306.40   0.0010
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Surface presets
# ---------------------------------------------------------------------------

@dataclass
class SurfaceParams:
    name: str
    c1: float
    c2: float
    c3: float

    def to_tuple(self) -> Tuple[float, float, float]:
        return self.c1, self.c2, self.c3


SURFACES = {
    "dry":  SurfaceParams("Dry asphalt", c1=1.2801, c2=23.99,  c3=0.5200),
    "wet":  SurfaceParams("Wet asphalt", c1=0.8570, c2=33.82,  c3=0.3470),
    "snow": SurfaceParams("Snow",        c1=0.1946, c2=94.13,  c3=0.0646),
    "ice":  SurfaceParams("Ice",         c1=0.0500, c2=306.40, c3=0.0010),
}


# ---------------------------------------------------------------------------
# Core model functions
# ---------------------------------------------------------------------------

def mu(s: np.ndarray, c1: float, c2: float, c3: float) -> np.ndarray:
    """
    Burckhardt friction coefficient.

    Parameters
    ----------
    s  : wheel slip ratio (scalar or array)
    c1 : amplitude parameter
    c2 : sharpness parameter
    c3 : descending slope parameter

    Returns
    -------
    μ(s)  friction coefficient (same shape as s)
    """
    s = np.asarray(s, dtype=float)
    return c1 * (1.0 - np.exp(-c2 * np.abs(s))) - c3 * np.abs(s)


def gradient(s: np.ndarray, c1: float, c2: float, c3: float) -> np.ndarray:
    """
    dμ/d|s| — analytical gradient of the friction curve.

    At s_opt: gradient = 0, giving s_opt = ln(c1*c2/c3) / c2
    """
    s = np.asarray(s, dtype=float)
    return c1 * c2 * np.exp(-c2 * np.abs(s)) - c3


def s_opt(c1: float, c2: float, c3: float) -> float:
    """Optimal slip: slip value that maximises μ(s). Returns 0.0 if no peak exists."""
    if c3 <= 0.0 or c2 <= 0.0:
        return 0.0
    ratio = c1 * c2 / c3
    if ratio <= 1.0:
        return 0.0
    return float(np.clip(np.log(ratio) / c2, 0.01, 0.45))


def mu_peak(c1: float, c2: float, c3: float) -> float:
    """Peak (maximum) friction coefficient."""
    s = s_opt(c1, c2, c3)
    return float(mu(s, c1, c2, c3))


def is_valid(c1: float, c2: float, c3: float) -> bool:
    """Returns True when a physical peak exists (c1*c2 > c3)."""
    return c1 * c2 > c3
