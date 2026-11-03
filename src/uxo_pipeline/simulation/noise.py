"""Realistic disturbances for synthetic survey data.

Noise-free synthetic data is too "clean": a model trained on it fails on real
data. The typical error sources are therefore reproduced:
- sensor noise (white)
- line offsets / heading error (stripes in the map, depending on walking direction)
- geological background (magnetic soil, spatially correlated)
- position error (GNSS) and varying sensor height
- drift (borehole probe)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class SurfaceNoise:
    sensor_sigma_nt: float = 0.3        # white noise per reading
    line_offset_sigma_nt: float = 0.8   # random offset per survey line
    heading_offset_nt: float = 0.5      # +/- depending on direction (zig-zag)
    geology_sigma_nt: float = 1.0       # strength of soil signal
    geology_corr_m: float = 1.5         # correlation length of soil signal
    position_sigma_m: float = 0.03      # GNSS error (horizontal)
    height_sigma_m: float = 0.04        # sensor height variation

    @classmethod
    def clean(cls) -> "SurfaceNoise":
        return cls(0, 0, 0, 0, 1.0, 0, 0)

    @classmethod
    def random(cls, rng: np.random.Generator) -> "SurfaceNoise":
        """Random survey conditions from 'good' to 'difficult'."""
        return cls(sensor_sigma_nt=rng.uniform(0.1, 0.6),
                   line_offset_sigma_nt=rng.uniform(0.2, 2.0),
                   heading_offset_nt=rng.uniform(0.0, 1.5),
                   geology_sigma_nt=rng.lognormal(np.log(0.8), 0.7),
                   geology_corr_m=rng.uniform(0.5, 3.0),
                   position_sigma_m=rng.uniform(0.01, 0.08),
                   height_sigma_m=rng.uniform(0.01, 0.08))


def geology_field(rng: np.random.Generator, shape: tuple[int, int],
                  spacing: tuple[float, float], sigma_nt: float, corr_m: float) -> np.ndarray:
    """Spatially correlated random field as a stand-in for magnetic soil."""
    if sigma_nt <= 0:
        return np.zeros(shape)
    white = rng.normal(size=shape)
    sig = (corr_m / spacing[0], corr_m / spacing[1])
    smooth = gaussian_filter(white, sigma=sig, mode="wrap")
    smooth /= smooth.std() + 1e-12
    return sigma_nt * smooth


def line_offsets(rng: np.random.Generator, n_lines: int, sigma_nt: float,
                 heading_nt: float) -> np.ndarray:
    """Offset per line: random part + alternating sign from zig-zag surveying."""
    zigzag = heading_nt * np.where(np.arange(n_lines) % 2 == 0, 1.0, -1.0)
    return rng.normal(0, sigma_nt, n_lines) + zigzag


@dataclass
class BoreholeNoise:
    sensor_sigma_nt: float = 0.5
    drift_nt_per_m: float = 0.3         # linear drift over borehole depth
    geology_sigma_nt: float = 2.0       # soil layers with varying susceptibility
    geology_corr_m: float = 0.4
    position_sigma_m: float = 0.02      # deviation of the borehole from vertical (per metre)

    @classmethod
    def clean(cls) -> "BoreholeNoise":
        return cls(0, 0, 0, 0.4, 0)

    @classmethod
    def random(cls, rng: np.random.Generator) -> "BoreholeNoise":
        return cls(sensor_sigma_nt=rng.uniform(0.2, 1.0),
                   drift_nt_per_m=rng.uniform(-0.8, 0.8),
                   geology_sigma_nt=rng.lognormal(np.log(1.5), 0.6),
                   geology_corr_m=rng.uniform(0.2, 1.0),
                   position_sigma_m=rng.uniform(0.0, 0.03))
