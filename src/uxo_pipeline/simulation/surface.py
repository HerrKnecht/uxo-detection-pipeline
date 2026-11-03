"""Simulation of a surface (area) magnetic survey.

Default: fluxgate vertical gradiometer (two vertical sensors stacked), as used by
the multi-channel systems common in Germany. Optional: total field (caesium).

The result is a regular grid (survey lines running North, line spacing dx).
Values are reported at the *nominal* positions; the *true* sensor positions are
shifted by GNSS error, just as in practice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import physics
from .noise import SurfaceNoise, geology_field, line_offsets
from .objects import Scene


@dataclass
class SurfaceSurvey:
    line_spacing: float = 0.5        # line spacing [m]
    sample_spacing: float = 0.1      # sample spacing along line [m]
    sensor_height: float = 0.3       # height of lower sensor above ground [m]
    gradiometer_base: float = 0.65   # vertical distance upper/lower sensor [m]
    measurement: str = "gradient"    # "gradient" (fluxgate) or "tmi" (total field)


@dataclass
class SurfaceResult:
    x: np.ndarray                    # (nx,) line positions (East)
    y: np.ndarray                    # (ny,) positions along line (North)
    data: np.ndarray                 # (ny, nx) reading [nT], with disturbances
    clean: np.ndarray                # (ny, nx) object signal only [nT], no disturbances
    survey: SurfaceSurvey
    noise: SurfaceNoise
    meta: dict = field(default_factory=dict)


def simulate_surface(scene: Scene, survey: SurfaceSurvey | None = None,
                     noise: SurfaceNoise | None = None,
                     rng: np.random.Generator | None = None) -> SurfaceResult:
    survey = survey or SurfaceSurvey()
    noise = noise or SurfaceNoise()
    rng = rng or np.random.default_rng()

    x = np.arange(scene.x_range[0], scene.x_range[1] + 1e-9, survey.line_spacing)
    y = np.arange(scene.y_range[0], scene.y_range[1] + 1e-9, survey.sample_spacing)
    xx, yy = np.meshgrid(x, y)  # (ny, nx)
    shape = xx.shape

    def measure(px, py, h):
        obs_low = np.column_stack([px.ravel(), py.ravel(), h.ravel()])
        sources = scene.sources()
        if survey.measurement == "gradient":
            obs_up = obs_low + np.array([0, 0, survey.gradiometer_base])
            # Geomagnetic convention: Z positive downward -> flip sign.
            # An induced-magnetised object then produces a positive peak.
            z_low = -physics.total_field(obs_low, sources)[:, 2]
            z_up = -physics.total_field(obs_up, sources)[:, 2]
            val = z_low - z_up
        elif survey.measurement == "tmi":
            val = physics.tmi_anomaly(physics.total_field(obs_low, sources), scene.b0)
        else:
            raise ValueError(f"Unknown measurement type: {survey.measurement}")
        return (val * 1e9).reshape(shape)

    h0 = np.full(shape, survey.sensor_height)
    clean = measure(xx, yy, h0)

    # True positions deviate from nominal positions
    px = xx + rng.normal(0, noise.position_sigma_m, shape)
    py = yy + rng.normal(0, noise.position_sigma_m, shape)
    h = h0 + _smooth_1d(rng, shape, noise.height_sigma_m, survey.sample_spacing, 0.5)
    h = np.maximum(h, 0.05)
    data = measure(px, py, h)

    data += rng.normal(0, noise.sensor_sigma_nt, shape)
    data += line_offsets(rng, shape[1], noise.line_offset_sigma_nt, noise.heading_offset_nt)[None, :]
    data += geology_field(rng, shape, (survey.sample_spacing, survey.line_spacing),
                          noise.geology_sigma_nt, noise.geology_corr_m)

    return SurfaceResult(x, y, data, clean, survey, noise)


def _smooth_1d(rng, shape, sigma, spacing, corr_m):
    """Variation correlated along lines (e.g. sensor height over uneven ground)."""
    if sigma <= 0:
        return np.zeros(shape)
    w = gaussian_filter1d(rng.normal(size=shape), corr_m / spacing, axis=0)
    return sigma * w / (w.std() + 1e-12)
