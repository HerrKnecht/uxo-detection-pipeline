"""Simulation of a borehole magnetic survey (depth survey).

Practice: boreholes on a grid (typ. 1.5–2 m), a few metres deep (typ. ~7 m).
A magnetometer probe is lowered into the (PVC-cased) borehole.

Simulated channels per depth sample:
    bz : vertical component of the anomaly [nT], positive downward (geomagnetic convention)
    bh : magnitude of the horizontal component [nT]. It is rotation-invariant,
         because the probe rotates in the borehole and Bx/By alone are unreliable.
    gz : vertical gradient (lower minus upper sensor) [nT]

Collision: if a borehole hits an object, the profile is truncated from there (NaN).
In practice drilling and measuring are done step by step to prevent exactly this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import physics
from .noise import BoreholeNoise
from .objects import Scene


@dataclass
class BoreholeSurvey:
    spacing: float = 1.5            # grid spacing of boreholes [m]
    max_depth: float = 7.0          # final depth [m]
    step: float = 0.05              # sample spacing [m]
    gradiometer_base: float = 0.5   # sensor separation in the probe [m]
    collision_margin: float = 0.05  # safety margin for collision check [m]


@dataclass
class BoreholeResult:
    holes_xy: np.ndarray            # (n_holes, 2)
    depth: np.ndarray               # (n_depth,) positive downward
    bz: np.ndarray                  # (n_holes, n_depth)
    bh: np.ndarray
    gz: np.ndarray
    bz_clean: np.ndarray
    hit_target: np.ndarray          # (n_holes,) target_id or -1
    survey: BoreholeSurvey
    noise: BoreholeNoise
    meta: dict = field(default_factory=dict)


def borehole_grid(x_range, y_range, spacing, offset=0.5):
    xs = np.arange(x_range[0] + offset, x_range[1] - offset + 1e-9, spacing)
    ys = np.arange(y_range[0] + offset, y_range[1] - offset + 1e-9, spacing)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def simulate_boreholes(scene: Scene, survey: BoreholeSurvey | None = None,
                       noise: BoreholeNoise | None = None,
                       rng: np.random.Generator | None = None,
                       holes_xy: np.ndarray | None = None) -> BoreholeResult:
    survey = survey or BoreholeSurvey()
    noise = noise or BoreholeNoise()
    rng = rng or np.random.default_rng()
    if holes_xy is None:
        holes_xy = borehole_grid(scene.x_range, scene.y_range, survey.spacing)
    depth = np.arange(0.0, survey.max_depth + 1e-9, survey.step)
    n_h, n_d = len(holes_xy), len(depth)
    sources = scene.sources()

    bz = np.full((n_h, n_d), np.nan)
    bh = np.full((n_h, n_d), np.nan)
    gz = np.full((n_h, n_d), np.nan)
    bz_clean = np.full((n_h, n_d), np.nan)
    hit = np.full(n_h, -1)

    for i, (hx, hy) in enumerate(holes_xy):
        # Borehole deviates slightly from vertical (linear deflection)
        tilt = rng.normal(0, noise.position_sigma_m, 2)
        px = hx + tilt[0] * depth
        py = hy + tilt[1] * depth
        obs = np.column_stack([px, py, -depth])

        # Collision check
        first_hit = n_d
        for t in scene.targets:
            d = np.linalg.norm(obs - t.position[None, :], axis=1)
            inside = d < t.radius + survey.collision_margin
            if inside.any() and int(np.argmax(inside)) < first_hit:
                first_hit = int(np.argmax(inside))
                hit[i] = t.target_id
        valid = np.arange(n_d) < first_hit

        # Geomagnetic convention for output: Z positive downward
        flip = np.array([1.0, 1.0, -1.0])
        b = physics.total_field(obs, sources) * 1e9 * flip
        obs_up = obs + np.array([0, 0, survey.gradiometer_base])
        b_up = physics.total_field(obs_up, sources) * 1e9 * flip

        geo = _layered_geology(rng, n_d, survey.step, noise.geology_sigma_nt, noise.geology_corr_m)
        drift = noise.drift_nt_per_m * depth
        e = lambda: rng.normal(0, noise.sensor_sigma_nt, n_d)

        bz_i = b[:, 2] + geo + drift + e()
        bh_i = np.hypot(b[:, 0] + 0.3 * geo, b[:, 1]) + np.abs(e())
        gz_i = (b[:, 2] - b_up[:, 2]) + np.gradient(geo, survey.step) * survey.gradiometer_base + e()

        bz[i, valid] = bz_i[valid]
        bh[i, valid] = bh_i[valid]
        gz[i, valid] = gz_i[valid]
        bz_clean[i, valid] = b[valid, 2]

    return BoreholeResult(holes_xy, depth, bz, bh, gz, bz_clean, hit, survey, noise)


def _layered_geology(rng, n, step, sigma, corr_m):
    """Soil as a layer sequence: correlated noise along depth."""
    if sigma <= 0:
        return np.zeros(n)
    w = gaussian_filter1d(rng.normal(size=n), max(corr_m / step, 0.5))
    return sigma * w / (w.std() + 1e-12)
