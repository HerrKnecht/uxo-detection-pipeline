"""Create example figures for surface and borehole surveys (docs/img/).

    python scripts/demo_plots.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uxo_pipeline.simulation import (BoreholeNoise, BoreholeSurvey, Scene, SurfaceNoise, SurfaceSurvey,
                                     earth_field, random_scene, sample_target, simulate_boreholes,
                                     simulate_surface)

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
MARK = {"bomb": ("*", "black", 160), "shell": ("^", "black", 70),
        "small": ("v", "black", 50), "clutter": ("o", "white", 25)}


def surface_demo(rng):
    scene = random_scene(rng, (0, 20), (0, 20), n_uxo=3, n_clutter=15)
    res = simulate_surface(scene, SurfaceSurvey(), SurfaceNoise(), rng)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    lim = np.percentile(np.abs(res.data), 99)
    for ax, img, title in [(axes[0], res.clean, "Object signal only (noise-free)"),
                           (axes[1], res.data, "Simulated survey (with noise)")]:
        im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[res.x[0], res.x[-1], res.y[0], res.y[-1]])
        for t in scene.targets:
            m, c, s = MARK[t.label]
            ax.scatter(t.x, t.y, marker=m, s=s, facecolor=c, edgecolor="k", lw=0.7)
        ax.set(title=title, xlabel="East [m]", ylabel="North [m]")
    fig.colorbar(im, ax=axes, label="Vertical gradient [nT]", shrink=0.8)
    fig.suptitle("Surface survey, fluxgate gradiometer   ★ bomb  ▲ shell  ▼ small calibre  ○ clutter")
    fig.savefig(OUT / "surface_demo.png", dpi=130)
    print("Objects:")
    for t in scene.targets:
        print(f"  {t.type_name:12s} depth {t.depth:4.2f} m  |m| = {np.linalg.norm(t.moment):.4f} A·m²")


def borehole_demo(rng):
    b0 = earth_field()
    bomb = sample_target(rng, "bomb_500lb", 4.6, 4.4, b0, target_id=0)
    bomb.depth = 4.5
    scrap = sample_target(rng, "scrap_large", 2.0, 7.0, b0, target_id=1)
    scene = Scene((0, 9), (0, 9), b0, [bomb, scrap])
    res = simulate_boreholes(scene, BoreholeSurvey(spacing=1.5), BoreholeNoise(), rng)

    dist = np.hypot(*(res.holes_xy - [bomb.x, bomb.y]).T)
    order = np.argsort(dist)[:6]
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True, constrained_layout=True)
    for i in order:
        lab = f"distance {dist[i]:.1f} m" + (" (collision)" if res.hit_target[i] >= 0 else "")
        for ax, arr in zip(axes, [res.bz, res.bh, res.gz]):
            ax.plot(arr[i], res.depth, lw=1, label=lab)
    for ax, t in zip(axes, ["Bz [nT]", "|Bh| [nT] (rotation-invariant)", "Gradient Gz [nT]"]):
        ax.set(xlabel=t); ax.axhline(bomb.depth, color="gray", ls=":", lw=1); ax.grid(alpha=.3)
    axes[0].set(ylabel="Depth [m]"); axes[0].invert_yaxis()
    axes[2].legend(fontsize=8)
    fig.suptitle(f"Borehole survey: 500 lb bomb at {bomb.depth:.1f} m depth, 6 nearest boreholes")
    fig.savefig(OUT / "borehole_demo.png", dpi=130)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    surface_demo(rng)
    borehole_demo(rng)
    print(f"Figures saved to {OUT}")
