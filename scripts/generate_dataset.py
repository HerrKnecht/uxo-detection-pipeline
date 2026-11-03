"""Generate a synthetic dataset (surface + borehole) with ground truth.

Example:
    python scripts/generate_dataset.py --out data/v1 --n-surface 200 --n-borehole 50 --seed 42

Output:
    <out>/surface/survey_0000.npz   x, y, data, clean (+ parameters as JSON string)
    <out>/surface/targets.csv       all objects with survey_id, position, type, moment ...
    <out>/borehole/field_0000.npz   holes_xy, depth, bz, bh, gz, bz_clean, hit_target
    <out>/borehole/targets.csv
    <out>/config.json               generator settings (reproducibility)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from uxo_pipeline.simulation import (BoreholeNoise, BoreholeSurvey, SurfaceNoise, SurfaceSurvey,
                                     random_scene, simulate_boreholes, simulate_surface)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/synthetic_v1"))
    p.add_argument("--n-surface", type=int, default=50)
    p.add_argument("--n-borehole", type=int, default=20)
    p.add_argument("--size", type=float, default=20.0, help="edge length of survey area [m]")
    p.add_argument("--borehole-size", type=float, default=9.0, help="edge length of borehole field [m]")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    (args.out / "surface").mkdir(parents=True, exist_ok=True)
    (args.out / "borehole").mkdir(parents=True, exist_ok=True)
    args.out.joinpath("config.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2))

    # --- Surface surveys ---
    records = []
    for i in range(args.n_surface):
        scene = random_scene(rng, (0, args.size), (0, args.size), n_uxo=(0, 3), n_clutter=(3, 20))
        survey = SurfaceSurvey(line_spacing=float(rng.choice([0.25, 0.5])))
        noise = SurfaceNoise.random(rng)
        res = simulate_surface(scene, survey, noise, rng)
        np.savez_compressed(args.out / "surface" / f"survey_{i:04d}.npz",
                            x=res.x, y=res.y, data=res.data.astype(np.float32),
                            clean=res.clean.astype(np.float32),
                            params=json.dumps({"survey": asdict(survey), "noise": asdict(noise)}))
        records += [dict(survey_id=i, **t.to_record()) for t in scene.targets]
        print(f"\rsurface {i + 1}/{args.n_surface}", end="")
    pd.DataFrame(records).to_csv(args.out / "surface" / "targets.csv", index=False)
    print()

    # --- Borehole surveys ---
    # Borehole fields are smaller and more often contain a deep bomb.
    records = []
    bomb_mix = {"bomb_1000lb": 0.2, "bomb_500lb": 0.4, "bomb_250lb": 0.4}
    for i in range(args.n_borehole):
        scene = random_scene(rng, (0, args.borehole_size), (0, args.borehole_size),
                             n_uxo=(0, 2), n_clutter=(2, 8), uxo_mix=bomb_mix, margin=1.5)
        survey = BoreholeSurvey(spacing=float(rng.choice([1.5, 2.0])))
        noise = BoreholeNoise.random(rng)
        res = simulate_boreholes(scene, survey, noise, rng)
        np.savez_compressed(args.out / "borehole" / f"field_{i:04d}.npz",
                            holes_xy=res.holes_xy, depth=res.depth,
                            bz=res.bz.astype(np.float32), bh=res.bh.astype(np.float32),
                            gz=res.gz.astype(np.float32), bz_clean=res.bz_clean.astype(np.float32),
                            hit_target=res.hit_target,
                            params=json.dumps({"survey": asdict(survey), "noise": asdict(noise)}))
        records += [dict(field_id=i, **t.to_record()) for t in scene.targets]
        print(f"\rborehole {i + 1}/{args.n_borehole}", end="")
    pd.DataFrame(records).to_csv(args.out / "borehole" / "targets.csv", index=False)
    print(f"\nDone: {args.out.resolve()}")


if __name__ == "__main__":
    main()
