# Changelog

## [0.1.0] – 2026-11-03

First public release: synthetic data simulator.

### Added
- Magnetic dipole physics: Earth field, demagnetisation factors of prolate/oblate
  spheroids, induced and remanent moments.
- Object catalogue: aircraft bombs (250/500/1000 lb), artillery shells (75–150 mm),
  81 mm mortar, 20 mm projectiles, fragments, scrap, pipes/rebar.
- Surface survey simulation (fluxgate vertical gradiometer or total-field magnetometer).
- Borehole survey simulation (Bz, rotation-invariant |Bh|, vertical gradient, collision check).
- Realistic noise: sensor noise, line offsets / heading error, magnetic soil,
  GNSS position error, sensor height variation, probe drift.
- Dataset generator with ground truth (`scripts/generate_dataset.py`).
- Physics tests against analytical solutions; CI via GitHub Actions.
