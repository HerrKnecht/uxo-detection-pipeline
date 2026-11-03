import numpy as np
import pytest

from uxo_pipeline.simulation import physics as ph
from uxo_pipeline.simulation import (Scene, SurfaceNoise, SurfaceSurvey, BoreholeNoise, BoreholeSurvey,
                                     random_scene, sample_target, simulate_surface, simulate_boreholes)


def test_demag_sphere():
    assert ph.demag_factors(1.0) == pytest.approx((1 / 3, 1 / 3))
    # limit near 1 is continuous from both sides
    assert ph.demag_factors(1.001)[0] == pytest.approx(1 / 3, abs=1e-3)
    assert ph.demag_factors(0.999)[0] == pytest.approx(1 / 3, abs=1e-3)


@pytest.mark.parametrize("aspect", [0.05, 0.3, 2.0, 5.0, 30.0])
def test_demag_sum_and_limits(aspect):
    n_ax, n_tr = ph.demag_factors(aspect)
    assert n_ax + 2 * n_tr == pytest.approx(1.0)
    assert 0 < n_ax < 1
    if aspect > 1:
        assert n_ax < 1 / 3   # elongated object is easily magnetised along its axis
    else:
        assert n_ax > 1 / 3


def test_dipole_on_axis():
    """On the dipole axis B = mu0/(4 pi) * 2m / r³."""
    m = np.array([0, 0, 1.0])
    b = ph.dipole_field(np.array([[0, 0, 2.0]]), np.zeros(3), m)[0]
    assert b[2] == pytest.approx(1e-7 * 2 / 8)
    assert b[0] == pytest.approx(0) and b[1] == pytest.approx(0)


def test_dipole_decay_r3():
    m = np.array([0.3, -0.2, 1.0])
    b1 = np.linalg.norm(ph.dipole_field(np.array([[1, 1, 1.0]]), np.zeros(3), m))
    b2 = np.linalg.norm(ph.dipole_field(np.array([[2, 2, 2.0]]), np.zeros(3), m))
    assert b1 / b2 == pytest.approx(8.0)


def test_sphere_induced_moment():
    """Highly permeable sphere: m = 3 V B0 / mu0 (chi -> infinity)."""
    b0 = ph.earth_field()
    d = 0.1
    m = ph.induced_moment(b0, d, d, np.array([0, 0, 1.0]), chi=1e9)
    vol = np.pi / 6 * d ** 3
    np.testing.assert_allclose(m, 3 * vol * b0 / ph.MU0, rtol=1e-6)


def test_bomb_amplitude_plausible():
    """Vertical 500 lb bomb at 3 m depth: positive gradient peak, order of 10–150 nT."""
    rng = np.random.default_rng(1)
    b0 = ph.earth_field()
    t = sample_target(rng, "bomb_500lb", 5.0, 5.0, b0)
    t.depth, t.remanence_ratio = 3.0, 0.0
    t.moment = ph.induced_moment(b0, t.length, t.diameter, ph.axis_vector(0, 90))
    scene = Scene((0, 10), (0, 10), b0, [t])
    res = simulate_surface(scene, SurfaceSurvey(), SurfaceNoise.clean(), rng)
    assert 10 < res.clean.max() < 150
    assert res.clean.max() > abs(res.clean.min())   # peak dominates (induced, northern hemisphere)


def test_surface_shapes_and_noise():
    rng = np.random.default_rng(0)
    scene = random_scene(rng, (0, 10), (0, 10))
    res = simulate_surface(scene, SurfaceSurvey(line_spacing=0.5, sample_spacing=0.1), SurfaceNoise(), rng)
    assert res.data.shape == (101, 21)
    assert not np.allclose(res.data, res.clean)


def test_borehole_collision():
    rng = np.random.default_rng(0)
    b0 = ph.earth_field()
    t = sample_target(rng, "bomb_500lb", 1.0, 1.0, b0)
    t.depth = 4.0
    scene = Scene((0, 2), (0, 2), b0, [t])
    res = simulate_boreholes(scene, BoreholeSurvey(), BoreholeNoise.clean(), rng,
                             holes_xy=np.array([[1.0, 1.0], [3.0, 1.0]]))
    assert res.hit_target[0] == t.target_id      # directly above -> collision
    assert np.isnan(res.bz[0, -1])               # profile truncated
    assert res.hit_target[1] == -1               # 2 m away -> free
    assert np.isfinite(res.bz[1]).all()
