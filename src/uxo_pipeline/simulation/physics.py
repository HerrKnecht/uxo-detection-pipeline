"""Basic magnetic physics for UXO simulation.

Coordinate system (used throughout the package):
    x = East, y = North, z = Up (z = 0 is the ground surface).
    Buried objects therefore have negative z values.
Units: SI (metre, tesla, A·m²). Measured values are reported in nT.

Model:
    A ferromagnetic object is approximated as a uniformly magnetised spheroid.
    In the far field (distance > ~2x object size) it acts as a point dipole with
    moment m = m_induced + m_remanent.
"""
from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi  # vacuum permeability [T·m/A]


def earth_field(total_nt: float = 49_500.0,
                inclination_deg: float = 66.0,
                declination_deg: float = 3.0) -> np.ndarray:
    """Earth's magnetic field as a vector [T] in (East, North, Up).

    Defaults roughly match central Germany (IGRF, ~2025):
    F ≈ 49,500 nT, inclination ≈ 66°, declination ≈ 3–4° East.
    """
    inc = np.radians(inclination_deg)
    dec = np.radians(declination_deg)
    bh = total_nt * np.cos(inc)
    return 1e-9 * np.array([bh * np.sin(dec),          # East
                            bh * np.cos(dec),          # North
                            -total_nt * np.sin(inc)])  # Up (field points downward)


def demag_factors(aspect: float) -> tuple[float, float]:
    """Demagnetisation factors (N_axial, N_transverse) of a spheroid.

    aspect = length along the symmetry axis / diameter
        > 1  prolate (shell, bomb, pipe)
        = 1  sphere (N = 1/3)
        < 1  oblate (sheet, disc)
    N_axial + 2 * N_transverse = 1.
    """
    a = float(aspect)
    if abs(a - 1.0) < 1e-6:
        return 1 / 3, 1 / 3
    if a > 1.0:  # prolate
        s = np.sqrt(a * a - 1.0)
        n_ax = (a / s * np.log(a + s) - 1.0) / (a * a - 1.0)
    else:  # oblate, m = diameter / thickness > 1
        m = 1.0 / a
        s = np.sqrt(m * m - 1.0)
        n_ax = m * m / (m * m - 1.0) * (1.0 - np.arcsin(s / m) / s)
    n_tr = (1.0 - n_ax) / 2.0
    return float(n_ax), float(n_tr)


def spheroid_volume(length: float, diameter: float) -> float:
    """Volume of a spheroid [m³]."""
    return np.pi / 6.0 * diameter ** 2 * length


def axis_vector(azimuth_deg: float, dip_deg: float) -> np.ndarray:
    """Unit vector of the object axis.

    azimuth: horizontal direction of the axis, 0° = North, 90° = East.
    dip: inclination below horizontal, 0° = lying flat, 90° = vertical.
    """
    az, dip = np.radians(azimuth_deg), np.radians(dip_deg)
    return np.array([np.cos(dip) * np.sin(az),
                     np.cos(dip) * np.cos(az),
                     -np.sin(dip)])


def induced_moment(b0: np.ndarray, length: float, diameter: float,
                   axis: np.ndarray, chi: float = 100.0) -> np.ndarray:
    """Induced dipole moment [A·m²] of a steel spheroid in the field b0 [T].

    Along each principal axis: M_i = chi / (1 + N_i * chi) * H0_i.
    For steel (chi >> 1) shape dominates: M_i ≈ H0_i / N_i,
    so elongated objects acquire a much larger moment along their axis.

    Simplification: hollow bodies (bomb casings) are treated as solid. For high
    permeability and not-too-thin walls this is a reasonable approximation.
    """
    aspect = length / diameter
    n_ax, n_tr = demag_factors(aspect)
    k_ax = chi / (1.0 + n_ax * chi)
    k_tr = chi / (1.0 + n_tr * chi)
    u = axis / np.linalg.norm(axis)
    tensor = k_tr * np.eye(3) + (k_ax - k_tr) * np.outer(u, u)
    vol = spheroid_volume(length, diameter)
    return vol * tensor @ (b0 / MU0)


def dipole_field(obs: np.ndarray, src: np.ndarray, moment: np.ndarray) -> np.ndarray:
    """Magnetic field [T] of a point dipole at observation points.

    obs: (N, 3) observation points, src: (3,) dipole position, moment: (3,) [A·m²]
    Returns: (N, 3) field vectors.
        B = mu0/(4 pi) * (3 (m·r̂) r̂ - m) / r³
    """
    r = np.atleast_2d(obs) - np.asarray(src)[None, :]
    dist = np.linalg.norm(r, axis=1)
    dist = np.maximum(dist, 1e-6)  # avoid singularity
    rhat = r / dist[:, None]
    mdotr = rhat @ moment
    return MU0 / (4 * np.pi) * (3 * mdotr[:, None] * rhat - moment[None, :]) / dist[:, None] ** 3


def total_field(obs: np.ndarray, sources: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Sum of the dipole fields of several sources [(position, moment), ...] -> (N, 3) [T]."""
    b = np.zeros((len(obs), 3))
    for pos, mom in sources:
        b += dipole_field(obs, pos, mom)
    return b


def tmi_anomaly(b_anom: np.ndarray, b0: np.ndarray) -> np.ndarray:
    """Total magnetic intensity anomaly (caesium / Overhauser magnetometers) [T].

    Approximation for |b_anom| << |b0|: ΔT ≈ b_anom · b0_hat.
    """
    return b_anom @ (b0 / np.linalg.norm(b0))
