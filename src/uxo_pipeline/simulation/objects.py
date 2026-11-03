"""Object catalogue: UXO and clutter (scrap) with randomised parameters.

NOTE: Dimensions are rounded indicative values for simulation (body without
tail fins). They should be checked against technical literature / ordnance
data sheets before serious use.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import physics


@dataclass(frozen=True)
class ObjectType:
    name: str
    category: str                   # "uxo" or "clutter"
    label: str                      # ML class, e.g. "bomb", "shell", "small", "clutter"
    diameter: tuple[float, float]   # [m] min, max
    aspect: tuple[float, float]     # length / diameter, min, max
    depth: tuple[float, float]      # depth of centre below ground surface [m]
    dip: tuple[float, float]        # inclination below horizontal [°]
    remanence: tuple[float, float]  # |m_rem| / |m_ind|
    chi: tuple[float, float] = (50.0, 300.0)


CATALOG: dict[str, ObjectType] = {t.name: t for t in [
    # --- Aircraft bombs ---
    ObjectType("bomb_1000lb", "uxo", "bomb", (0.46, 0.48), (3.2, 3.6), (2.0, 7.0), (20, 90), (0.0, 0.4)),
    ObjectType("bomb_500lb",  "uxo", "bomb", (0.35, 0.37), (3.1, 3.5), (1.5, 6.0), (20, 90), (0.0, 0.4)),
    ObjectType("bomb_250lb",  "uxo", "bomb", (0.27, 0.29), (3.0, 3.4), (1.0, 5.0), (20, 90), (0.0, 0.4)),
    # --- Artillery and mortar ---
    ObjectType("shell_150mm", "uxo", "shell", (0.149, 0.151), (3.4, 3.8), (0.2, 2.0), (0, 90), (0.0, 0.5)),
    ObjectType("shell_105mm", "uxo", "shell", (0.104, 0.106), (3.6, 4.0), (0.2, 1.5), (0, 90), (0.0, 0.5)),
    ObjectType("shell_75mm",  "uxo", "shell", (0.074, 0.076), (3.6, 4.2), (0.1, 1.2), (0, 90), (0.0, 0.5)),
    ObjectType("mortar_81mm", "uxo", "shell", (0.080, 0.082), (3.5, 4.2), (0.1, 1.0), (30, 90), (0.0, 0.5)),
    # --- Small calibre ---
    ObjectType("flak_20mm",   "uxo", "small", (0.019, 0.021), (3.5, 4.5), (0.0, 0.6), (0, 90), (0.0, 0.6)),
    # --- Clutter ---
    ObjectType("fragment",    "clutter", "clutter", (0.01, 0.06), (0.2, 3.0), (0.0, 0.5), (0, 90), (0.2, 2.0)),
    ObjectType("scrap_small", "clutter", "clutter", (0.03, 0.12), (0.1, 5.0), (0.0, 0.8), (0, 90), (0.1, 1.5)),
    ObjectType("scrap_large", "clutter", "clutter", (0.10, 0.30), (0.05, 3.0), (0.0, 1.5), (0, 90), (0.1, 1.0)),
    ObjectType("pipe_rebar",  "clutter", "clutter", (0.01, 0.04), (15.0, 80.0), (0.3, 1.5), (0, 15), (0.0, 0.8)),
]}


@dataclass
class Target:
    """A concrete, randomly generated buried object (= ground truth)."""
    target_id: int
    type_name: str
    category: str
    label: str
    x: float
    y: float
    depth: float           # depth of centre [m], positive downward
    length: float
    diameter: float
    azimuth: float
    dip: float
    chi: float
    remanence_ratio: float
    moment: np.ndarray = field(repr=False)   # total moment [A·m²]

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, -self.depth])

    @property
    def is_uxo(self) -> bool:
        return self.category == "uxo"

    @property
    def radius(self) -> float:
        """Radius of an enclosing sphere [m] (used for collision checks)."""
        return 0.5 * max(self.length, self.diameter)

    def to_record(self) -> dict:
        d = asdict(self)
        m = d.pop("moment")
        d.update(mx=m[0], my=m[1], mz=m[2], moment_abs=float(np.linalg.norm(m)), is_uxo=self.is_uxo)
        return d


def sample_target(rng: np.random.Generator, type_name: str, x: float, y: float,
                  b0: np.ndarray, target_id: int = 0) -> Target:
    """Create a random object of the given type at position (x, y)."""
    t = CATALOG[type_name]
    u = lambda lo_hi: rng.uniform(*lo_hi)
    diameter = u(t.diameter)
    aspect = u(t.aspect) if t.aspect[0] >= 1 or t.aspect[1] <= 1 else _log_uniform(rng, *t.aspect)
    length = diameter * aspect
    depth = u(t.depth)
    azimuth = u((0, 360))
    dip = u(t.dip)
    chi = u(t.chi)
    axis = physics.axis_vector(azimuth, dip)
    # The object must not stick out of the ground: half vertical extent of the spheroid
    uz = axis[2]
    half_z = np.sqrt((0.5 * length * uz) ** 2 + (0.5 * diameter) ** 2 * (1 - uz ** 2))
    depth = max(depth, half_z + 0.02)
    m_ind = physics.induced_moment(b0, length, diameter, axis, chi)
    rem_ratio = u(t.remanence)
    rem_dir = rng.normal(size=3)
    rem_dir /= np.linalg.norm(rem_dir)
    m_rem = rem_ratio * np.linalg.norm(m_ind) * rem_dir
    return Target(target_id, type_name, t.category, t.label, float(x), float(y), float(depth),
                  float(length), float(diameter), float(azimuth), float(dip), float(chi),
                  float(rem_ratio), m_ind + m_rem)


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Log-uniform sample, so sheets (<1) and rods (>1) are equally likely."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


@dataclass
class Scene:
    """Survey area with all buried objects."""
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    b0: np.ndarray
    targets: list[Target]

    def sources(self):
        return [(t.position, t.moment) for t in self.targets]


DEFAULT_UXO_MIX = {"bomb_1000lb": 0.05, "bomb_500lb": 0.15, "bomb_250lb": 0.15,
                   "shell_150mm": 0.1, "shell_105mm": 0.15, "shell_75mm": 0.15,
                   "mortar_81mm": 0.1, "flak_20mm": 0.15}
DEFAULT_CLUTTER_MIX = {"fragment": 0.35, "scrap_small": 0.35, "scrap_large": 0.2, "pipe_rebar": 0.1}


def random_scene(rng: np.random.Generator,
                 x_range=(0.0, 20.0), y_range=(0.0, 20.0),
                 n_uxo: int | tuple[int, int] = (0, 3),
                 n_clutter: int | tuple[int, int] = (3, 15),
                 uxo_mix: dict | None = None, clutter_mix: dict | None = None,
                 b0: np.ndarray | None = None, margin: float = 1.0) -> Scene:
    """Random survey area with UXO and clutter.

    margin: distance of objects from the area edge [m], so anomalies are not cut off.
    """
    b0 = physics.earth_field() if b0 is None else b0
    uxo_mix = uxo_mix or DEFAULT_UXO_MIX
    clutter_mix = clutter_mix or DEFAULT_CLUTTER_MIX
    n_u = n_uxo if isinstance(n_uxo, int) else int(rng.integers(n_uxo[0], n_uxo[1] + 1))
    n_c = n_clutter if isinstance(n_clutter, int) else int(rng.integers(n_clutter[0], n_clutter[1] + 1))

    names = list(rng.choice(list(uxo_mix), size=n_u, p=_norm(uxo_mix))) + \
            list(rng.choice(list(clutter_mix), size=n_c, p=_norm(clutter_mix)))
    targets = []
    for i, name in enumerate(names):
        x = rng.uniform(x_range[0] + margin, x_range[1] - margin)
        y = rng.uniform(y_range[0] + margin, y_range[1] - margin)
        targets.append(sample_target(rng, str(name), x, y, b0, target_id=i))
    return Scene(tuple(x_range), tuple(y_range), b0, targets)


def _norm(mix: dict) -> np.ndarray:
    p = np.array(list(mix.values()), dtype=float)
    return p / p.sum()
