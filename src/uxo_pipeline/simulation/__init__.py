"""Synthetic magnetometry data for UXO surveys (surface and borehole)."""
from .physics import earth_field, dipole_field, induced_moment, demag_factors, axis_vector
from .objects import CATALOG, Target, Scene, random_scene, sample_target
from .noise import SurfaceNoise, BoreholeNoise
from .surface import SurfaceSurvey, SurfaceResult, simulate_surface
from .borehole import BoreholeSurvey, BoreholeResult, simulate_boreholes, borehole_grid
