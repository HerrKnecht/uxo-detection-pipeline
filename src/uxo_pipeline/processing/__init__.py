"""Preprocessing – planned (phase 2).

Intended functions:
    - destripe(): remove line offsets and heading error (median per line)
    - highpass(): separate geological background
    - analytic_signal(): amplitude independent of magnetisation direction
    - grid(): interpolate irregular readings onto a grid
For borehole data: drift correction, smoothing, matching of neighbouring boreholes.
"""
