"""Anomaly detection (target picking) – planned (phase 3).

Approaches to compare:
    - threshold + local maxima (classical baseline)
    - clustering (e.g. DBSCAN / hierarchical) on filtered data
    - CNN segmentation (U-Net) on gridded images
Evaluation: detection rate (recall!) against simulation ground truth,
false alarms per hectare, position accuracy.
"""
