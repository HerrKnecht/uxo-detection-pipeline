"""UXO vs. clutter classification – planned (phase 4).

Comparison:
    - physics-based: dipole inversion (depth, moment, direction) + rules
      (e.g. does the moment match the depth? is the direction consistent with induced field?)
    - classical ML: random forest / XGBoost on inversion parameters
    - deep learning: CNN directly on image patches around the anomaly
Key point: choose the threshold so that (almost) no UXO is missed;
evaluate via ROC curve and "digs required for 100 % recall".
"""
