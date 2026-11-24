"""Speaker-free floor: measuring how much LLM conformity survives speaker removal.

Modules:
    perturbations  build the inserted Round-2 pressure text (framings, baselines, dose, role).
    arbitration    deterministic log-probability option read and Round-1/Round-2 messages.
    metrics        harmful/beneficial revision rates, probability shift, Wilson intervals.
"""

__author__ = "Yibo Hu"
