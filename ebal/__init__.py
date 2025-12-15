"""
Entropy Balancing for Causal Inference

This package provides implementations of entropy balancing for:
- Binary treatment effects (ATT, ATC, ATE)
- Continuous treatment effects

References:
    - Hainmueller, J. (2012). Entropy balancing for causal effects.
    - Vegetabile et al. (2021). Nonparametric estimation of population
      average dose-response curves.
    - Tübbicke, S. (2020). Entropy balancing for continuous treatments.
"""

from ebal.binary import ebal_bin
from ebal.continuous import ebal_con

__version__ = "1.0.0"
__author__ = "Eddie Yang"
__all__ = ["ebal_bin", "ebal_con"]
