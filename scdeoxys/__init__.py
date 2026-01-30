"""
scDeoxys: Single-cell Deconvolution using Pareto-optimal Archetypal Analysis

A deep learning framework for mapping curved gene expression manifolds
to interpretable simplex spaces using variational autoencoders.
"""

__version__ = "0.1.0"

from scdeoxys.models.vae import ParetoVAE
from scdeoxys.data.synthetic import generate_swiss_roll_simplex
from scdeoxys.train.trainer import Trainer

__all__ = [
    "ParetoVAE",
    "generate_swiss_roll_simplex",
    "Trainer",
]
