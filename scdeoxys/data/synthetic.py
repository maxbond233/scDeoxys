"""
Synthetic data generation for scDeoxys.

Generates Swiss Roll Simplex data: a curved manifold in gene expression space
that corresponds to a regular simplex structure in latent space.
"""

import numpy as np
import torch
import anndata as ad
from scdeoxys.models.distributions import NegativeBinomial


def generate_archetypal_profiles(n_genes, n_archetypes, seed=None):
    """
    Generate archetypal gene expression profiles.

    Each archetype represents a distinct cell state with characteristic
    gene expression patterns.

    Args:
        n_genes: Number of genes
        n_archetypes: Number of archetypal states
        seed: Random seed for reproducibility

    Returns:
        archetypes: Gene expression profiles, shape (n_archetypes, n_genes)
    """
    if seed is not None:
        np.random.seed(seed)

    archetypes = []
    genes_per_archetype = n_genes // n_archetypes
    remainder = n_genes % n_archetypes

    gene_idx = 0
    for k in range(n_archetypes):
        profile = np.zeros(n_genes)

        # Distribute remainder genes round-robin to first archetypes
        n_marker = genes_per_archetype + (1 if k < remainder else 0)
        start_idx = gene_idx
        end_idx = gene_idx + n_marker

        # High expression genes for this archetype
        profile[start_idx:end_idx] = np.random.lognormal(
            mean=3.0, sigma=0.5, size=n_marker
        )

        # Low baseline expression for other genes
        other_indices = list(range(0, start_idx)) + list(range(end_idx, n_genes))
        profile[other_indices] = np.random.lognormal(
            mean=0.5, sigma=0.3, size=len(other_indices)
        )

        archetypes.append(profile)
        gene_idx = end_idx

    return np.array(archetypes)


def generate_expression_from_swiss_roll(swiss_roll_3d, archetypes, z_true, curvature=1.0):
    """
    Generate gene expression from Swiss Roll 3D coordinates.

    This creates a complex mapping from 3D Swiss Roll to high-dimensional
    gene expression space, while preserving the underlying simplex structure.

    Args:
        swiss_roll_3d: Swiss Roll coordinates, shape (n_cells, 3)
        archetypes: Archetypal profiles, shape (n_archetypes, n_genes)
        z_true: True simplex coordinates, shape (n_cells, n_archetypes)
        curvature: Strength of nonlinear effects

    Returns:
        mu: Mean gene expression, shape (n_cells, n_genes)
    """
    n_cells = swiss_roll_3d.shape[0]
    n_genes = archetypes.shape[1]

    # Base expression from simplex coordinates (preserves simplex structure)
    base_expression = z_true @ archetypes

    # Add complex nonlinear perturbations based on Swiss Roll coordinates
    perturbation = np.zeros((n_cells, n_genes))

    # Use Swiss Roll coordinates to create complex patterns
    x, y, z = swiss_roll_3d[:, 0], swiss_roll_3d[:, 1], swiss_roll_3d[:, 2]

    # Normalize coordinates for stability
    x_norm = (x - x.mean()) / (x.std() + 1e-8)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    z_norm = (z - z.mean()) / (z.std() + 1e-8)

    # Create nonlinear features from Swiss Roll coordinates
    for i in range(n_genes):
        # Different genes respond to different combinations of coordinates
        phase_x = 2 * np.pi * (i / n_genes)
        phase_y = 2 * np.pi * ((i + n_genes // 3) / n_genes)
        phase_z = 2 * np.pi * ((i + 2 * n_genes // 3) / n_genes)

        # Complex nonlinear combination
        perturbation[:, i] = curvature * (
            np.sin(x_norm + phase_x) * np.cos(y_norm + phase_y) +
            np.cos(y_norm + phase_y) * np.sin(z_norm + phase_z) +
            0.5 * np.sin(x_norm * y_norm + phase_x) +
            0.3 * np.cos(y_norm * z_norm + phase_z)
        ) * base_expression[:, i]

    mu = base_expression + perturbation
    mu = np.maximum(mu, 0.1)  # Ensure positive values

    return mu


def generate_swiss_roll_simplex(
    n_cells=1000,
    n_genes=500,
    n_archetypes=3,
    alpha=1.0,
    curvature=1.0,
    theta=10.0,
    seed=None,
):
    """
    Generate Swiss Roll Simplex synthetic data.

    Creates a dataset where cells lie on a curved manifold in gene expression
    space, but correspond to a regular simplex structure in latent space.

    Args:
        n_cells: Number of cells to generate
        n_genes: Number of genes
        n_archetypes: Number of archetypal states (K)
        alpha: Dirichlet concentration parameter (uniform if scalar)
        curvature: Strength of nonlinear effects (0 = linear, higher = more curved)
        theta: Dispersion parameter for NB noise
        seed: Random seed for reproducibility

    Returns:
        adata: AnnData object containing:
            - layers["counts"]: Gene expression counts, shape (n_cells, n_genes)
            - obsm["X_true_simplex"]: True simplex coordinates, shape (n_cells, n_archetypes)
            - obsm["X_swiss_roll_3d"]: 3D Swiss Roll coordinates, shape (n_cells, 3)
            - obs["color_param"]: Color parameter for plotting, shape (n_cells,)
            - uns["scdeoxys_synthetic"]: Generation metadata

    Note:
        The Swiss Roll geometry has 2 intrinsic dimensions, which faithfully
        encodes K=3 archetypes (a 2D simplex). For K>3, the base expression
        z_true @ archetypes still uses all K coordinates, but the 3D manifold
        structure only reflects the first coordinate (angle) and the mean of
        remaining coordinates (height). Manifold-preservation metrics (kNN,
        trustworthiness, continuity) will be unreliable for K>3.
    """
    if n_archetypes < 2:
        raise ValueError(
            f"n_archetypes must be >= 2, got {n_archetypes}. "
            "Archetypal analysis requires at least 2 archetypes."
        )

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Generate archetypal gene expression profiles
    archetypes = generate_archetypal_profiles(n_genes, n_archetypes, seed)

    # Sample simplex coordinates from Dirichlet distribution
    if np.isscalar(alpha):
        alpha = np.ones(n_archetypes) * alpha

    z_true = np.random.dirichlet(alpha, size=n_cells)

    # Generate 3D Swiss Roll coordinates from simplex coordinates
    # Map simplex to a parameter t in [0, 1]
    t = z_true[:, 0]  # Use first simplex coordinate as the parameter

    # Create Swiss Roll in 3D
    # Standard Swiss Roll parametrization
    angle = 1.5 * np.pi * (1 + 2 * t)  # Angle increases with t
    x = angle * np.cos(angle)
    y = angle * np.sin(angle)
    z = 10 * z_true[:, 1:].mean(axis=1)  # Use all remaining simplex coords for height

    swiss_roll_3d = np.column_stack([x, y, z])

    # Generate gene expression from Swiss Roll coordinates
    # This creates complex high-dimensional data that encodes the Swiss Roll structure
    mu = generate_expression_from_swiss_roll(swiss_roll_3d, archetypes, z_true, curvature)

    # Sample counts from Negative Binomial distribution
    mu_torch = torch.tensor(mu, dtype=torch.float32)
    theta_torch = torch.tensor(theta, dtype=torch.float32)

    counts = NegativeBinomial.sample(mu_torch, theta_torch).numpy()

    adata = ad.AnnData(X=counts)
    adata.layers["counts"] = counts
    adata.obsm["X_true_simplex"] = z_true
    adata.obsm["X_swiss_roll_3d"] = swiss_roll_3d
    adata.obs["color_param"] = t
    adata.uns["scdeoxys_synthetic"] = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "n_archetypes": n_archetypes,
        "alpha": alpha.tolist() if hasattr(alpha, "tolist") else alpha,
        "curvature": curvature,
        "theta": theta,
        "seed": seed,
        "archetypes": archetypes,
    }

    return adata
