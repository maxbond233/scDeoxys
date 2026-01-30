"""
Basic example demonstrating scDeoxys workflow.

This script:
1. Generates synthetic Swiss Roll Simplex data
2. Trains a ParetoVAE model
3. Visualizes the results
4. Computes evaluation metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment

from scdeoxys import ParetoVAE, generate_swiss_roll_simplex, Trainer
from scdeoxys.plotting import plot_training_curves, plot_comparison


def compute_simplex_metrics(z_pred, z_true):
    """
    Compute metrics comparing predicted and true simplex coordinates.

    Args:
        z_pred: Predicted simplex coordinates, shape (n_cells, n_archetypes)
        z_true: True simplex coordinates, shape (n_cells, n_archetypes)

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    n_archetypes = z_true.shape[1]

    # Per-dimension Pearson correlation (without alignment)
    correlations_raw = []
    for k in range(n_archetypes):
        r, _ = pearsonr(z_pred[:, k], z_true[:, k])
        correlations_raw.append(float(r))

    # Raw MSE and simplex distance (without alignment)
    mse_raw = float(np.mean((z_pred - z_true) ** 2))
    simplex_dist_raw = float(np.mean(np.sum(np.abs(z_pred - z_true), axis=1)))

    # Find optimal alignment using Hungarian algorithm
    # cost_matrix[i, j] = -|corr(z_pred[:, i], z_true[:, j])|
    # Matching: z_pred[:, row_ind[k]] corresponds to z_true[:, col_ind[k]]
    cost_matrix = np.zeros((n_archetypes, n_archetypes))
    for i in range(n_archetypes):
        for j in range(n_archetypes):
            r, _ = pearsonr(z_pred[:, i], z_true[:, j])
            cost_matrix[i, j] = -abs(r)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build permutation: perm[j] = i means z_pred[:, i] maps to z_true[:, j]
    perm = np.zeros(n_archetypes, dtype=int)
    for k in range(len(row_ind)):
        perm[col_ind[k]] = row_ind[k]

    # Reorder z_pred to align with z_true
    z_pred_aligned = z_pred[:, perm]

    # Aligned correlations
    correlations_aligned = []
    for k in range(n_archetypes):
        r, _ = pearsonr(z_pred_aligned[:, k], z_true[:, k])
        correlations_aligned.append(float(r))

    # Aligned MSE and simplex distance
    mse_aligned = float(np.mean((z_pred_aligned - z_true) ** 2))
    simplex_dist_aligned = float(np.mean(np.sum(np.abs(z_pred_aligned - z_true), axis=1)))

    return {
        "correlations_raw": correlations_raw,
        "correlations_aligned": correlations_aligned,
        "mean_correlation_raw": float(np.mean(correlations_raw)),
        "mean_correlation_aligned": float(np.mean(correlations_aligned)),
        "mse_raw": mse_raw,
        "mse_aligned": mse_aligned,
        "simplex_distance_raw": simplex_dist_raw,
        "simplex_distance_aligned": simplex_dist_aligned,
        "alignment_perm": perm.tolist(),
    }


def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("scDeoxys MVP - Basic Example")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic Swiss Roll Simplex data...")
    adata = generate_swiss_roll_simplex(
        n_cells=1000,
        n_genes=500,
        n_archetypes=3,
        alpha=1.0,
        curvature=1.0,
        theta=10.0,
        seed=42,
    )

    x_train = adata.layers["counts"]
    z_true = adata.obsm["X_true_simplex"]

    print(f"   - Generated {x_train.shape[0]} cells with {x_train.shape[1]} genes")
    print(f"   - Number of archetypes: {z_true.shape[1]}")
    print(f"   - Mean count per cell: {x_train.mean():.2f}")

    # Create and train model
    print("\n2. Creating ParetoVAE model...")
    model = ParetoVAE(
        n_genes=x_train.shape[1],
        n_archetypes=3,
        hidden_dims=[512, 256, 128],
    )

    print(f"   - Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("\n3. Training model...")
    trainer = Trainer(
        model,
        learning_rate=1e-3,
        beta_warmup_epochs=100,
        beta_min=0.0,
        beta_max=1.0,
        free_bits=0.1,
    )

    history = trainer.train(
        adata,
        n_epochs=250,
        batch_size=128,
        verbose=True,
    )

    print("\n   Training complete!")
    print(f"   - Final total loss: {history['total_loss'][-1]:.4f}")
    print(f"   - Final recon loss: {history['recon_loss'][-1]:.4f}")
    print(f"   - Final KL loss: {history['kl_loss'][-1]:.4f}")

    # Encode data to simplex coordinates
    print("\n4. Encoding data to simplex space...")
    trainer.encode(adata)
    z_scdeoxys = adata.obsm["X_scdeoxys"]
    print(f"   - Encoded shape: {z_scdeoxys.shape}")

    # Compute evaluation metrics
    print("\n5. Computing evaluation metrics...")
    metrics = compute_simplex_metrics(z_scdeoxys, z_true)

    # Add training metrics
    metrics["final_total_loss"] = history["total_loss"][-1]
    metrics["final_recon_loss"] = history["recon_loss"][-1]
    metrics["final_kl_loss"] = history["kl_loss"][-1]
    adata.uns.setdefault("scdeoxys", {})["metrics"] = metrics

    # Save metrics to file
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("   - Saved metrics.json")

    # Print summary
    print(f"\n   Raw (no alignment):")
    print(f"     - Correlations: {[f'{r:.3f}' for r in metrics['correlations_raw']]}")
    print(f"     - Mean correlation: {metrics['mean_correlation_raw']:.3f}")
    print(f"     - MSE: {metrics['mse_raw']:.4f}")
    print(f"     - Simplex distance (L1): {metrics['simplex_distance_raw']:.4f}")
    print(f"\n   Aligned (optimal permutation {metrics['alignment_perm']}):")
    print(f"     - Correlations: {[f'{r:.3f}' for r in metrics['correlations_aligned']]}")
    print(f"     - Mean correlation: {metrics['mean_correlation_aligned']:.3f}")
    print(f"     - MSE: {metrics['mse_aligned']:.4f}")
    print(f"     - Simplex distance (L1): {metrics['simplex_distance_aligned']:.4f}")

    # Visualize results
    print("\n6. Generating visualizations...")

    # Training curves
    fig1 = plot_training_curves(adata)
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    print("   - Saved training_curves.png")

    # Comparison plot
    fig2 = plot_comparison(adata)
    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    print("   - Saved comparison.png")

    print("\n" + "=" * 60)
    print("Example complete! Check the generated PNG files.")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    main()
