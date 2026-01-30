"""
Visualization utilities for scDeoxys.

Implements simplex projection plots and training curve visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
from scipy import sparse
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def plot_simplex_projection(z, labels=None, title="Simplex Projection", ax=None, cmap=None, show_colorbar=True):
    """
    Plot simplex coordinates in 2D.

    For K=3 (triangle), uses ternary coordinates.
    For K>3, uses first two dimensions.

    Args:
        z: Simplex coordinates, shape (n_cells, n_archetypes)
        labels: Optional labels/colors for coloring points
        title: Plot title
        ax: Matplotlib axis (creates new if None)
        cmap: Colormap to use (default: 'tab10' for discrete, or custom for continuous)
        show_colorbar: Whether to show colorbar (default True)

    Returns:
        ax: Matplotlib axis with plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    n_archetypes = z.shape[1]

    if n_archetypes == 3:
        # Ternary plot for K=3
        # Convert barycentric coordinates to 2D Cartesian
        x = z[:, 1] + 0.5 * z[:, 2]
        y = (np.sqrt(3) / 2) * z[:, 2]

        # Draw triangle
        triangle = Polygon(
            [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]],
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(triangle)

        # Plot points
        if labels is not None:
            use_cmap = cmap if cmap is not None else "tab10"
            scatter = ax.scatter(x, y, c=labels, cmap=use_cmap, alpha=0.6, s=20)
            if show_colorbar:
                plt.colorbar(scatter, ax=ax, label="Value")
        else:
            ax.scatter(x, y, alpha=0.6, s=20, color="blue")

        # Label vertices
        ax.text(-0.05, -0.05, "A1", fontsize=12, fontweight="bold")
        ax.text(1.05, -0.05, "A2", fontsize=12, fontweight="bold")
        ax.text(0.5, np.sqrt(3) / 2 + 0.05, "A3", fontsize=12, fontweight="bold")

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, np.sqrt(3) / 2 + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")

    else:
        # For K>3, plot first two dimensions
        if labels is not None:
            use_cmap = cmap if cmap is not None else "tab10"
            scatter = ax.scatter(
                z[:, 0], z[:, 1], c=labels, cmap=use_cmap, alpha=0.6, s=20
            )
            if show_colorbar:
                plt.colorbar(scatter, ax=ax, label="Value")
        else:
            ax.scatter(z[:, 0], z[:, 1], alpha=0.6, s=20, color="blue")

        ax.set_xlabel("Archetype 1")
        ax.set_ylabel("Archetype 2")

    ax.set_title(title)

    return ax


def plot_training_curves(history_or_adata, figsize=(12, 4)):
    """
    Plot training curves showing loss components over epochs.

    Args:
        history_or_adata: History dict or AnnData with uns["scdeoxys"]["history"]
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    if isinstance(history_or_adata, ad.AnnData):
        history = history_or_adata.uns.get("scdeoxys", {}).get("history")
        if history is None:
            raise KeyError("AnnData.uns['scdeoxys']['history'] not found.")
    else:
        history = history_or_adata

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(len(history["total_loss"]))

    # Total loss
    axes[0].plot(epochs, history["total_loss"], linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Total Loss (ELBO)")
    axes[0].grid(True, alpha=0.3)

    # Reconstruction and KL loss (with log scale for KL if values are small)
    ax1 = axes[1]
    ax1.plot(epochs, history["recon_loss"], label="Reconstruction", linewidth=2, color="C0")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reconstruction Loss", color="C0")
    ax1.tick_params(axis='y', labelcolor="C0")
    ax1.set_title("Loss Components")
    ax1.grid(True, alpha=0.3)

    # Secondary y-axis for KL with log scale
    ax2 = ax1.twinx()
    kl_values = np.array(history["kl_loss"])
    ax2.plot(epochs, kl_values, label="KL Divergence", linewidth=2, color="C1")
    ax2.set_ylabel("KL Loss (log scale)", color="C1")
    ax2.tick_params(axis='y', labelcolor="C1")
    # Use log scale if KL values span multiple orders of magnitude or are very small
    if kl_values.max() > 0 and kl_values.min() > 0:
        ax2.set_yscale('log')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Beta warmup
    axes[2].plot(epochs, history["beta"], linewidth=2, color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Beta")
    axes[2].set_title("Beta Warmup Schedule")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_comparison(adata, z_scdeoxys=None, z_true=None, swiss_roll_3d=None, colors=None, figsize=(20, 5)):
    """
    Compare 3D Swiss Roll, PCA projection, and scDeoxys simplex projection.

    Args:
        adata: AnnData with layers["counts"] and obsm/obs metadata
        z_scdeoxys: scDeoxys simplex coordinates override (optional)
        z_true: True simplex coordinates override (optional)
        swiss_roll_3d: Swiss Roll coordinates override (optional)
        colors: Color values override (optional)
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("plot_comparison expects AnnData input.")

    if "counts" not in adata.layers:
        raise KeyError("AnnData.layers['counts'] is required for PCA projection.")

    x = adata.layers["counts"]
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x)

    if z_scdeoxys is None:
        z_scdeoxys = adata.obsm.get("X_scdeoxys")
    if z_true is None:
        z_true = adata.obsm.get("X_true_simplex")
    if swiss_roll_3d is None:
        swiss_roll_3d = adata.obsm.get("X_swiss_roll_3d")
    if colors is None and "color_param" in adata.obs:
        colors = adata.obs["color_param"].to_numpy()

    if z_scdeoxys is None:
        raise KeyError("scDeoxys projection not found. Expected obsm['X_scdeoxys'] or z_scdeoxys override.")

    z_scdeoxys = np.asarray(z_scdeoxys)
    if z_true is not None:
        z_true = np.asarray(z_true)
    if swiss_roll_3d is not None:
        swiss_roll_3d = np.asarray(swiss_roll_3d)
    # Determine number of subplots
    n_plots = 2  # PCA + scDeoxys
    if swiss_roll_3d is not None:
        n_plots += 1  # Add 3D Swiss Roll
    if z_true is not None:
        n_plots += 1  # Add true simplex

    fig = plt.figure(figsize=figsize)
    plot_idx = 1

    # Use rainbow colormap if colors provided
    cmap = 'rainbow' if colors is not None else None

    # 3D Swiss Roll (if available)
    if swiss_roll_3d is not None:
        ax = fig.add_subplot(1, n_plots, plot_idx, projection='3d')
        if colors is not None:
            scatter = ax.scatter(swiss_roll_3d[:, 0], swiss_roll_3d[:, 1], swiss_roll_3d[:, 2],
                                alpha=0.6, s=20, c=colors, cmap=cmap)
        else:
            scatter = ax.scatter(swiss_roll_3d[:, 0], swiss_roll_3d[:, 1], swiss_roll_3d[:, 2],
                                alpha=0.6, s=20, c=swiss_roll_3d[:, 2], cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Swiss Roll')
        plot_idx += 1

    # PCA projection
    ax = fig.add_subplot(1, n_plots, plot_idx)
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)
    if colors is not None:
        ax.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.6, s=20, c=colors, cmap=cmap)
    else:
        ax.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.6, s=20)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Projection")
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # scDeoxys projection
    ax = fig.add_subplot(1, n_plots, plot_idx)
    plot_simplex_projection(z_scdeoxys, labels=colors, title="scDeoxys Projection", ax=ax, cmap=cmap, show_colorbar=False)
    plot_idx += 1

    # True simplex (if available)
    if z_true is not None:
        ax = fig.add_subplot(1, n_plots, plot_idx)
        plot_simplex_projection(z_true, labels=colors, title="True Simplex", ax=ax, cmap=cmap, show_colorbar=True)

    plt.tight_layout()

    return fig
