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


def _is_discrete_labels(labels):
    """Check if labels are discrete (string/categorical) rather than numeric."""
    arr = np.asarray(labels)
    if arr.dtype.kind in ('U', 'S', 'O'):
        return True
    try:
        import pandas as pd
        if hasattr(labels, 'dtype') and isinstance(labels.dtype, pd.CategoricalDtype):
            return True
    except ImportError:
        pass
    return False


def plot_simplex_projection(z, labels=None, title="Simplex Projection", ax=None, cmap=None, show_colorbar=True):
    """
    Plot simplex coordinates in 2D.

    For K=3 (triangle), uses ternary coordinates.
    For K>3, uses Circular Projection (RadViz-style) where archetypes are
    arranged evenly on a unit circle and cells are projected as weighted
    averages of archetype positions.

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
            if _is_discrete_labels(labels):
                unique_labels = np.unique(labels)
                colors = plt.cm.get_cmap(use_cmap)(np.linspace(0, 1, len(unique_labels)))
                for i, label in enumerate(unique_labels):
                    mask = np.asarray(labels) == label
                    ax.scatter(x[mask], y[mask], c=[colors[i]], label=str(label), alpha=0.6, s=20)
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            else:
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
        # For K>3, use Circular Projection (RadViz-style)
        # Arrange archetypes evenly on a unit circle
        angles = np.linspace(0, 2 * np.pi, n_archetypes, endpoint=False) + np.pi / 2
        archetype_coords = np.column_stack([np.cos(angles), np.sin(angles)])

        # Project cells as weighted average of archetype positions
        projected_coords = z @ archetype_coords
        x, y = projected_coords[:, 0], projected_coords[:, 1]

        # Draw background unit circle
        circle_theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(circle_theta), np.sin(circle_theta),
                color='gray', linewidth=1.5, linestyle='-', alpha=0.5)

        # Draw axes from center to each archetype
        for i in range(n_archetypes):
            ax.plot([0, archetype_coords[i, 0]], [0, archetype_coords[i, 1]],
                    color='gray', linewidth=1, linestyle='--', alpha=0.5)

        # Plot archetype positions on the circle
        ax.scatter(archetype_coords[:, 0], archetype_coords[:, 1],
                   color='black', s=80, zorder=5, marker='o')

        # Label archetypes
        for i in range(n_archetypes):
            # Position labels slightly outside the circle
            label_x = archetype_coords[i, 0] * 1.15
            label_y = archetype_coords[i, 1] * 1.15
            ax.text(label_x, label_y, f"A{i+1}", fontsize=11, fontweight="bold",
                    ha='center', va='center')

        # Plot cell points
        if labels is not None:
            use_cmap = cmap if cmap is not None else "tab10"
            if _is_discrete_labels(labels):
                unique_labels = np.unique(labels)
                colors = plt.cm.get_cmap(use_cmap)(np.linspace(0, 1, len(unique_labels)))
                for i, label in enumerate(unique_labels):
                    mask = np.asarray(labels) == label
                    ax.scatter(x[mask], y[mask], c=[colors[i]], label=str(label), alpha=0.6, s=20, zorder=3)
                ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            else:
                scatter = ax.scatter(x, y, c=labels, cmap=use_cmap, alpha=0.6, s=20, zorder=3)
                if show_colorbar:
                    plt.colorbar(scatter, ax=ax, label="Value")
        else:
            ax.scatter(x, y, alpha=0.6, s=20, color="blue", zorder=3)

        # Set axis properties
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")

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


def plot_umap_weights(
    z,
    labels=None,
    color_by_dominant=True,
    n_neighbors=15,
    min_dist=0.1,
    title="UMAP of Archetype Weights",
    ax=None,
    cmap=None,
    show_colorbar=True,
    show_legend=True,
    figsize=(8, 7),
):
    """
    Plot UMAP projection of archetype weights for K>3 visualization.

    This is the recommended visualization for high-dimensional simplex (K>3).
    Cells near vertices represent pure archetype states, while cells in the
    middle represent transitional states.

    Args:
        z: Simplex coordinates, shape (n_cells, n_archetypes)
        labels: Optional labels for coloring (overrides color_by_dominant)
        color_by_dominant: If True and labels is None, color by dominant archetype
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        title: Plot title
        ax: Matplotlib axis (creates new if None)
        cmap: Colormap to use
        show_colorbar: Whether to show colorbar (for continuous labels)
        show_legend: Whether to show legend (for discrete labels)
        figsize: Figure size if creating new figure

    Returns:
        fig: Matplotlib figure
        umap_coords: UMAP coordinates (n_cells, 2)
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError(
            "umap-learn is required for plot_umap_weights. "
            "Install with: pip install umap-learn"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Run UMAP on weight matrix
    reducer = UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42
    )
    umap_coords = reducer.fit_transform(z)

    n_archetypes = z.shape[1]

    # Determine coloring
    if labels is not None:
        use_labels = labels
        is_discrete = (
            isinstance(labels[0], (str, np.str_)) or len(np.unique(labels)) <= 20
        )
    elif color_by_dominant:
        use_labels = np.argmax(z, axis=1)
        is_discrete = True
    else:
        use_labels = None
        is_discrete = False

    # Plot
    if use_labels is not None:
        if is_discrete:
            unique_labels = np.unique(use_labels)
            use_cmap = cmap if cmap is not None else "tab10"
            colors = plt.cm.get_cmap(use_cmap)(np.linspace(0, 1, len(unique_labels)))

            for i, label in enumerate(unique_labels):
                mask = use_labels == label
                if color_by_dominant and labels is None:
                    label_name = f"A{label + 1}"
                else:
                    label_name = str(label)
                ax.scatter(
                    umap_coords[mask, 0],
                    umap_coords[mask, 1],
                    c=[colors[i]],
                    label=label_name,
                    alpha=0.6,
                    s=20,
                )

            if show_legend:
                ax.legend(
                    title="Archetype" if color_by_dominant else "Label",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                )
        else:
            use_cmap = cmap if cmap is not None else "viridis"
            scatter = ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=use_labels,
                cmap=use_cmap,
                alpha=0.6,
                s=20,
            )
            if show_colorbar:
                plt.colorbar(scatter, ax=ax, label="Value")
    else:
        ax.scatter(umap_coords[:, 0], umap_coords[:, 1], alpha=0.6, s=20, color="blue")

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig, umap_coords


def plot_archetype_heatmap(
    z,
    cell_labels=None,
    archetype_names=None,
    cluster_cells=True,
    cluster_archetypes=False,
    n_cell_clusters=None,
    show_dendrogram=True,
    cmap="YlOrRd",
    figsize=(10, 8),
    title="Archetype Weights Heatmap",
):
    """
    Plot heatmap of archetype weights with optional clustering dendrogram.

    This is the standard visualization for single-cell analysis papers,
    showing the distribution of archetype weights across cells.

    Args:
        z: Simplex coordinates, shape (n_cells, n_archetypes)
        cell_labels: Optional cell type/cluster labels for row annotation
        archetype_names: Optional names for archetypes (default: A1, A2, ...)
        cluster_cells: Whether to cluster cells (rows)
        cluster_archetypes: Whether to cluster archetypes (columns)
        n_cell_clusters: Number of cell clusters for color bar (auto if None)
        show_dendrogram: Whether to show dendrogram
        cmap: Colormap for heatmap
        figsize: Figure size
        title: Plot title

    Returns:
        fig: Matplotlib figure
        cell_order: Order of cells after clustering
    """
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist

    n_cells, n_archetypes = z.shape

    # Default archetype names
    if archetype_names is None:
        archetype_names = [f"A{i+1}" for i in range(n_archetypes)]

    # Cluster cells if requested
    if cluster_cells and n_cells > 1:
        cell_linkage = linkage(z, method="ward")
        cell_order = dendrogram(cell_linkage, no_plot=True)["leaves"]
    else:
        cell_linkage = None
        cell_order = np.arange(n_cells)

    # Cluster archetypes if requested
    if cluster_archetypes and n_archetypes > 1:
        arch_linkage = linkage(z.T, method="ward")
        arch_order = dendrogram(arch_linkage, no_plot=True)["leaves"]
    else:
        arch_linkage = None
        arch_order = np.arange(n_archetypes)

    # Reorder data
    z_ordered = z[cell_order][:, arch_order]
    archetype_names_ordered = [archetype_names[i] for i in arch_order]

    # Determine cell clusters for annotation
    if cell_labels is not None:
        cell_labels_ordered = np.array(cell_labels)[cell_order]
    elif cluster_cells and cell_linkage is not None:
        if n_cell_clusters is None:
            n_cell_clusters = min(n_archetypes, 10)
        cell_labels_ordered = fcluster(cell_linkage, n_cell_clusters, criterion="maxclust")
        cell_labels_ordered = cell_labels_ordered[cell_order]
    else:
        cell_labels_ordered = None

    # Create figure with gridspec
    if show_dendrogram and cluster_cells:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            1, 3, width_ratios=[0.15, 0.05, 0.8], wspace=0.02
        )
        ax_dendro = fig.add_subplot(gs[0])
        ax_labels = fig.add_subplot(gs[1])
        ax_heat = fig.add_subplot(gs[2])
    else:
        fig, ax_heat = plt.subplots(figsize=figsize)
        ax_dendro = None
        ax_labels = None

    # Plot dendrogram
    if ax_dendro is not None and cell_linkage is not None:
        dendrogram(
            cell_linkage,
            orientation="left",
            ax=ax_dendro,
            no_labels=True,
            color_threshold=0,
            above_threshold_color="gray",
        )
        ax_dendro.set_xticks([])
        ax_dendro.set_yticks([])
        ax_dendro.spines["top"].set_visible(False)
        ax_dendro.spines["right"].set_visible(False)
        ax_dendro.spines["bottom"].set_visible(False)
        ax_dendro.spines["left"].set_visible(False)
        ax_dendro.invert_yaxis()

    # Plot cell labels color bar
    if ax_labels is not None and cell_labels_ordered is not None:
        unique_labels = np.unique(cell_labels_ordered)
        label_colors = plt.cm.get_cmap("tab10")(
            np.linspace(0, 1, len(unique_labels))
        )
        label_to_color = {l: label_colors[i] for i, l in enumerate(unique_labels)}
        colors_array = np.array([label_to_color[l] for l in cell_labels_ordered])
        ax_labels.imshow(
            colors_array.reshape(-1, 1, 4),
            aspect="auto",
            interpolation="nearest",
        )
        ax_labels.set_xticks([])
        ax_labels.set_yticks([])
        ax_labels.set_xlabel("Type", fontsize=8)
    elif ax_labels is not None:
        ax_labels.axis("off")

    # Plot heatmap
    im = ax_heat.imshow(z_ordered, aspect="auto", cmap=cmap, interpolation="nearest")
    ax_heat.set_xticks(np.arange(n_archetypes))
    ax_heat.set_xticklabels(archetype_names_ordered, rotation=45, ha="right")
    ax_heat.set_ylabel("Cells")
    ax_heat.set_yticks([])
    ax_heat.set_title(title)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.6, label="Weight")

    plt.tight_layout()

    return fig, cell_order
