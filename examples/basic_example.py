"""
Basic example demonstrating scDeoxys workflow with multi-method comparison.

This script:
1. Generates synthetic Swiss Roll Simplex data
2. Runs multiple dimensionality reduction methods (scDeoxys, PCA, etc.)
3. Computes unified evaluation metrics for all methods
4. Visualizes and compares results
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from abc import ABC, abstractmethod
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import (
        silhouette_score,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from scdeoxys import ParetoVAE, generate_swiss_roll_simplex, Trainer
from scdeoxys.plotting import (
    plot_training_curves,
    plot_simplex_projection,
    plot_umap_weights,
    plot_archetype_heatmap,
)


# =============================================================================
# Method Wrapper Base Class
# =============================================================================


class BaseMethod(ABC):
    """Abstract base class for dimensionality reduction methods."""

    def __init__(self, name: str, n_components: int):
        self.name = name
        self.n_components = n_components
        self.embedding_ = None
        self.is_simplex = False

    @abstractmethod
    def fit_transform(self, x: np.ndarray, adata=None) -> np.ndarray:
        """Fit the model and return embeddings."""
        pass

    def to_simplex(self, z: np.ndarray) -> np.ndarray:
        """Project embeddings to simplex (for non-simplex methods)."""
        z_shifted = z - z.min(axis=0)
        row_sums = z_shifted.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return z_shifted / row_sums


class ScDeoxysMethod(BaseMethod):
    """scDeoxys (ParetoVAE) method wrapper."""

    def __init__(
        self,
        n_archetypes: int = 5,
        hidden_dims: list = None,
        n_epochs: int = 250,
        learning_rate: float = 1e-3,
    ):
        super().__init__("scDeoxys", n_archetypes)
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.is_simplex = True
        self.history_ = None
        self.trainer_ = None

    def fit_transform(self, x: np.ndarray, adata=None) -> np.ndarray:
        model = ParetoVAE(
            n_genes=x.shape[1],
            n_archetypes=self.n_components,
            hidden_dims=self.hidden_dims,
        )
        self.trainer_ = Trainer(
            model,
            learning_rate=self.learning_rate,
            beta_warmup_epochs=100,
            beta_min=0.0,
            beta_max=0.5,
            free_bits=0.1,
        )
        self.history_ = self.trainer_.train(
            adata, n_epochs=self.n_epochs, batch_size=128, verbose=True
        )
        self.trainer_.encode(adata)
        self.embedding_ = adata.obsm["X_scdeoxys"]
        return self.embedding_


class PCAMethod(BaseMethod):
    """PCA baseline method wrapper."""

    def __init__(self, n_components: int = 3):
        super().__init__("PCA", n_components)
        self.pca_ = None

    def fit_transform(self, x: np.ndarray, adata=None) -> np.ndarray:
        x_log = np.log1p(x)
        self.pca_ = PCA(n_components=self.n_components)
        self.embedding_ = self.pca_.fit_transform(x_log)
        return self.embedding_


# =============================================================================
# Evaluation Metrics
# =============================================================================


def knn_preservation(z_true: np.ndarray, z_pred: np.ndarray, k: int = 10) -> float:
    """Compute kNN preservation between original and embedded spaces."""
    nn_true = NearestNeighbors(n_neighbors=k + 1).fit(z_true)
    nn_pred = NearestNeighbors(n_neighbors=k + 1).fit(z_pred)

    _, idx_true = nn_true.kneighbors(z_true)
    _, idx_pred = nn_pred.kneighbors(z_pred)

    idx_true = idx_true[:, 1:]
    idx_pred = idx_pred[:, 1:]

    preservation = []
    for i in range(len(z_true)):
        overlap = len(set(idx_true[i]) & set(idx_pred[i]))
        preservation.append(overlap / k)

    return float(np.mean(preservation))


def continuity_score(z_true: np.ndarray, z_pred: np.ndarray, k: int = 10) -> float:
    """Continuity metric (higher is better). Uses neighbor rank penalty."""
    n = z_true.shape[0]

    nn_true = NearestNeighbors(n_neighbors=k + 1).fit(z_true)
    _, idx_true = nn_true.kneighbors(z_true)
    idx_true = idx_true[:, 1:]

    diff = z_pred[:, None, :] - z_pred[None, :, :]
    dist = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist, np.inf)
    order = np.argsort(dist, axis=1)

    ranks = np.empty_like(order)
    rows = np.arange(n)[:, None]
    ranks[rows, order] = np.arange(n)

    penalty = 0.0
    for i in range(n):
        for j in idx_true[i]:
            r_ij = ranks[i, j] + 1
            if r_ij > k:
                penalty += r_ij - k

    normalizer = n * k * (2 * n - 3 * k - 1)
    if normalizer <= 0:
        return 0.0

    return float(1.0 - (2.0 / normalizer) * penalty)


def align_to_simplex(z_pred: np.ndarray, z_true: np.ndarray):
    """
    Align predicted coordinates to true simplex using Hungarian algorithm.

    Returns:
        z_pred_aligned: Aligned prediction array
        perm: Permutation used for alignment
    """
    n_true = z_true.shape[1]
    n_pred = z_pred.shape[1]

    if n_pred < n_true:
        z_pred = np.hstack([z_pred, np.zeros((z_pred.shape[0], n_true - n_pred))])
    elif n_pred > n_true:
        correlations = []
        for i in range(n_pred):
            max_corr = max(
                abs(pearsonr(z_pred[:, i], z_true[:, j])[0]) for j in range(n_true)
            )
            correlations.append((i, max_corr))
        correlations.sort(key=lambda x: -x[1])
        selected = [c[0] for c in correlations[:n_true]]
        z_pred = z_pred[:, selected]

    cost_matrix = np.zeros((n_true, n_true))
    for i in range(n_true):
        for j in range(n_true):
            r, _ = pearsonr(z_pred[:, i], z_true[:, j])
            cost_matrix[i, j] = -r  # maximize positive correlation

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    perm = np.zeros(n_true, dtype=int)
    for k in range(len(row_ind)):
        perm[col_ind[k]] = row_ind[k]

    return z_pred[:, perm], perm.tolist()


def clustering_metrics(z_pred: np.ndarray, true_labels: np.ndarray) -> dict:
    """Compute clustering-based metrics."""
    n_clusters = len(np.unique(true_labels))
    pred_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
        z_pred
    )
    return {
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
    }


def evaluate_method(
    method: BaseMethod,
    z_pred: np.ndarray,
    z_true: np.ndarray,
    true_labels: np.ndarray,
    k: int = 10,
) -> dict:
    """
    Unified evaluation for any method.

    Args:
        method: Method wrapper instance
        z_pred: Predicted embeddings
        z_true: True simplex coordinates
        true_labels: True archetype labels
        k: Number of neighbors for kNN metrics

    Returns:
        Dictionary of metrics
    """
    metrics = {"method": method.name, "is_simplex": method.is_simplex}

    # For simplex comparison, project non-simplex methods
    if method.is_simplex:
        z_for_simplex = z_pred
    else:
        z_for_simplex = method.to_simplex(z_pred)

    # Align to true simplex
    z_aligned, perm = align_to_simplex(z_for_simplex, z_true)
    metrics["alignment_perm"] = perm

    # Simplex metrics (correlation, MSE)
    n_archetypes = z_true.shape[1]
    correlations = []
    for i in range(n_archetypes):
        r, _ = pearsonr(z_aligned[:, i], z_true[:, i])
        correlations.append(float(r))

    metrics["correlations"] = correlations
    metrics["mean_correlation"] = float(np.mean(correlations))
    metrics["mse"] = float(np.mean((z_aligned - z_true) ** 2))
    metrics["simplex_distance"] = float(
        np.mean(np.sum(np.abs(z_aligned - z_true), axis=1))
    )

    # Structure preservation metrics
    if SKLEARN_AVAILABLE:
        metrics["knn_preservation"] = knn_preservation(z_true, z_aligned, k=k)
        metrics["trustworthiness"] = float(
            trustworthiness(z_true, z_aligned, n_neighbors=k)
        )
        metrics["continuity"] = continuity_score(z_true, z_aligned, k=k)
        metrics["silhouette"] = float(silhouette_score(z_aligned, true_labels))

        # Clustering metrics
        cluster_metrics = clustering_metrics(z_aligned, true_labels)
        metrics.update(cluster_metrics)

    return metrics, z_aligned


# =============================================================================
# Visualization
# =============================================================================


def plot_method_comparison(results: dict, z_true: np.ndarray, colors: np.ndarray):
    """
    Plot comparison of all methods.

    Args:
        results: Dictionary mapping method names to (metrics, z_aligned)
        z_true: True simplex coordinates
        colors: Color values for points
    """
    n_methods = len(results) + 1  # +1 for ground truth
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))

    if n_methods == 1:
        axes = [axes]

    # Plot ground truth
    plot_simplex_projection(
        z_true, labels=colors, title="Ground Truth", ax=axes[0], cmap="rainbow"
    )

    # Plot each method
    for idx, (name, (metrics, z_aligned)) in enumerate(results.items(), 1):
        title = f"{name}\n(r={metrics['mean_correlation']:.3f})"
        plot_simplex_projection(
            z_aligned, labels=colors, title=title, ax=axes[idx], cmap="rainbow"
        )

    plt.tight_layout()
    return fig


def plot_metrics_comparison(all_metrics: list):
    """
    Plot bar chart comparing metrics across methods.

    Args:
        all_metrics: List of metric dictionaries
    """
    metric_names = [
        "mean_correlation",
        "knn_preservation",
        "trustworthiness",
        "continuity",
        "silhouette",
        "ari",
        "nmi",
    ]
    metric_labels = [
        "Correlation",
        "kNN Pres.",
        "Trust.",
        "Continuity",
        "Silhouette",
        "ARI",
        "NMI",
    ]

    methods = [m["method"] for m in all_metrics]
    n_metrics = len(metric_names)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    for i, metrics in enumerate(all_metrics):
        values = [metrics.get(name, 0) or 0 for name in metric_names]
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metrics["method"])

    ax.set_ylabel("Score")
    ax.set_title("Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def print_metrics_table(all_metrics: list):
    """Print metrics comparison table."""
    print("\n" + "=" * 80)
    print("METRICS COMPARISON")
    print("=" * 80)

    headers = ["Method", "Corr", "kNN", "Trust", "Cont", "Silh", "ARI", "NMI"]
    print(f"{'Method':<12} {'Corr':>8} {'kNN':>8} {'Trust':>8} {'Cont':>8} "
          f"{'Silh':>8} {'ARI':>8} {'NMI':>8}")
    print("-" * 80)

    for m in all_metrics:
        print(
            f"{m['method']:<12} "
            f"{m.get('mean_correlation', 0):>8.3f} "
            f"{m.get('knn_preservation', 0) or 0:>8.3f} "
            f"{m.get('trustworthiness', 0) or 0:>8.3f} "
            f"{m.get('continuity', 0) or 0:>8.3f} "
            f"{m.get('silhouette', 0) or 0:>8.3f} "
            f"{m.get('ari', 0) or 0:>8.3f} "
            f"{m.get('nmi', 0) or 0:>8.3f}"
        )

    print("=" * 80)


# =============================================================================
# Main
# =============================================================================


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("scDeoxys - Multi-Method Comparison Example")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic Swiss Roll Simplex data...")
    n_archetypes = 3  # K=3 matches Swiss Roll geometry (2 intrinsic dimensions)
    adata = generate_swiss_roll_simplex(
        n_cells=1000,
        n_genes=500,
        n_archetypes=n_archetypes,
        alpha=0.5,
        curvature=0.5,
        theta=15.0,
        seed=42,
    )

    x_train = adata.layers["counts"]
    z_true = adata.obsm["X_true_simplex"]
    colors = adata.obs["color_param"].values if "color_param" in adata.obs else None
    true_labels = np.argmax(z_true, axis=1)

    print(f"   - Cells: {x_train.shape[0]}, Genes: {x_train.shape[1]}")
    print(f"   - Archetypes: {n_archetypes}")

    # 2. Define methods to compare
    print("\n2. Setting up methods...")
    methods = [
        ScDeoxysMethod(n_archetypes=n_archetypes, n_epochs=250),
        PCAMethod(n_components=n_archetypes),
    ]

    # 3. Run all methods and evaluate
    print("\n3. Running methods and computing metrics...")
    results = {}
    all_metrics = []

    for method in methods:
        print(f"\n   --- {method.name} ---")
        z_pred = method.fit_transform(x_train, adata)
        metrics, z_aligned = evaluate_method(
            method, z_pred, z_true, true_labels, k=10
        )
        results[method.name] = (metrics, z_aligned)
        all_metrics.append(metrics)

        print(f"   Correlation: {metrics['mean_correlation']:.3f}")
        print(f"   MSE: {metrics['mse']:.4f}")
        if metrics.get("knn_preservation"):
            print(f"   kNN preservation: {metrics['knn_preservation']:.3f}")

    # 4. Print comparison table
    print_metrics_table(all_metrics)

    # 5. Save metrics
    print("\n4. Saving results...")
    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("   - Saved metrics.json")

    # 6. Generate visualizations
    print("\n5. Generating visualizations...")

    # Training curves (scDeoxys only)
    scdeoxys_method = methods[0]
    if scdeoxys_method.history_:
        adata.uns.setdefault("scdeoxys", {})["history"] = scdeoxys_method.history_
        fig1 = plot_training_curves(adata)
        plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
        print("   - Saved training_curves.png")
        plt.close(fig1)

    # Method comparison plot
    fig2 = plot_method_comparison(results, z_true, colors)
    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    print("   - Saved comparison.png")
    plt.close(fig2)

    # Metrics bar chart
    fig3 = plot_metrics_comparison(all_metrics)
    plt.savefig("metrics_comparison.png", dpi=150, bbox_inches="tight")
    print("   - Saved metrics_comparison.png")
    plt.close(fig3)

    # UMAP of archetype weights (for K>3 visualization)
    z_scdeoxys = results["scDeoxys"][1]
    try:
        fig4, umap_coords = plot_umap_weights(
            z_scdeoxys,
            color_by_dominant=True,
            title="UMAP of scDeoxys Weights",
        )
        plt.savefig("umap_weights.png", dpi=150, bbox_inches="tight")
        print("   - Saved umap_weights.png")
        plt.close(fig4)
    except ImportError:
        print("   - Skipped umap_weights.png (umap-learn not installed)")

    # Archetype heatmap with dendrogram
    fig5, cell_order = plot_archetype_heatmap(
        z_scdeoxys,
        cell_labels=true_labels,
        cluster_cells=True,
        show_dendrogram=True,
        title="scDeoxys Archetype Weights",
    )
    plt.savefig("archetype_heatmap.png", dpi=150, bbox_inches="tight")
    print("   - Saved archetype_heatmap.png")
    plt.close(fig5)

    print("\n" + "=" * 60)
    print("Example complete! Check the generated PNG files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
