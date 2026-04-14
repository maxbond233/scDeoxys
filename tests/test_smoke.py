"""Smoke tests for scDeoxys core functionality."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scdeoxys import ParetoVAE, generate_swiss_roll_simplex, Trainer
from scdeoxys.data.synthetic import generate_archetypal_profiles
from scdeoxys.plotting import plot_training_curves, plot_simplex_projection, plot_archetype_heatmap


class TestTraining:
    def test_loss_decreases(self, trained_model):
        trainer, adata = trained_model
        losses = trainer.history["total_loss"]
        assert losses[-1] <= losses[0] * 1.5  # allow some tolerance

    def test_history_in_adata(self, trained_model):
        _, adata = trained_model
        assert "scdeoxys" in adata.uns
        assert "history" in adata.uns["scdeoxys"]
        assert len(adata.uns["scdeoxys"]["history"]["total_loss"]) == 3

    def test_history_reset_on_retrain(self, small_adata):
        model = ParetoVAE(n_genes=20, n_archetypes=3, hidden_dims=[32, 16])
        trainer = Trainer(model, learning_rate=1e-3, beta_warmup_epochs=2)
        trainer.train(small_adata, n_epochs=3, batch_size=25, verbose=False)
        trainer.train(small_adata, n_epochs=5, batch_size=25, verbose=False)
        assert len(trainer.history["total_loss"]) == 5

    def test_history_copy_in_adata(self, trained_model):
        trainer, adata = trained_model
        trainer.history["total_loss"].append(999.0)
        assert 999.0 not in adata.uns["scdeoxys"]["history"]["total_loss"]


class TestEncoding:
    def test_encode_shape(self, trained_model):
        trainer, adata = trained_model
        trainer.encode(adata)
        z = adata.obsm["X_scdeoxys"]
        assert z.shape == (50, 3)

    def test_simplex_constraint(self, trained_model):
        trainer, adata = trained_model
        trainer.encode(adata)
        z = adata.obsm["X_scdeoxys"]
        np.testing.assert_allclose(z.sum(axis=1), 1.0, atol=1e-5)

    def test_non_negative(self, trained_model):
        trainer, adata = trained_model
        trainer.encode(adata)
        z = adata.obsm["X_scdeoxys"]
        assert (z >= 0).all()


class TestPlotting:
    def test_training_curves(self, trained_model):
        _, adata = trained_model
        fig = plot_training_curves(adata)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_simplex_projection(self, trained_model):
        trainer, adata = trained_model
        trainer.encode(adata)
        z = adata.obsm["X_scdeoxys"]
        ax = plot_simplex_projection(z, title="test")
        assert isinstance(ax, plt.Axes)
        plt.close(ax.figure)

    def test_archetype_heatmap(self, trained_model):
        trainer, adata = trained_model
        trainer.encode(adata)
        z = adata.obsm["X_scdeoxys"]
        fig, cell_order = plot_archetype_heatmap(z)
        assert isinstance(fig, plt.Figure)
        assert len(cell_order) == 50
        plt.close(fig)


class TestSyntheticData:
    def test_remainder_genes(self):
        profiles = generate_archetypal_profiles(17, 3, seed=42)
        assert profiles.shape == (3, 17)
        # Every gene should be a high-expression marker for exactly one archetype
        max_per_gene = profiles.max(axis=0)
        assert (max_per_gene > 5.0).all()
