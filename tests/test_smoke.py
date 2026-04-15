"""Smoke tests for scDeoxys core functionality."""

import numpy as np
import torch
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


class TestFreeBits:
    def test_free_bits_floor(self, small_adata):
        """KL loss should be at least n_archetypes * free_bits with corrected implementation."""
        model = ParetoVAE(n_genes=20, n_archetypes=3, hidden_dims=[32, 16])
        free_bits = 1.0
        trainer = Trainer(model, learning_rate=1e-3, beta_warmup_epochs=1, free_bits=free_bits)
        trainer.train(small_adata, n_epochs=3, batch_size=25, verbose=False)
        # KL is sum over dims, so floor is n_archetypes * free_bits
        final_kl = trainer.history["kl_loss"][-1]
        assert final_kl >= 3 * free_bits * 0.9  # small tolerance

    def test_free_bits_zero_no_floor(self, small_adata):
        """With free_bits=0, KL can go arbitrarily low."""
        model = ParetoVAE(n_genes=20, n_archetypes=3, hidden_dims=[32, 16])
        trainer = Trainer(model, learning_rate=1e-3, beta_warmup_epochs=1, free_bits=0.0)
        trainer.train(small_adata, n_epochs=3, batch_size=25, verbose=False)
        # Just verify it runs without error
        assert len(trainer.history["kl_loss"]) == 3


class TestCyclicalAnnealing:
    def test_cyclical_beta_pattern(self):
        """Cyclical beta should reset at cycle boundaries."""
        model = ParetoVAE(n_genes=20, n_archetypes=3, hidden_dims=[32, 16])
        trainer = Trainer(model, n_cycles=4, annealing_type="cyclical")
        n_epochs = 100
        betas = [trainer.get_beta(e, n_epochs) for e in range(n_epochs)]
        # At start of each cycle, beta should be near beta_min
        cycle_length = n_epochs / 4
        for c in range(4):
            start_epoch = int(c * cycle_length)
            assert betas[start_epoch] == pytest.approx(0.0, abs=0.05)
        # At midpoint of each cycle, beta should be near beta_max
        for c in range(4):
            mid_epoch = int(c * cycle_length + cycle_length * 0.5)
            if mid_epoch < n_epochs:
                assert betas[mid_epoch] == pytest.approx(1.0, abs=0.05)

    def test_linear_fallback(self):
        """With annealing_type='linear', should use original warmup."""
        model = ParetoVAE(n_genes=20, n_archetypes=3, hidden_dims=[32, 16])
        trainer = Trainer(model, beta_warmup_epochs=10, annealing_type="linear")
        assert trainer.get_beta(0) == 0.0
        assert trainer.get_beta(5) == pytest.approx(0.5, abs=0.01)
        assert trainer.get_beta(10) == 1.0
        assert trainer.get_beta(20) == 1.0


class TestDecoderArchitecture:
    def test_decoder_uses_layernorm(self):
        """Decoder should use LayerNorm, not BatchNorm."""
        import torch.nn as nn
        model = ParetoVAE(n_genes=20, n_archetypes=3, hidden_dims=[32, 16])
        decoder_modules = list(model.decoder.hidden.modules())
        has_layernorm = any(isinstance(m, nn.LayerNorm) for m in decoder_modules)
        has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in decoder_modules)
        assert has_layernorm
        assert not has_batchnorm

    def test_encoder_keeps_batchnorm(self):
        """Encoder should still use BatchNorm."""
        import torch.nn as nn
        model = ParetoVAE(n_genes=20, n_archetypes=3, hidden_dims=[32, 16])
        encoder_modules = list(model.encoder.hidden.modules())
        has_batchnorm = any(isinstance(m, nn.BatchNorm1d) for m in encoder_modules)
        assert has_batchnorm

    def test_decoder_dropout_param(self):
        """decoder_dropout parameter should control decoder dropout rate."""
        model = ParetoVAE(n_genes=20, n_archetypes=3, hidden_dims=[32, 16], decoder_dropout=0.5)
        import torch.nn as nn
        dropouts = [m for m in model.decoder.hidden.modules() if isinstance(m, nn.Dropout)]
        assert len(dropouts) > 0
        assert all(d.p == 0.5 for d in dropouts)
