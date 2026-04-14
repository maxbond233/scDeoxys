import pytest
import torch
from scdeoxys import ParetoVAE, generate_swiss_roll_simplex, Trainer


@pytest.fixture
def small_adata():
    return generate_swiss_roll_simplex(
        n_cells=50, n_genes=20, n_archetypes=3, seed=42
    )


@pytest.fixture
def trained_model(small_adata):
    model = ParetoVAE(n_genes=20, n_archetypes=3, hidden_dims=[32, 16])
    trainer = Trainer(model, learning_rate=1e-3, beta_warmup_epochs=2)
    trainer.train(small_adata, n_epochs=3, batch_size=25, verbose=False)
    return trainer, small_adata
