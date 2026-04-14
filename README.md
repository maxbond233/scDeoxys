# scDeoxys

**Single-cell Deconvolution using Pareto-optimal Archetypal Analysis**

A deep learning framework for mapping curved gene expression manifolds to interpretable simplex spaces using variational autoencoders.

## Overview

scDeoxys implements a Pareto-optimal Variational Autoencoder (ParetoVAE) that learns to represent single-cell gene expression data as convex combinations of archetypal cell states. The key innovation is mapping curved, nonlinear manifolds in gene expression space to a regular simplex structure in latent space, enabling interpretable analysis of cell state transitions.

### Key Features

- **ParetoVAE Architecture**: Encoder-decoder VAE with simplex latent space and library size normalization
- **Negative Binomial Likelihood**: Proper modeling of count data with learned per-gene dispersion
- **ELBO Loss Function**: Reconstruction loss + KL divergence with beta warmup and free bits
- **Synthetic Data Generation**: Swiss Roll Simplex for validation
- **Visualization Tools**: Simplex projection, UMAP weights, archetype heatmap, training curves

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- SciPy >= 1.10.0
- pandas >= 2.0.0
- tqdm >= 4.65.0
- anndata >= 0.10.0
- scanpy >= 1.9.0

### Install from source

```bash
git clone https://github.com/maxbond233/scDeoxys.git
cd scDeoxys
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from scdeoxys import ParetoVAE, generate_swiss_roll_simplex, Trainer

# Generate synthetic data (AnnData)
adata = generate_swiss_roll_simplex(
    n_cells=1000,
    n_genes=500,
    n_archetypes=3,
    seed=42
)

# Create and train model
model = ParetoVAE(n_genes=500, n_archetypes=3)
trainer = Trainer(
    model,
    learning_rate=1e-3,
    beta_warmup_epochs=100,
    beta_max=0.5,
    free_bits=0.1,
)
history = trainer.train(adata, n_epochs=250)

# Encode to simplex space
trainer.encode(adata)
z = adata.obsm["X_scdeoxys"]
```

## Example

Run the complete example with multi-method comparison:

```bash
python examples/basic_example.py
```

This will:
1. Generate synthetic Swiss Roll Simplex data (K=3, curvature=0.5)
2. Train ParetoVAE and run PCA baseline
3. Compute evaluation metrics (correlation, kNN, trustworthiness, ARI, etc.)
4. Generate visualization plots

## Project Structure

```
scDeoxys/
├── scdeoxys/              # Main package
│   ├── models/            # Model definitions
│   │   ├── vae.py        # ParetoVAE (Encoder, Decoder, loss)
│   │   └── distributions.py  # Negative Binomial distribution
│   ├── data/              # Data generation
│   │   └── synthetic.py  # Swiss Roll Simplex generator
│   ├── train/             # Training logic
│   │   └── trainer.py    # Trainer with beta warmup
│   └── plotting/          # Visualization
│       └── simplex.py    # Simplex, UMAP, heatmap plots
├── examples/              # Usage examples
│   └── basic_example.py  # Multi-method comparison demo
└── tests/                 # Unit tests
    ├── conftest.py       # Shared fixtures
    └── test_smoke.py     # Smoke tests (train, encode, plot)
```

## Model Architecture

### Encoder
- Input: Log-normalized gene expression (n_genes,) — raw counts are automatically normalized via `log1p(x / library_size * 1e4)`
- Hidden layers: [n_genes -> 256 -> 128] with BatchNorm, ReLU, Dropout(0.1)
- Variational: outputs mu and logvar (n_archetypes each)
- Reparameterization trick: logits = mu + sigma * epsilon
- Output: Simplex coordinates (n_archetypes,) via Softmax on logits

### Decoder
- Input: Simplex coordinates (n_archetypes,) + per-cell library size
- Hidden layers: [n_archetypes -> 128 -> 256] with BatchNorm, ReLU, Dropout(0.1)
- Output mu: Softplus activation, scaled by per-cell library size (scVI-style)
- Output theta: Learnable per-gene dispersion parameter, Softplus activation

### Loss Function

```
L_total = L_recon + beta * L_KL
```

- **L_recon**: Negative log-likelihood under Negative Binomial
- **L_KL**: KL divergence from standard normal prior (Logistic-Normal latent via softmax)
- **beta**: Linear warmup from 0.0 to beta_max (configurable, default 50 epochs)
- **Free bits**: Optional minimum KL per dimension to prevent posterior collapse (default 0.1)

### Library Size Normalization

The model automatically handles per-cell sequencing depth differences:
- Encoder input is log-normalized to remove depth variation
- Decoder output is scaled by per-cell library size
- NB likelihood is evaluated against raw counts

This prevents the latent space from capturing sequencing depth instead of biological cell state.

## Validation

Validated on Swiss Roll Simplex synthetic data (K=3, curvature=0.5, 1000 cells, 500 genes):

| Method | Correlation | kNN | Trustworthiness | Silhouette | ARI |
|--------|------------|-----|-----------------|------------|-----|
| scDeoxys | **0.961** | **0.526** | **0.983** | **0.426** | 0.730 |
| PCA | 0.809 | 0.509 | 0.978 | 0.389 | 0.743 |

## Testing

```bash
# Run smoke tests
pytest tests/ -v
```

Tests cover: training convergence, history management, encoding shape/simplex constraints, plotting, and synthetic data generation.

## Future Directions

- **Level 2**: Group Lasso/ARD for automatic K selection
- **Level 3**: Mixed likelihood models for noise robustness
- **Level 4**: Batch correction with conditional VAE

## Citation

If you use scDeoxys in your research, please cite:

```
@software{scdeoxys2024,
  title={scDeoxys: Single-cell Deconvolution using Pareto-optimal Archetypal Analysis},
  author={scDeoxys Team},
  year={2024}
}
```

## License

MIT License

## Contact

For questions and feedback, please open an issue on GitHub.
