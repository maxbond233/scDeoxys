# scDeoxys

**Single-cell Deconvolution using Pareto-optimal Archetypal Analysis**

A deep learning framework for mapping curved gene expression manifolds to interpretable simplex spaces using variational autoencoders.

## Overview

scDeoxys implements a Pareto-optimal Variational Autoencoder (ParetoVAE) that learns to represent single-cell gene expression data as convex combinations of archetypal cell states. The key innovation is mapping curved, nonlinear manifolds in gene expression space to a regular simplex structure in latent space, enabling interpretable analysis of cell state transitions.

### Key Features (MVP - Level 1)

- **ParetoVAE Architecture**: Encoder-decoder VAE with simplex latent space
- **Negative Binomial Likelihood**: Proper modeling of count data with over-dispersion
- **ELBO Loss Function**: Reconstruction loss + KL divergence with beta warmup
- **Synthetic Data Generation**: Swiss Roll Simplex for validation
- **Visualization Tools**: Simplex projection and training curve plots

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- SciPy >= 1.10.0
- pandas >= 2.0.0
- anndata >= 0.10.0
- scanpy >= 1.9.0

### Install from source

```bash
git clone https://github.com/yourusername/scDeoxys.git
cd scDeoxys
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
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
trainer = Trainer(model, learning_rate=1e-3)
history = trainer.train(adata, n_epochs=100)

# Encode to simplex space
trainer.encode(adata)
z = adata.obsm["X_scdeoxys"]
```

## Example

Run the complete example:

```bash
python examples/basic_example.py
```

This will:
1. Generate synthetic Swiss Roll Simplex data
2. Train a ParetoVAE model
3. Generate visualization plots (training_curves.png, comparison.png)

## Project Structure

```
scDeoxys/
├── scdeoxys/              # Main package
│   ├── models/            # Model definitions
│   │   ├── vae.py        # ParetoVAE model
│   │   └── distributions.py  # Negative Binomial
│   ├── data/              # Data generation
│   │   └── synthetic.py  # Swiss Roll Simplex
│   ├── train/             # Training logic
│   │   └── trainer.py    # Trainer class
│   └── plotting/          # Visualization
│       └── simplex.py    # Simplex plots
├── examples/              # Usage examples
│   └── basic_example.py
└── tests/                 # Unit tests
```

## Model Architecture

### Encoder
- Input: Gene expression counts (n_genes,)
- Hidden layers: [n_genes → 256 → 128]
- Output: Simplex coordinates (n_archetypes,) via Softmax

### Decoder
- Input: Simplex coordinates (n_archetypes,)
- Hidden layers: [n_archetypes → 128 → 256]
- Output: NB parameters (μ, θ)

### Loss Function

```
L_total = L_recon + β * L_KL
```

- **L_recon**: Negative log-likelihood under Negative Binomial
- **L_KL**: KL divergence from uniform Dirichlet prior
- **β**: Warmup from 0.1 to 1.0 over 50 epochs

## Validation

The MVP validates the core concept using Swiss Roll Simplex data:

1. **Training Stability**: Loss converges without divergence
2. **Manifold Unfolding**: Curved manifold mapped to regular simplex
3. **Archetype Recovery**: Learned archetypes match ground truth

## Future Directions (Post-MVP)

- **Level 2**: Group Lasso/ARD for automatic K selection
- **Level 3**: Mixed likelihood models for noise robustness
- **Level 4**: Batch correction with conditional VAE
- **Integration**: AnnData compatibility for real scRNA-seq data

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
