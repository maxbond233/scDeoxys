"""
Pareto-optimal Variational Autoencoder for single-cell data.

Implements a VAE that maps gene expression data to a simplex latent space,
where each point represents a convex combination of archetypal cell states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scdeoxys.models.distributions import NegativeBinomial


class Encoder(nn.Module):
    """
    Encoder network: maps gene expression to simplex coordinates.

    Architecture:
        Input (n_genes) → Hidden layers → Simplex coordinates (n_archetypes)
    """

    def __init__(self, n_genes, n_archetypes, hidden_dims=[256, 128]):
        super().__init__()

        layers = []
        input_dim = n_genes

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        # Output layers for variational inference
        self.fc_mu = nn.Linear(input_dim, n_archetypes)
        self.fc_logvar = nn.Linear(input_dim, n_archetypes)

    def forward(self, x):
        """
        Encode gene expression to simplex coordinates using variational inference.

        Uses Logistic-Normal distribution with reparameterization trick:
        1. Output mean (mu) and log-variance (logvar)
        2. Sample: logits = mu + sigma * epsilon
        3. Apply softmax to get simplex coordinates

        Args:
            x: Gene expression, shape (batch_size, n_genes)

        Returns:
            mu: Mean of latent distribution, shape (batch_size, n_archetypes)
            logvar: Log variance of latent distribution, shape (batch_size, n_archetypes)
            z: Simplex coordinates (sampled), shape (batch_size, n_archetypes)
        """
        h = self.hidden(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            logits = mu + std * eps
        else:
            # During inference, use mean
            logits = mu

        # Apply softmax to get simplex coordinates
        z = F.softmax(logits, dim=-1)

        return mu, logvar, z


class Decoder(nn.Module):
    """
    Decoder network: maps simplex coordinates to NB distribution parameters.

    Architecture:
        Simplex (n_archetypes) → Hidden layers → NB params (μ, θ)
    """

    def __init__(self, n_archetypes, n_genes, hidden_dims=[128, 256], dropout=0.3):
        super().__init__()

        layers = []
        input_dim = n_archetypes

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.mu_output = nn.Linear(input_dim, n_genes)

        # Learnable dispersion parameter (one per gene)
        self.theta_log = nn.Parameter(torch.zeros(n_genes))

    def forward(self, z, library_size):
        """
        Decode simplex coordinates to NB parameters.

        Args:
            z: Simplex coordinates, shape (batch_size, n_archetypes)
            library_size: Per-cell total counts, shape (batch_size, 1)

        Returns:
            mu: Mean parameter (library-scaled), shape (batch_size, n_genes)
            theta: Dispersion parameter, shape (n_genes,)
        """
        h = self.hidden(z)
        rho = F.softplus(self.mu_output(h))
        mu = rho * library_size
        theta = F.softplus(self.theta_log)
        return mu, theta


class ParetoVAE(nn.Module):
    """
    Pareto-optimal Variational Autoencoder.

    Maps gene expression data to a simplex latent space representing
    convex combinations of archetypal cell states.

    Args:
        n_genes: Number of genes
        n_archetypes: Number of archetypal states (K)
        hidden_dims: Hidden layer dimensions for encoder/decoder
    """

    def __init__(self, n_genes, n_archetypes, hidden_dims=[256, 128], decoder_dropout=0.3):
        super().__init__()

        self.n_genes = n_genes
        self.n_archetypes = n_archetypes

        self.encoder = Encoder(n_genes, n_archetypes, hidden_dims)
        decoder_hidden = hidden_dims[::-1]
        self.decoder = Decoder(n_archetypes, n_genes, decoder_hidden, dropout=decoder_dropout)

    def forward(self, x):
        """
        Forward pass through the VAE.

        Computes library size from raw counts, log-normalizes input for the
        encoder, and scales decoder output by library size.

        Args:
            x: Gene expression counts, shape (batch_size, n_genes)

        Returns:
            mu_recon: Reconstructed mean (library-scaled), shape (batch_size, n_genes)
            theta: Dispersion parameter, shape (n_genes,)
            z: Simplex coordinates, shape (batch_size, n_archetypes)
            mu_latent: Mean of latent distribution, shape (batch_size, n_archetypes)
            logvar_latent: Log variance of latent distribution, shape (batch_size, n_archetypes)
        """
        library_size = x.sum(dim=1, keepdim=True)
        x_norm = torch.log1p(x / (library_size + 1e-8) * 1e4)
        mu_latent, logvar_latent, z = self.encoder(x_norm)
        mu_recon, theta = self.decoder(z, library_size)
        return mu_recon, theta, z, mu_latent, logvar_latent

    def encode(self, x):
        """
        Encode raw counts to simplex coordinates with proper normalization.

        Args:
            x: Gene expression counts, shape (batch_size, n_genes)

        Returns:
            mu: Mean of latent distribution, shape (batch_size, n_archetypes)
            logvar: Log variance, shape (batch_size, n_archetypes)
            z: Simplex coordinates, shape (batch_size, n_archetypes)
        """
        library_size = x.sum(dim=1, keepdim=True)
        x_norm = torch.log1p(x / (library_size + 1e-8) * 1e4)
        return self.encoder(x_norm)

    def loss(self, x, mu_recon, theta, z, mu_latent, logvar_latent, beta=1.0, free_bits=0.0, eps=1e-8):
        """
        Compute ELBO loss with correct KL divergence and free bits.

        L_total = L_recon + β * L_KL

        Free bits: ensures minimum KL per dimension to prevent posterior collapse.
        KL_dim = max(KL_dim, free_bits) for each latent dimension.

        Args:
            x: Original counts, shape (batch_size, n_genes)
            mu_recon: Reconstructed mean, shape (batch_size, n_genes)
            theta: Dispersion parameter, shape (n_genes,)
            z: Simplex coordinates, shape (batch_size, n_archetypes)
            mu_latent: Mean of latent distribution, shape (batch_size, n_archetypes)
            logvar_latent: Log variance of latent distribution, shape (batch_size, n_archetypes)
            beta: Weight for KL divergence term
            free_bits: Minimum KL per dimension (default 0.0, set >0 to prevent collapse)
            eps: Small constant for numerical stability

        Returns:
            total_loss: Total ELBO loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # Reconstruction loss: negative log-likelihood (sum over genes, mean over batch)
        log_prob = NegativeBinomial.log_prob(x, mu_recon, theta, eps)
        recon_loss = -log_prob.sum(dim=-1).mean()

        # KL divergence per dimension: KL(N(mu, sigma^2) || N(0, 1))
        # KL_dim = 0.5 * (sigma^2 + mu^2 - 1 - log(sigma^2))
        kl_per_dim = 0.5 * (logvar_latent.exp() + mu_latent.pow(2) - 1 - logvar_latent)

        # Apply free bits: average over batch first, then clamp per dimension
        # (Kingma et al., 2016 formulation)
        if free_bits > 0:
            kl_per_dim_mean = kl_per_dim.mean(dim=0)  # (n_archetypes,)
            kl_per_dim_mean = torch.clamp(kl_per_dim_mean, min=free_bits)
            kl_loss = kl_per_dim_mean.sum()
        else:
            kl_loss = kl_per_dim.sum(dim=-1).mean()

        # Scale beta by n_genes/n_archetypes to balance recon (sum over genes)
        # and KL (sum over archetypes) dimensions
        n_genes = x.shape[1]
        n_latent = mu_latent.shape[1]
        beta_scaled = beta * (n_genes / n_latent)

        total_loss = recon_loss + beta_scaled * kl_loss

        return total_loss, recon_loss, kl_loss
