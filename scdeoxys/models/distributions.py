"""
Probability distributions for scDeoxys.

Implements the Negative Binomial distribution for modeling gene expression counts.
"""

import torch
import torch.nn.functional as F


class NegativeBinomial:
    """
    Negative Binomial distribution for count data.

    The NB distribution is parameterized by:
    - mu (μ): mean parameter
    - theta (θ): dispersion parameter (inverse of over-dispersion)

    The probability mass function is:
    P(k | μ, θ) = Γ(k + θ) / (Γ(θ) * k!) * (θ/(θ+μ))^θ * (μ/(θ+μ))^k

    where Γ is the gamma function.
    """

    @staticmethod
    def log_prob(counts, mu, theta, eps=1e-8):
        """
        Compute log probability of counts under NB distribution.

        Args:
            counts: Observed counts, shape (batch_size, n_genes)
            mu: Mean parameter, shape (batch_size, n_genes)
            theta: Dispersion parameter, shape (n_genes,) or scalar
            eps: Small constant for numerical stability

        Returns:
            log_prob: Log probability, shape (batch_size, n_genes)
        """
        # Ensure positive parameters
        mu = torch.clamp(mu, min=eps)
        theta = torch.clamp(theta, min=eps)

        # Compute log probability using lgamma for numerical stability
        # log P(k | μ, θ) = log Γ(k + θ) - log Γ(θ) - log Γ(k + 1)
        #                   + θ * log(θ/(θ+μ)) + k * log(μ/(θ+μ))

        log_theta_mu_eps = torch.log(theta + mu + eps)
        log_theta_eps = torch.log(theta + eps)
        log_mu_eps = torch.log(mu + eps)

        lgamma_counts_theta = torch.lgamma(counts + theta)
        lgamma_theta = torch.lgamma(theta)
        lgamma_counts_plus_1 = torch.lgamma(counts + 1.0)

        log_prob = (
            lgamma_counts_theta
            - lgamma_theta
            - lgamma_counts_plus_1
            + theta * (log_theta_eps - log_theta_mu_eps)
            + counts * (log_mu_eps - log_theta_mu_eps)
        )

        return log_prob

    @staticmethod
    def sample(mu, theta, eps=1e-8):
        """
        Sample from NB distribution using Gamma-Poisson mixture.

        NB can be represented as a Poisson distribution where the rate
        parameter is drawn from a Gamma distribution.

        Args:
            mu: Mean parameter, shape (batch_size, n_genes)
            theta: Dispersion parameter, shape (n_genes,) or scalar
            eps: Small constant for numerical stability

        Returns:
            samples: Sampled counts, shape (batch_size, n_genes)
        """
        # Ensure positive parameters
        mu = torch.clamp(mu, min=eps)
        theta = torch.clamp(theta, min=eps)

        # Gamma-Poisson mixture representation
        # 1. Sample rate from Gamma(theta, theta/mu)
        # 2. Sample counts from Poisson(rate)
        concentration = theta
        rate = theta / mu

        gamma_dist = torch.distributions.Gamma(concentration, rate)
        lambda_sample = gamma_dist.sample()

        poisson_dist = torch.distributions.Poisson(lambda_sample)
        samples = poisson_dist.sample()

        return samples
