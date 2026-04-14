"""
Training utilities for scDeoxys.

Implements the Trainer class for training ParetoVAE models with
beta warmup and loss monitoring.
"""

import copy

import torch
import numpy as np
import anndata as ad
from scipy import sparse
from tqdm import tqdm


class Trainer:
    """
    Trainer for ParetoVAE models.

    Handles training loop, beta warmup, and metric tracking.

    Args:
        model: ParetoVAE model instance
        learning_rate: Learning rate for optimizer
        beta_warmup_epochs: Number of epochs for beta warmup
        beta_min: Starting beta value for warmup (default 0.0)
        beta_max: Final beta value after warmup (default 1.0)
        free_bits: Minimum KL per dimension to prevent posterior collapse (default 0.1)
        device: Device to train on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        model,
        learning_rate=1e-3,
        beta_warmup_epochs=50,
        beta_min=0.0,
        beta_max=1.0,
        free_bits=0.1,
        device=None,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.beta_warmup_epochs = beta_warmup_epochs
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.free_bits = free_bits

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        self.history = {
            "total_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "beta": [],
        }

    def get_beta(self, epoch):
        """
        Compute beta value with warmup schedule.

        Beta increases linearly from beta_min to beta_max over warmup_epochs.

        Args:
            epoch: Current epoch number

        Returns:
            beta: Beta value for this epoch
        """
        if epoch < self.beta_warmup_epochs:
            progress = epoch / self.beta_warmup_epochs
            beta = self.beta_min + (self.beta_max - self.beta_min) * progress
        else:
            beta = self.beta_max
        return beta

    def train_epoch(self, data_loader, epoch):
        """
        Train for one epoch.

        Args:
            data_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            metrics: Dictionary of average metrics for this epoch
        """
        self.model.train()
        beta = self.get_beta(epoch)

        total_losses = []
        recon_losses = []
        kl_losses = []

        for batch in data_loader:
            x = batch[0].to(self.device)

            # Forward pass
            mu_recon, theta, z, mu_latent, logvar_latent = self.model(x)

            # Compute loss with free bits
            total_loss, recon_loss, kl_loss = self.model.loss(
                x, mu_recon, theta, z, mu_latent, logvar_latent,
                beta=beta, free_bits=self.free_bits
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Record losses
            total_losses.append(total_loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())

        metrics = {
            "total_loss": np.mean(total_losses),
            "recon_loss": np.mean(recon_losses),
            "kl_loss": np.mean(kl_losses),
            "beta": beta,
        }

        return metrics

    def _get_counts(self, adata):
        """
        Extract count matrix from AnnData.

        Args:
            adata: AnnData with layers["counts"]

        Returns:
            x: Count matrix as numpy array
        """
        if not isinstance(adata, ad.AnnData):
            raise TypeError("Expected AnnData input. Got: {}".format(type(adata)))
        if "counts" not in adata.layers:
            raise KeyError("AnnData.layers['counts'] is required for training.")

        x = adata.layers["counts"]
        if sparse.issparse(x):
            x = x.toarray()
        return np.asarray(x)

    def train(self, adata, n_epochs=100, batch_size=128, verbose=True):
        """
        Train the model.

        Args:
            adata: AnnData with layers["counts"]
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Whether to display progress bar

        Returns:
            history: Dictionary of training metrics over epochs
        """
        x_train = self._get_counts(adata)

        # Reset history for this training run
        self.history = {
            "total_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "beta": [],
        }

        x_train = torch.tensor(x_train, dtype=torch.float32)

        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(x_train)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Training loop
        iterator = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)

        for epoch in iterator:
            metrics = self.train_epoch(data_loader, epoch)

            # Record metrics
            self.history["total_loss"].append(metrics["total_loss"])
            self.history["recon_loss"].append(metrics["recon_loss"])
            self.history["kl_loss"].append(metrics["kl_loss"])
            self.history["beta"].append(metrics["beta"])

            # Update progress bar
            if verbose:
                iterator.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "recon": f"{metrics['recon_loss']:.4f}",
                    "kl": f"{metrics['kl_loss']:.4f}",
                    "beta": f"{metrics['beta']:.2f}",
                })

        scdeoxys_meta = adata.uns.get("scdeoxys", {})
        scdeoxys_meta["history"] = copy.deepcopy(self.history)
        scdeoxys_meta["training_params"] = {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "learning_rate": self.learning_rate,
            "beta_warmup_epochs": self.beta_warmup_epochs,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "free_bits": self.free_bits,
            "device": str(self.device),
        }
        scdeoxys_meta["model_params"] = {
            "n_genes": getattr(self.model, "n_genes", None),
            "n_archetypes": getattr(self.model, "n_archetypes", None),
        }
        adata.uns["scdeoxys"] = scdeoxys_meta

        return self.history

    def encode(self, adata):
        """
        Encode data to simplex coordinates.

        Args:
            adata: AnnData with layers["counts"]

        Returns:
            adata: AnnData with obsm["X_scdeoxys"] set
        """
        self.model.eval()

        x = self._get_counts(adata)
        x = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            mu_latent, logvar_latent, z = self.model.encode(x)

        adata.obsm["X_scdeoxys"] = z.cpu().numpy()
        return adata
