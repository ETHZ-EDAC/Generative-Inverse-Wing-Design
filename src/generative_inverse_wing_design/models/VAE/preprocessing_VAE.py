from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Optional

import aerosandbox.numpy as np
import numpy as npo
from omegaconf import DictConfig
import pytorch_lightning as L
from scipy.special import comb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels: int, layers: list, output_dim: int):
        super(Encoder, self).__init__()
        modules = []
        in_channels = input_channels
        for h_dim in layers:
            modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        modules.append(nn.Linear(in_channels, output_dim))
        self.encoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, layers: list, output_dim: int):
        super(Decoder, self).__init__()
        modules = []
        in_channels = input_dim
        for h_dim in layers:
            modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        modules.append(nn.Linear(in_channels, output_dim))
        self.decoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class VAE(L.LightningModule):
    VERSION = "1.2"

    def __init__(self, cfg: DictConfig, scaler_aero: Optional[StandardScaler] = None):
        # the scaler is only passed for training else it is not relevant
        super(VAE, self).__init__()
        self.id: str = "none"
        self.input_length: int = cfg.input_length
        self.encoder_layers: list = cfg.encoder_layers
        self.latent_dim: int = cfg.latent_dim
        self.decoder_layers: list = cfg.decoder_layers
        self.kl_weight: float = cfg.kl_weight
        self.kl_similarity_annealing_epochs: int = cfg.kl_similarity_annealing_epochs
        self.reconstruction_weight: float = cfg.reconstruction_weight
        self.aero_similarity_weight: float = cfg.aero_similarity_weight
        self.IoU_weight: float = cfg.IoU_weight
        self.lr: float = cfg.lr

        self._save_learing_curves = False
        self._seed = None
        self.eval_n_clusters = 6
        self._val_epoch_means: list[torch.Tensor] = []
        self._val_epoch_aero: list[torch.Tensor] = []

        # Precompute x-coordinates as buffer
        self.n_points_per_side = 96
        x_coords = np.cosspace(0, 1, self.n_points_per_side)
        self.register_buffer("x_coords", torch.tensor(x_coords, dtype=torch.float32))

        if scaler_aero is not None:
            self.register_buffer(
                "scaler_mean", torch.tensor(scaler_aero.mean_[:7], dtype=torch.float32)
            )  # without the aero feature dims
            self.register_buffer(
                "scaler_scale", torch.tensor(scaler_aero.scale_[:7], dtype=torch.float32)
            )
        else:
            self.scaler_mean = None
            self.scaler_scale = None

        # Encoder returns both mean and logvar (so output dim is latent_dim * 2) --> 2 * latent_dim for mean and logvar
        self.encoder = self.build_encoder(
            self.input_length, self.encoder_layers, self.latent_dim * 2
        )

        # Decoder reconstructs the original signal from latent vector
        self.decoder = self.build_decoder(self.latent_dim, self.decoder_layers, self.input_length)

        self._set_id()

    def _set_id(self):
        if self.id == "none":
            self.id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            pass

    def build_encoder(self, input_channels: int, layers: list, output_dim: int) -> nn.Sequential:
        return Encoder(input_channels, layers, output_dim)

    def build_decoder(self, latent_dim: int, layers: list, output_channels: int) -> nn.Sequential:
        return Decoder(latent_dim, layers, output_channels)

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Pass input through encoder to get mean and log variance.
        """
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=-1)  # Split output
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mean, std^2) using noise
        Allows backpropagation through stochastic variable.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct original signal from latent vector.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full VAE pass: encode → sample → decode
        Returns: reconstructed x, mean, logvar
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def IoU_loss(
        self, x: torch.Tensor, x_recon: torch.Tensor  # (batch_size, af_cst_length)
    ) -> torch.Tensor:
        """
        Compute Intersection over Union (IoU) between two sets of airfoils defined by CST parameters,
        Chord lengths, and twist angles. The rotation is performed around the spar position (spar_pos, 0)
        This is an approx for the shearcenter, wich cannot be computed here directly.
        None that the batch size can also be batch size * n_eval_points.
        Args:
            x: CST parameters of first airfoil (tensor of shape (batch_size, af_cst_length))
            x_recon: CST parameters of second airfoil (tensor of shape (batch_size, af_cst_length))
        Returns:
            IoU_loss: Intersection over Union (tensor of shape (batch_size,))
        """

        intersecting_area, union_area = self.intersection_and_union(x, x_recon)
        iou = intersecting_area / (union_area + 1e-6)

        return 1 - torch.mean(iou)

    def intersection_and_union(
        self, af1_cst: torch.Tensor, af2_cst: torch.Tensor  # (batch_size, af_cst_length)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Intersection over Union (IoU) between two sets of airfoils defined by CST parameters,
        Chord lengths, and twist angles. The rotation is performed around the spar position (spar_pos, 0)
        This is an approx for the shearcenter, wich cannot be computed here directly.
        None that the batch size can also be batch size * n_eval_points.
        Args:
            af1_cst: CST parameters of first airfoil (tensor of shape (batch_size, af_cst_length))
            af2_cst: CST parameters of second airfoil (tensor of shape (batch_size, af_cst_length))
        Returns:
            intersecting_area: Intersection area (tensor of shape (batch_size,))
            union_area: Union area (tensor of shape (batch_size,))
        """
        batch_size = af1_cst.shape[0]

        af1_coords = self.cst_to_y_coords(af1_cst)
        # (batch_size, 2*n_points_per_side-1)
        af2_coords = self.cst_to_y_coords(af2_cst)

        # Use registered buffer x_coords (already on correct device)
        x_common = self.x_coords.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_points)
        af1_y = af1_coords  # (batch_size, 2*n_points_per_side-1)
        af2_y = af2_coords  # (batch_size, 2*n_points_per_side-1)

        # note the x domain is the same for upper and lower surface

        af1_y_upper = torch.flip(af1_y[:, : self.n_points_per_side], [-1])
        # include leading edge point again in lower surface
        af1_y_lower = af1_y[:, self.n_points_per_side - 1 :]  # (batch_size, n_points_per_side)

        af2_y_upper = torch.flip(af2_y[:, : self.n_points_per_side], [-1])
        # include leading edge point again in lower surface
        af2_y_lower = af2_y[:, self.n_points_per_side - 1 :]  # (batch_size, n_points_per_side)

        # Compute intersection and union surfaces
        upper_intersection = torch.min(af1_y_upper, af2_y_upper)  # (batch_size, n_points_per_side)
        lower_intersection = torch.max(af1_y_lower, af2_y_lower)

        intersecting_area = self._compute_area(
            x_common, upper_intersection, lower_intersection
        )  # (batch_size,)
        # xupper is shared as they are the same points
        area_af1 = self._compute_area(x_common, af1_y_upper, af1_y_lower)  # (batch_size,)
        area_af2 = self._compute_area(x_common, af2_y_upper, af2_y_lower)  # (batch_size,)

        union_area = area_af1 + area_af2 - intersecting_area  # (batch_size,)

        return intersecting_area, union_area  # (batch_size,), (batch_size,)

    def _compute_area(
        self, x: torch.Tensor, y_upper: torch.Tensor, y_lower: torch.Tensor
    ) -> torch.Tensor:
        """Compute area between upper and lower surface using trapezoidal rule."""
        dx = x[:, 1:] - x[:, :-1]  # (batch_size, n_points_per_side-1)
        # If y_upper < y_lower (no overlap), height should be 0, not positive.
        hights = F.relu(y_upper - y_lower)  # (batch_size, n_points_per_side)
        area = torch.sum(0.5 * (hights[:, 1:] + hights[:, :-1]) * dx, dim=1)
        return area  # (batch_size,)

    def bernstein_torch(self, i, n, x):
        """Compute Bernstein polynomial with torch operations"""
        binom_coeff = comb(n, i)  # scipy calculation (scalar)
        binom_coeff_torch = torch.tensor(binom_coeff, dtype=x.dtype, device=x.device)
        return binom_coeff_torch * (x**i) * ((1 - x) ** (n - i))

    def cst_to_y_coords(
        self, cst_params: torch.Tensor, N1: float = 0.5, N2: float = 1.0, dz_te: float = 0.0
    ) -> torch.Tensor:
        """
        Convert CST parameters to airfoil coordinates in PyTorch

        Args:
            cst_params: tensor of shape (batch_size, 1 + n_upper + n_lower) containing CST parameters
            n_points_per_side: number of points per side (upper/lower)
            N1, N2: class function exponents
            n_upper: number of upper shape coefficients
            n_lower: number of lower shape coefficients
            dz_te: trailing edge thickness
        Returns:
            y_coords: tensor of shape (batch_size, 2 * n_points_per_side-1) containing y coordinates of airfoil
        """
        batch_size = cst_params.shape[0]

        # Inverse scale if scaler is set
        if self.scaler_mean is not None and self.scaler_scale is not None:
            cst_params = cst_params * self.scaler_scale + self.scaler_mean

        # Extract parameters
        le = cst_params[:, 0]  # leading edge weight
        Au = cst_params[:, 1:4]  # upper weights
        Al = cst_params[:, 4:7]  # lower weights

        n_upper, n_lower = Au.shape[1], Al.shape[1]

        # Use registered buffer x_coords (already on correct device)
        x = self.x_coords.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_points)

        # Class function
        C = x**N1 * (1 - x) ** N2  # (batch_size, n_points)

        # Upper surface shape function
        S_upper = torch.zeros_like(x)
        for i in range(n_upper):
            B_i = self.bernstein_torch(i, n_upper - 1, x)
            S_upper += Au[:, i : i + 1] * B_i

        yu = C * S_upper + x * dz_te / 2
        yu += le.unsqueeze(-1) * x * (1 - x) ** (n_upper + 0.5)

        # Lower surface shape function
        S_lower = torch.zeros_like(x)
        for i in range(n_lower):
            B_i = self.bernstein_torch(i, n_lower - 1, x)
            S_lower += Al[:, i : i + 1] * B_i

        yl = C * S_lower + x * dz_te / 2
        yl += le.unsqueeze(-1) * x * (1 - x) ** (n_lower + 0.5)

        # Combine surfaces
        y_upper = torch.flip(yu, [-1])  # reverse for upper surface
        y_lower = yl[:, 1:]  # exclude first point (leading edge)
        y_full = torch.cat([y_upper, y_lower], dim=1)

        return y_full  # (batch_size, 2*n_points_per_side-1)

    def recon_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss: proportional root mean square relative error (RMSRE) in y-coordinates
        The values are normalized by the squared true values to avoid bias towards thick airfoils
        or fitting only to the thicker parts of the airfoil.
        """
        y_recon = self.cst_to_y_coords(x_recon)
        y_true = self.cst_to_y_coords(x)

        n_points_per_side = self.n_points_per_side

        y_upper_true = y_true[:, :n_points_per_side]
        y_upper_recon = y_recon[:, :n_points_per_side]

        y_lower_true = y_true[:, n_points_per_side:]
        y_lower_recon = y_recon[:, n_points_per_side:]

        # shift so we can normalize without numerical instability at trailing edge
        y_upper_true_shifted = y_upper_true + 1.0
        y_upper_recon_shifted = y_upper_recon + 1.0

        y_lower_true_shifted = y_lower_true - 1.0
        y_lower_recon_shifted = y_lower_recon - 1.0

        # normalize by squared true values
        msre_upper = torch.mean(
            ((y_upper_true_shifted - y_upper_recon_shifted) ** 2)
            / (y_upper_true_shifted**2 + 1e-6)
        )
        msre_lower = torch.mean(
            ((y_lower_true_shifted - y_lower_recon_shifted) ** 2)
            / (y_lower_true_shifted**2 + 1e-6)
        )

        # make to root mean square relative error and multiply by 100 to get percentage
        normalized_mse = (msre_upper + msre_lower) / 2

        return normalized_mse * 10000

    def similarity_loss(self, mean: torch.Tensor, aero_features: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity loss: similar aerodynamics should have similar latent representations
        """
        batch_size = mean.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=mean.device)

        mean_normalized = F.normalize(mean)
        aero_normalized = F.normalize(aero_features)

        latent_similarity = torch.matmul(mean_normalized, mean_normalized.t())
        aero_similarity = torch.matmul(aero_normalized, aero_normalized.t())

        # remove self-similarity
        mask = torch.eye(batch_size, device=mean.device).bool()
        latent_similarity = latent_similarity.masked_fill(mask, 0)
        aero_similarity = aero_similarity.masked_fill(mask, 0)

        loss = F.mse_loss(latent_similarity, aero_similarity)

        return loss

    def get_current_weights(self) -> float:
        """Calculate current KL weight with annealing (gradual increase)"""
        if self.kl_similarity_annealing_epochs > 0:
            # Start from 0.1, gradually increase to full kl_weight
            progress = min(1.0, self.current_epoch / self.kl_similarity_annealing_epochs)
            kl_progress = max(0.1, progress)
            aero_progress = max(0.1, 1 - progress)

            return kl_progress, aero_progress

        return 1.0, 1.0

    def vae_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        aero_features: torch.Tensor,
        stage: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss: reconstruction loss + KL divergence
        """
        # Reconstruction loss (MSE)
        recon_loss = self.recon_loss(x, x_recon)
        # IoU_loss = self.IoU_loss(x, x_recon)
        IoU_loss = self.IoU_loss(x, x_recon)

        current_kl_progress, current_aero_progress = self.get_current_weights()
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        kl_weight = current_kl_progress * self.kl_weight
        # Similarity loss
        similarity_loss = self.similarity_loss(mean, aero_features)
        aero_weight = current_aero_progress * self.aero_similarity_weight

        # Total loss
        total_loss = (
            self.reconstruction_weight * recon_loss
            + kl_weight * kl_loss
            + aero_weight * similarity_loss
            + self.IoU_weight * IoU_loss
        )

        # Log individual components
        self.log(
            f"{stage}_recon_loss",
            self.reconstruction_weight * recon_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )
        self.log(
            f"{stage}_kl_loss",
            kl_weight * kl_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )
        self.log(
            f"{stage}_similarity_loss",
            aero_weight * similarity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )
        self.log(
            f"{stage}_IoU_loss",
            self.IoU_weight * IoU_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )

        # Log total loss for early stopping and monitoring
        self.log(
            f"{stage}_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        if self._save_learing_curves:
            path = Path(f"data/raw/learning_curves_preprocessing_vae/seed_{self._seed}")
            loss_dict = {
                "recon_loss": recon_loss.detach().cpu(),
                "kl_loss": kl_loss.detach().cpu(),
                "similarity_loss": similarity_loss.detach().cpu(),
                "IoU_loss": IoU_loss.detach().cpu(),
                "loss": total_loss.detach().cpu(),
                "kl_weight": kl_weight,
                "aero_weight": aero_weight,
            }
            self._save_learning_curve_csv(path, stage, loss_dict)

        return total_loss

    def save_learning_curves(self, seed: int) -> None:
        self._save_learing_curves = True
        self._seed = seed

    def on_validation_epoch_start(self) -> None:
        self._val_epoch_means = []
        self._val_epoch_aero = []

    def _save_learning_curve_csv(self, filepath: Path, stage: str, metrics: dict) -> None:
        """saves a csv file for each stage (train, val, test) with the learning curve (epoch, loss)"""

        filepath = Path(filepath) / f"{self.id}_{stage}_learning_curve.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not filepath.exists():
            with open(filepath, "w") as f:
                header = "epoch," + ",".join(metrics.keys()) + "\n"
                f.write(header)

        with open(filepath, "a") as f:
            line = f"{self.current_epoch}," + ",".join(f"{v:.6f}" for v in metrics.values()) + "\n"
            f.write(line)

    def _save_latent_metric_csv(self, filepath: Path, metrics: dict) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not filepath.exists():
            with open(filepath, "w") as f:
                header = "epoch," + ",".join(metrics.keys()) + "\n"
                f.write(header)

        with open(filepath, "a") as f:
            line = f"{self.current_epoch}," + ",".join(f"{v:.6f}" for v in metrics.values()) + "\n"
            f.write(line)

    def _compute_latent_eval_metrics(
        self,
        latent_means: npo.ndarray,
        aero_features: npo.ndarray,
    ) -> tuple[float, float, npo.ndarray]:
        n_samples = latent_means.shape[0]
        n_clusters = max(2, min(self.eval_n_clusters, n_samples - 1))
        if n_samples < 3 or n_clusters < 2:
            empty_labels = npo.zeros(n_samples, dtype=int)
            return float("nan"), float("nan"), empty_labels

        labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(
            latent_means
        )
        unique_labels = npo.unique(labels)
        if unique_labels.size < 2:
            return float("nan"), float("nan"), labels

        silhouette = float(silhouette_score(latent_means, labels))

        cluster_variances = []
        for cluster_id in unique_labels:
            cluster_aero = aero_features[labels == cluster_id]
            if cluster_aero.shape[0] < 2:
                continue
            cluster_variances.append(float(npo.mean(npo.var(cluster_aero, axis=0))))

        aero_cluster_variance = (
            float(npo.mean(cluster_variances)) if cluster_variances else float("nan")
        )
        return silhouette, aero_cluster_variance, labels

    def _step(self, batch: torch.Tensor, batch_idx: int, stage: str) -> torch.Tensor:
        x_geom = batch[:, :7]  # First 7 columns (geometric)
        x_aero = batch[:, 7:]  # Last 3 columns (aerodynamic)
        x_recon, mean, logvar = self.forward(x_geom)
        if stage == "val" and not self.trainer.sanity_checking:
            self._val_epoch_means.append(mean.detach().cpu())
            self._val_epoch_aero.append(x_aero.detach().cpu())
        loss = self.vae_loss(x_geom, x_recon, mean, logvar, x_aero, stage)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking or not self._val_epoch_means:
            return

        latent_means = torch.cat(self._val_epoch_means, dim=0).numpy()
        aero_features = torch.cat(self._val_epoch_aero, dim=0).numpy()
        silhouette, aero_cluster_variance, labels = self._compute_latent_eval_metrics(
            latent_means,
            aero_features,
        )

        self.log("val_silhouette", silhouette, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val_aero_cluster_variance",
            aero_cluster_variance,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        if self._save_learing_curves:
            root = Path(f"data/raw/learning_curves_preprocessing_vae/seed_{self._seed}")
            metric_path = root / f"{self.id}_val_latent_metrics.csv"
            latent_dir = root / f"{self.id}_val_latent"
            self._save_latent_metric_csv(
                metric_path,
                {
                    "silhouette": silhouette,
                    "aero_cluster_variance": aero_cluster_variance,
                },
            )
            latent_dir.mkdir(parents=True, exist_ok=True)
            npo.savez(
                latent_dir / f"epoch_{self.current_epoch:04d}.npz",
                latent_means=latent_means,
                aero_features=aero_features,
                cluster_labels=labels,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def save(self, filepath: Path, full_config: DictConfig) -> None:
        """
        Save the VAE model to a checkpoint file.
        Args:
            filepath (Path): Path to save the checkpoint file.
            full_config (DictConfig): Full configuration includeing training params.
        """
        save_dict = {
            "version": self.VERSION,
            "id": self.id,
            "model_state_dict": self.state_dict(),
            "model": self,  # Complete model with architecture
            "config": full_config,
            "model_class": self.__class__,
            "pytorch_version": torch.__version__,
        }
        torch.save(save_dict, filepath)

    @staticmethod
    def load(filepath: Path, add_details: bool = False) -> VAE:
        """
        Load a saved VAE model from a checkpoint file.
        Args:
            filepath (Path): Path to the checkpoint file.
            add_details (bool): If True, print model version and config.
        Returns:
            VAE: The loaded VAE model.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(filepath, weights_only=False, map_location=device)
        model = checkpoint["model"]
        id = checkpoint.get("id", "none")
        if id != model.id if hasattr(model, "id") else "x":
            model_id = model.id if hasattr(model, "id") else "x"
            print(f"Warning: Loaded model ID '{id}' does not match model's ID '{model_id}'.")
        cfg = checkpoint["config"]
        version = checkpoint["version"]

        details = {}
        if add_details:
            details["version"] = version
            details["config"] = cfg
            details["id"] = id
            print(f"Loaded VAE model version: {version}, ID: {id}")

        return model, details
