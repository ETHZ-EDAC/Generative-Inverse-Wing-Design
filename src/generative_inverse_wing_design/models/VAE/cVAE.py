from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_inverse_wing_design.DataGenerator.appconfig import AppConfig
from generative_inverse_wing_design.models.VAE.wing_dataset_VAE import Wing_VAE


class cEmbedding(nn.Module):
    def __init__(
        self,
        input_channels: int,
        layers: list,
        output_dim: int,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
    ):
        super().__init__()
        modules = []
        in_channels = input_channels
        for h_dim in layers:
            modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        modules.append(nn.Sequential(nn.Linear(in_channels, output_dim)))

        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)
        self.scaler_mean: torch.Tensor  # type: ignore
        self.scaler_scale: torch.Tensor  # type: ignore
        self.cembedding = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale input and encode
        x = (x - self.scaler_mean) / self.scaler_scale
        return self.cembedding(x)


class cVAE(Wing_VAE):
    def __init__(self, cfg: DictConfig, scaler_aero: StandardScaler, wing_cfg: AppConfig):
        self.c_prior_layers: list = cfg.c_prior_layers
        self.c_embedding_layers: list = cfg.c_embedding_layers
        self.c_embedding_dim: int = cfg.c_embedding_dim
        self.c_prior_similarity_weight: float = cfg.c_prior_similarity_weight
        self.prior_similarity_temp: float = cfg.prior_similarity_temp
        self.embedded_similarity_weight: float = cfg.embedded_similarity_weight
        self.available_metrics = [
            "L",
            "D",
            "CL",
            "CD",
            "CM",
            "CMa",
            "CLa",
            "max_heave_displacement",
            "max_twist_displacement",
            "max_bending_moment",
            "max_von_mises_stress",
            "mass",
            "CL/CD",
            "max_jacobian_eigenvalue",  # appended later with dataset 2.0
            "min_jacobian_eigenvalue",
        ]
        condition_params = [
            "max_von_mises_stress" if metric == "max_van_mises_stress" else metric
            for metric in cfg.condition_params
        ]
        self.conditioning_idxs = [
            idx for idx, metric in enumerate(self.available_metrics) if metric in condition_params
        ]

        self._save_learing_curves = False
        self._seed = None

        super().__init__(cfg, scaler_aero, wing_cfg)

        # The Encoder got now the full Scaler saved but only the scaler for the features
        # and the conditioning metrics are needed

        encoder_scaler_mean = torch.concatenate(
            [self.scaler_mean[: self.feature_length], torch.zeros(self.c_embedding_dim)]
        )
        # No centering for already embedded conditions
        encoder_scaler_scale = torch.concatenate(
            [self.scaler_scale[: self.feature_length], torch.ones(self.c_embedding_dim)]
        )
        # No scaling for already embedded conditions

        # Override encoder and decoder input dimensions
        self.encoder = self.build_encoder(
            self.feature_length + self.c_embedding_dim,
            self.encoder_layers,
            self.latent_dim * 2,
            encoder_scaler_mean,
            encoder_scaler_scale,
        )

        # Decoder reconstructs the original signal from latent vector
        self.decoder = self.build_decoder(
            self.latent_dim + self.c_embedding_dim,
            self.decoder_layers,
            self.feature_length,
            self.scaler_mean[: self.feature_length],
            self.scaler_scale[: self.feature_length],
        )

        # Conditional Prior Network
        prior_input_dim = len(self.conditioning_idxs)
        # access the scaler for the conditioning metrics only and than select the correct ones
        self.register_buffer(
            "conditional_scaler_mean",
            self.scaler_mean[self.feature_length :][self.conditioning_idxs],
        )
        self.register_buffer(
            "conditional_scaler_scale",
            self.scaler_scale[self.feature_length :][self.conditioning_idxs],
        )
        self.conditional_scaler_mean: torch.Tensor  # type: ignore
        self.conditional_scaler_scale: torch.Tensor  # type: ignore

        self.cprior = cEmbedding(
            prior_input_dim,
            self.c_prior_layers,
            self.latent_dim * 2,
            self.conditional_scaler_mean,
            self.conditional_scaler_scale,
        )

        self.cembedding = cEmbedding(
            prior_input_dim,
            self.c_embedding_layers,
            self.c_embedding_dim,
            self.conditional_scaler_mean,
            self.conditional_scaler_scale,
        )

    def kl_divergence(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        c_mean: torch.Tensor,
        c_logvar: torch.Tensor,
    ) -> torch.Tensor:
        # KL Divergence between two Gaussians
        kl_div = 0.5 * torch.sum(
            c_logvar
            - logvar
            + (torch.exp(logvar) + (mean - c_mean) ** 2) / torch.exp(c_logvar)
            - 1,
            dim=1,
        )
        return torch.mean(kl_div)

    def cvae_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        c_mean: torch.Tensor,
        c_logvar: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:

        num_sampled_params = len(vars(self.wing_cfg.variables).keys())
        geometry_loss = torch.mean(self.geometry_loss(x, x_recon))
        # add direct param recon loss, for initial guidance
        param_recon_mse_loss = self.param_recon_loss(
            x[:, 0:num_sampled_params], x_recon[:, 0:num_sampled_params]
        )
        afs_recon_loss = torch.mean(
            self.afs_recon_loss(x[:, num_sampled_params:], x_recon[:, num_sampled_params:])
        )

        # KL divergence loss
        kl_loss = self.kl_divergence(mean, logvar, c_mean, c_logvar)

        # IoU loss
        iou = torch.mean(self.IoU(x, x_recon))
        iou_loss = 1.0 - iou

        # Annealing weights
        cosine_factor = self._cosine_factor()
        geometry_weight = (
            self.geometry_weight - self.sched_geom_weight_start
        ) * cosine_factor + self.sched_geom_weight_start
        iou_weight = (
            self.iou_weight - self.sched_iou_weight_start
        ) * cosine_factor + self.sched_iou_weight_start

        # Total loss
        total_loss = (
            self.af_recon_weight * afs_recon_loss
            + self.param_recon_weight * param_recon_mse_loss
            + self.kl_weight * kl_loss
            + geometry_weight * geometry_loss
            + iou_weight * iou_loss
        )

        # Log individual components
        self.log(
            f"{stage}_afs_recon_loss",
            afs_recon_loss * self.af_recon_weight,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )
        self.log(
            f"{stage}_param_recon_loss",
            param_recon_mse_loss * self.param_recon_weight,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )
        self.log(
            f"{stage}_geometry_loss",
            geometry_loss * geometry_weight,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )
        self.log(
            f"{stage}_kl_loss",
            kl_loss * self.kl_weight,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )
        self.log(
            f"{stage}_iou_loss",
            iou_loss * iou_weight,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
        )

        if self._save_learing_curves:
            path = Path(f"data/raw/learning_curves_cvae/seed_{self._seed}")
            loss_dict = {
                "afs_recon_loss": afs_recon_loss.detach().cpu(),
                "kl_loss": kl_loss.detach().cpu(),
                "param_recon_loss": param_recon_mse_loss.detach().cpu(),
                "geometry_loss": geometry_loss.detach().cpu(),
                "IoU_loss": iou_loss.detach().cpu(),
                "loss": total_loss.detach().cpu(),
                "geometry_weight": geometry_weight,
                "iou_weight": iou_weight,
            }
            self._save_learning_curve_csv(path, stage, loss_dict)

        return total_loss

    def save_learning_curves(self, seed: int) -> None:
        self._save_learing_curves = True
        self._seed = seed

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

    def embed_conditioning(self, c: torch.Tensor) -> torch.Tensor:
        return self.cembedding(c)

    def learnable_prior(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_mean, c_logvar = self.cprior(c).chunk(2, dim=-1)
        return c_mean, c_logvar

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # type: ignore

        assert (
            x.shape[1] == self.feature_length
        ), f"Input feature length {x.shape[1]} does not match model feature length {self.feature_length}."
        assert c.shape[1] == len(
            self.conditioning_idxs
        ), f"Conditioning feature length {c.shape[1]} does not match expected length {len(self.conditioning_idxs)}."

        c_embedded = self.embed_conditioning(c)

        mean, logvar = self.encode(torch.concatenate([x, c_embedded], dim=1))
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(torch.concatenate([z, c_embedded], dim=1))

        return x_recon, mean, logvar

    def c_prior_similarity_loss(
        self, c: torch.Tensor, c_mean: torch.Tensor, c_logvar: torch.Tensor, stage: str
    ) -> torch.Tensor:

        c_scaled = (c - self.conditional_scaler_mean) / self.conditional_scaler_scale

        # is normalization here necessary?
        c_norm = F.normalize(c_scaled, dim=1)  # Normalize each row to unit length

        sim_matrix = torch.mm(c_norm, c_norm.t())

        diff = c_mean.unsqueeze(1) - c_mean.unsqueeze(
            0
        )  # shape (batch_size, batch_size, latent_dim)
        dist_sq = torch.sum(diff**2, dim=2) + 1e-8  # shape (batch_size, batch_size)

        weights = torch.exp(
            sim_matrix / self.prior_similarity_temp
        )  # shape (batch_size, batch_size)

        # Exclude self-comparisons
        mask = ~torch.eye(c.size(0), dtype=torch.bool, device=c.device)

        # distance is the penalty, weighted by similarity
        loss = (
            torch.sum(dist_sq * weights * mask)
            / (mask.sum() + 1e-8)
            * self.c_prior_similarity_weight
        )

        self.log(
            f"{stage}_c_prior_similarity_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=c.size(0),
        )

        return loss

    def embedded_similarity_loss(
        self, c: torch.Tensor, x_mean: torch.Tensor, x_logvar: torch.Tensor, stage: str
    ) -> torch.Tensor:
        c_scaled = (c - self.conditional_scaler_mean) / self.conditional_scaler_scale

        c_norm = F.normalize(c_scaled, dim=1)  # Normalize each row to unit length
        sim_matrix = torch.mm(c_norm, c_norm.t())
        diff = x_mean.unsqueeze(1) - x_mean.unsqueeze(
            0
        )  # shape (batch_size, batch_size, latent_dim)
        dist_sq = torch.sum(diff**2, dim=2) + 1e-8  # shape (batch_size, batch_size)
        weights = torch.exp(
            sim_matrix / self.prior_similarity_temp
        )  # shape (batch_size, batch_size)

        mask = ~torch.eye(c.size(0), dtype=torch.bool, device=c.device)

        loss = (
            torch.sum(dist_sq * weights * mask)
            / (mask.sum() + 1e-8)
            * self.embedded_similarity_weight
        )

        self.log(
            f"{stage}_embedded_similarity_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=c.size(0),
        )
        return loss

    def _step(self, batch: torch.Tensor, batch_idx: int, stage: str) -> torch.Tensor:
        x_batch = batch[:, : self.feature_length]
        c_batch = batch[:, self.feature_length :][:, self.conditioning_idxs]

        x_recon, mean, logvar = self.forward(x_batch, c_batch)
        c_mean, c_logvar = self.learnable_prior(c_batch)

        c_vae_loss = self.cvae_loss(x_batch, x_recon, mean, logvar, c_mean, c_logvar, stage)

        c_prior_similarity_loss = self.c_prior_similarity_loss(c_batch, c_mean, c_logvar, stage)

        embedded_similarity_loss = self.embedded_similarity_loss(c_batch, mean, logvar, stage)

        loss = c_vae_loss + c_prior_similarity_loss + embedded_similarity_loss

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x_batch.size(0),
        )

        return loss

    def generate_stochastic(self, c: torch.Tensor) -> torch.Tensor:
        """Generate samples from the conditional prior given conditions c.

        Args:
            c (torch.Tensor): Conditioning variables of shape (num_samples, num_conditioning_vars).
        Returns:
            torch.Tensor: Generated samples of shape (num_samples, feature_length).
        """
        c_embedded = self.embed_conditioning(c)
        c_mean, c_logvar = self.learnable_prior(c)
        z = self.reparameterize(c_mean, c_logvar)
        x_generated = self.decode(torch.concatenate([z, c_embedded], dim=1))
        return x_generated

    def generate_deterministic(self, c: torch.Tensor) -> torch.Tensor:
        """Generate samples from the conditional prior given conditions c using the mean.

        Args:
            c (torch.Tensor): Conditioning variables of shape (num_samples, num_conditioning_vars).
        Returns:
            torch.Tensor: Generated samples of shape (num_samples, feature_length).
        """
        c_embedded = self.embed_conditioning(c)
        c_mean, _ = self.learnable_prior(c)
        z = c_mean  # Use the mean for deterministic generation
        x_generated = self.decode(torch.concatenate([z, c_embedded], dim=1))
        return x_generated
