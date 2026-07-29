from __future__ import annotations

import datetime
from pathlib import Path

import aerosandbox.numpy as np
from omegaconf import DictConfig
import pytorch_lightning as L
from scipy.special import comb
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_inverse_wing_design.DataGenerator.appconfig import AppConfig


class Encoder(nn.Module):
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
        self.encoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale input and encode
        x = (x - self.scaler_mean) / self.scaler_scale
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layers: list,
        output_dim: int,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
    ):
        super().__init__()
        modules = []
        in_channels = input_dim
        for h_dim in layers:
            modules.append(nn.Sequential(nn.Linear(in_channels, h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        modules.append(nn.Sequential(nn.Linear(in_channels, output_dim)))
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)
        self.scaler_mean: torch.Tensor  # type: ignore
        self.scaler_scale: torch.Tensor  # type: ignore
        self.decoder = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Decode and inverse scale
        x = self.decoder(x)
        return x * self.scaler_scale + self.scaler_mean


class Wing_VAE(L.LightningModule):
    VERSION: str = "1.0"

    def __init__(self, cfg: DictConfig, scaler_aero: StandardScaler, wing_cfg: AppConfig):
        super().__init__()
        # the scaler is only passed for training else it is not relevant
        self.id: str = "none"
        self.feature_length: int = cfg.feature_length
        self.encoder_layers: list = cfg.encoder_layers
        self.latent_dim: int = cfg.latent_dim
        self.decoder_layers: list = cfg.decoder_layers
        self.kl_weight: float = cfg.kl_weight
        self.af_recon_weight: float = cfg.af_recon_weight
        self.param_recon_weight: float = cfg.param_recon_weight
        self.geometry_weight: float = cfg.geometry_weight
        self.iou_weight: float = cfg.iou_weight
        self.lr: float = cfg.lr

        # Sheduler parameters
        self.sched_T_max: int = cfg.lr_scheduler.T_max
        self.sched_lr_min: float = (
            cfg.lr_scheduler.lr_min if "lr_min" in cfg.lr_scheduler else 0.1 * self.lr
        )
        self.sched_geom_weight_start: float = (
            cfg.lr_scheduler.geom_weight_start
            if "geom_weight_start" in cfg.lr_scheduler
            else 0.1 * self.geometry_weight
        )
        self.sched_iou_weight_start: float = (
            cfg.lr_scheduler.iou_weight_start
            if "iou_weight_start" in cfg.lr_scheduler
            else 0.1 * self.iou_weight
        )
        self.sched_kl_weight_start: float = (
            cfg.lr_scheduler.kl_weight_start
            if "kl_weight_start" in cfg.lr_scheduler
            else 0.1 * self.kl_weight
        )

        self.wing_cfg: AppConfig = wing_cfg

        gaussian_integral_degree = cfg.gaussian_integral_degree
        y_eval, weights = self._gaussian_quadrature_points(
            self.wing_cfg.fixed.span, gaussian_integral_degree
        )
        self.register_buffer("y_eval", y_eval)
        self.y_eval: torch.Tensor  # type: ignore
        self.register_buffer("gaussian_weights", weights)
        self.gaussian_weights: torch.Tensor  # type: ignore

        self._insetion_idxs_lookup_table: torch.Tensor = self._prcompute_insertion_idxs()

        root_chord = torch.tensor(self.wing_cfg.fixed.root_chord, dtype=torch.float32)
        self.register_buffer("root_chord", root_chord)
        self.root_chord: torch.Tensor  # type: ignore

        # Precompute x-coordinates as buffer
        self.n_points_per_side = 96
        x_coords = torch.tensor(np.linspace(0, 1, self.n_points_per_side), dtype=torch.float32)
        self.register_buffer("x_coords", x_coords)
        self.x_coords: torch.Tensor  # type: ignore

        # Scaler buffers for inverse scaling --> needed for af_recon loss
        # get the scaler based on the data module on the training data split
        scaler_mean: torch.Tensor = torch.tensor(scaler_aero.mean_, dtype=torch.float32)
        scaler_scale: torch.Tensor = torch.tensor(scaler_aero.scale_, dtype=torch.float32)
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)
        self.scaler_mean: torch.Tensor  # type: ignore
        self.scaler_scale: torch.Tensor  # type: ignore

        # Encoder returns both mean and logvar (so output dim is latent_dim * 2)
        # --> 2 * latent_dim for mean and logvar
        self.encoder = self.build_encoder(
            self.feature_length,
            self.encoder_layers,
            self.latent_dim * 2,
            self.scaler_mean,
            self.scaler_scale,
        )

        # Decoder reconstructs the original signal from latent vector
        self.decoder = self.build_decoder(
            self.latent_dim,
            self.decoder_layers,
            self.feature_length,
            self.scaler_mean,
            self.scaler_scale,
        )

        self._set_id()
        self._save_learing_curves = False
        self._seed = None

    def _set_id(self):
        if self.id == "none":
            self.id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            pass

    def build_encoder(
        self,
        input_channels: int,
        layers: list,
        output_dim: int,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
    ) -> Encoder:
        return Encoder(input_channels, layers, output_dim, scaler_mean, scaler_scale)

    def build_decoder(
        self,
        latent_dim: int,
        layers: list,
        output_channels: int,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
    ) -> Decoder:
        return Decoder(latent_dim, layers, output_channels, scaler_mean, scaler_scale)

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
        assert (
            x.shape[1] == self.feature_length
        ), f"Input feature length {x.shape[1]} does not match model feature length {self.feature_length}."

        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

    def _gaussian_quadrature_points(
        self, span: float, degree: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Gaussian quadrature points along half-span of the wing.
        Args:
            span: Total wingspan of the wing.
            degree: Number of Gaussian quadrature points.
        Returns:
            y_eval: Tensor of shape (degree,) containing the spanwise positions.
        """
        x, w = np.polynomial.legendre.leggauss(degree)
        # transorm from [-1, 1] to [0, 1]
        # x_i = b-a/2 * node_i + (b+a)/2 = 0.5 * node_i + 0.5
        transformed_nodes = [node * 0.5 + 0.5 for node in x]
        # transform weights according to transformation
        # w_i_normalized = weight_i * (b-a)/2 = weight_i * (1-0)/2 = weight_i * 0.5
        # finally scale to half-span
        # w_i = wi_normalized * (span/2) = weight_i * 0.5 * (span/2) = weight_i * span/4
        transformed_weights = [weight * (span / 4) for weight in w]

        nodes_torch = torch.tensor(transformed_nodes, dtype=torch.float32)  # (degree,)
        weights_torch = torch.tensor(transformed_weights, dtype=torch.float32)  # (degree,)

        return nodes_torch, weights_torch

    def get_af_cst_at_idx(self, afs: torch.Tensor, sec_idx: int) -> torch.Tensor:
        """Extract airfoil CST parameters at a given section index from the full wing representation.
        Args:
            afs: Tensor of shape (batch_size, total_af_cst_length) containing CST parameters for all sections.
            sec_idx: Index of the section to extract.
        Returns:
            Tensor of shape (batch_size, num_cst_params) containing CST parameters for the specified section.
        """

        num_secs = len(self.wing_cfg.fixed.ys_over_half_span)

        num_cst_params = afs.shape[1] / num_secs  # should be 7 for CST representation
        assert num_cst_params.is_integer(), "Input length is not divisible by number of sections."
        num_cst_params = int(num_cst_params)

        num_surface_w = (num_cst_params - 1) / 2
        assert (
            num_surface_w.is_integer()
        ), "Airfoil CST representation has to consist of 1 leading edge parameter\
            + n upper + n lower coefficients."
        num_surface_w = int(num_surface_w)

        start_idx = sec_idx * num_cst_params
        end_idx = start_idx + num_cst_params
        return afs[:, start_idx:end_idx]  # (batch_size, num_cst_params)

    def IoU(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute Intersection over Union (IoU) between reconstructed and original input signals.
        Args:
            recon_x: Reconstructed input signals (tensor of shape (batch_size, feature_length))
            x: Original input signals (tensor of shape (batch_size, feature_length))
        """
        y_eval = self.y_eval  # (n_eval_points,)
        batch_size, _ = x.shape
        device = x.device
        n_quad_points = self.y_eval.shape[0]

        num_sampled_params = len(vars(self.wing_cfg.variables).keys())
        x_params = x[:, 0:num_sampled_params]
        x_recon_params = x_recon[:, 0:num_sampled_params]

        x_chord = x_params[:, 0:3]  # (batch_size, 3)
        x_twist = x_params[:, 3:6]  # (batch_size, 3)
        x_sweep = x_params[:, 6:7]  # (batch_size, 1)
        x_recon_chord = x_recon_params[:, 0:3]
        x_recon_twist = x_recon_params[:, 3:6]
        x_recon_sweep = x_recon_params[:, 6:7]

        chords_at_y_eval = self._chords_at_y_eval(x_chord, y_eval)  # (batch_size, num_eval_points)
        chords_at_y_eval_flat = chords_at_y_eval.view(batch_size * n_quad_points, 1)
        # (batch_size*n_quad_points, 1)
        twists_at_y_eval = self._twists_at_y_eval(x_twist, y_eval)  # (batch_size, num_eval_points)
        twists_at_y_eval_flat = twists_at_y_eval.view(batch_size * n_quad_points, 1)
        # (batch_size*n_quad_points, 1)
        offsets_at_y_eval = self._offsets_at_y_eval(
            x_sweep, chords_at_y_eval, y_eval
        )  # (batch_size, num_eval_points)
        offsets_at_y_eval_flat = offsets_at_y_eval.view(batch_size * n_quad_points, 1)
        # (batch_size*n_quad_points, 1)

        recon_chords_at_y_eval = self._chords_at_y_eval(x_recon_chord, y_eval)
        recon_chords_at_y_eval_flat = recon_chords_at_y_eval.view(batch_size * n_quad_points, 1)
        recon_twists_at_y_eval = self._twists_at_y_eval(x_recon_twist, y_eval)
        recon_twists_at_y_eval_flat = recon_twists_at_y_eval.view(batch_size * n_quad_points, 1)
        recon_offsets_at_y_eval = self._offsets_at_y_eval(
            x_recon_sweep, recon_chords_at_y_eval, y_eval
        )
        recon_offsets_at_y_eval_flat = recon_offsets_at_y_eval.view(batch_size * n_quad_points, 1)

        x_afs = x[:, num_sampled_params:]
        x_recon_afs = x_recon[:, num_sampled_params:]

        assert (
            x_afs.shape == x_recon_afs.shape
        ), "Input and reconstruction must have the same shape."

        num_secs = len(self.wing_cfg.fixed.ys_over_half_span)

        num_cst_params = x_afs.shape[1] / num_secs
        assert num_cst_params.is_integer(), "Input length is not divisible by number of sections."
        num_cst_params = int(num_cst_params)

        # transform to (batch_size, num_secs, num_cst_params)
        x_afs_reshaped = x_afs.view(batch_size, num_secs, num_cst_params)
        x_recon_afs_reshaped = x_recon_afs.view(batch_size, num_secs, num_cst_params)

        # use precomputed lookup table: (n_quad, 3) = [idx1, idx2, frac] ---
        idx1 = self._insetion_idxs_lookup_table[:, 0].to(
            device=device, dtype=torch.long
        )  # (n_quad_points,)
        idx2 = self._insetion_idxs_lookup_table[:, 1].to(
            device=device, dtype=torch.long
        )  # (n_quad_points,)
        frac = self._insetion_idxs_lookup_table[:, 2].to(
            device=device, dtype=torch.float32
        )  # (n_quad_points,)

        # gather sections for all quadrature points: (batch, n_quad, cst_dim)
        cst1 = x_afs_reshaped[:, idx1, :]  # (batch_size, n_quad_points, num_cst_params)
        cst2 = x_afs_reshaped[:, idx2, :]
        recon_cst1 = x_recon_afs_reshaped[:, idx1, :]
        recon_cst2 = x_recon_afs_reshaped[:, idx2, :]

        cst1_reshaped_flat = cst1.view(
            batch_size * n_quad_points, num_cst_params
        )  # (batch_size*n_quad_points, num_cst_params)
        cst2_reshaped_flat = cst2.view(batch_size * n_quad_points, num_cst_params)
        recon_cst1_reshaped_flat = recon_cst1.view(batch_size * n_quad_points, num_cst_params)
        recon_cst2_reshaped_flat = recon_cst2.view(batch_size * n_quad_points, num_cst_params)

        frac_flat = frac.unsqueeze(0).expand(batch_size, -1).reshape(batch_size * n_quad_points, 1)
        # (batch_size*n_quad_points, 1)

        cst_interpolated_flat = self._interpolate_af(
            frac_flat, cst1_reshaped_flat, cst2_reshaped_flat
        )
        # (batch_size*n_quad_points, num_cst_params)
        cst_recon_interpolated_flat = self._interpolate_af(
            frac_flat, recon_cst1_reshaped_flat, recon_cst2_reshaped_flat
        )

        intersection_area_flat, union_area_flat = self.intersection_and_union(
            cst_interpolated_flat,
            chords_at_y_eval_flat,
            twists_at_y_eval_flat,
            offsets_at_y_eval_flat,
            cst_recon_interpolated_flat,
            recon_chords_at_y_eval_flat,
            recon_twists_at_y_eval_flat,
            recon_offsets_at_y_eval_flat,
        )  # (batch_size*n_quad_points,), (batch_size*n_quad_points,)

        intersection_areas = intersection_area_flat.view(
            batch_size, n_quad_points
        )  # (batch_size, n_quad_points)
        union_areas = union_area_flat.view(
            batch_size, n_quad_points
        )  # (batch_size, n_quad_points)

        # Integrate IoU over span using Gaussian quadrature
        weights_expanded = self.gaussian_weights.unsqueeze(0).expand(
            batch_size, -1
        )  # (batch_size, n_quad_points)
        weighted_intersection = (
            intersection_areas * weights_expanded
        )  # (batch_size, n_quad_points)
        weighted_union = union_areas * weights_expanded  # (batch_size, n_quad_points)

        integrated_intersection = torch.sum(weighted_intersection, dim=1)  # (batch_size,)
        integrated_union = torch.sum(weighted_union, dim=1)  # (batch_size,)
        iou = integrated_intersection / (integrated_union + 1e-8)
        return iou  # (batch_size,)

    def _prcompute_insertion_idxs(self):
        """Precompute the insertion indices and interpolation fractions for each evaluation point.
        The list is ordered according to y_eval. and computes the indexes of the closest sections
        as in ys_over_half_span."""

        device = self.y_eval.device
        idx1_list = []
        idx2_list = []
        frac_list = []

        # Convert to numpy for calculation (safe here because it's init)
        ys_np = np.array(self.wing_cfg.fixed.ys_over_half_span)
        y_eval_np = self.y_eval.numpy()

        for y in y_eval_np:
            idx = int(np.searchsorted(ys_np, y))

            # Before or at the first section -> use first section
            if idx == 0:
                idx1_list.append(0)
                idx2_list.append(0)
                frac_list.append(0.0)

            # After or at the last section -> use last section
            elif idx >= len(ys_np):
                last = len(ys_np) - 1
                idx1_list.append(last)
                idx2_list.append(last)
                frac_list.append(0.0)

            # Between sections -> interpolate
            else:
                y0 = ys_np[idx - 1]
                y1 = ys_np[idx]
                frac = (y - y0) / (y1 - y0)
                idx1_list.append(idx - 1)
                idx2_list.append(idx)
                frac_list.append(frac)

        idx1 = torch.tensor(idx1_list, dtype=torch.long, device=device).unsqueeze(
            1
        )  # (n_eval_points, 1)
        idx2 = torch.tensor(idx2_list, dtype=torch.long, device=device).unsqueeze(1)
        frac = torch.tensor(frac_list, dtype=torch.float32, device=device).unsqueeze(1)

        return torch.concat([idx1, idx2, frac], dim=1)  # (n_eval_points, 3)

    def intersection_and_union(
        self,
        af1_cst: torch.Tensor,  # (batch_size, af_cst_length)
        af1_chord: torch.Tensor,  # (batch_size, 1)
        af1_twist: torch.Tensor,  # (batch_size, 1)
        af1_offset: torch.Tensor,  # (batch_size, 1)
        af2_cst: torch.Tensor,
        af2_chord: torch.Tensor,
        af2_twist: torch.Tensor,
        af2_offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Intersection over Union (IoU) between two sets of airfoils defined by CST parameters,
        Chord lengths, and twist angles. The rotation is performed around the spar position (spar_pos, 0)
        This is an approx for the shearcenter, wich cannot be computed here directly.
        None that the batch size can also be batch size * n_eval_points.
        Args:
            af1_cst: CST parameters of first airfoil (tensor of shape (batch_size, af_cst_length))
            af1_chord: chord length of first airfoil (tensor of shape (batch_size, 1))
            af1_twist: twist angle of first airfoil (tensor of shape (batch_size, 1))
            af1_offset: leading edge offest of first airfoil (tensor of shape (batch_size, 1))
            --> same for af2
        """

        af1_coords = self.af_coords_transformed(af1_cst, af1_chord, af1_twist, af1_offset)
        # (batch_size, 2*n_points_per_side-1, 2)
        af2_coords = self.af_coords_transformed(af2_cst, af2_chord, af2_twist, af2_offset)

        af1_x = af1_coords[:, :, 0]  # (batch_size, 2*n_points_per_side-1)
        af1_y = af1_coords[:, :, 1]
        af2_x = af2_coords[:, :, 0]
        af2_y = af2_coords[:, :, 1]

        # note the x domain is the same for upper and lower surface
        af1_x_domain = torch.flip(
            af1_x[:, : self.n_points_per_side], [-1]
        )  # (batch_size, n_points_per_side)
        af1_y_upper = torch.flip(af1_y[:, : self.n_points_per_side], [-1])
        # include leading edge point again in lower surface
        af1_y_lower = af1_y[:, self.n_points_per_side - 1 :]  # (batch_size, n_points_per_side)

        af2_x_domain = torch.flip(
            af2_x[:, : self.n_points_per_side], [-1]
        )  # (batch_size, n_points_per_side)
        af2_y_upper = torch.flip(af2_y[:, : self.n_points_per_side], [-1])
        # include leading edge point again in lower surface
        af2_y_lower = af2_y[:, self.n_points_per_side - 1 :]  # (batch_size, n_points_per_side)

        x_common, upper_intersection, lower_intersection = self._get_union_coords(
            af1_x_domain, af1_y_upper, af1_y_lower, af2_x_domain, af2_y_upper, af2_y_lower
        )  # (batch_size, n_points_per_side), (batch_size, n_points_per_side), (batch_size, n_points_per_side)

        intersecting_area = self._compute_area(
            x_common, upper_intersection, lower_intersection
        )  # (batch_size,)
        # xupper is shared as they are the same points
        area_af1 = self._compute_area(af1_x_domain, af1_y_upper, af1_y_lower)  # (batch_size,)
        area_af2 = self._compute_area(af2_x_domain, af2_y_upper, af2_y_lower)  # (batch_size,)

        union_area = area_af1 + area_af2 - intersecting_area  # (batch_size,)

        return intersecting_area, union_area  # (batch_size,), (batch_size,)

    def _get_union_coords(
        self,
        af1_x_domain: torch.Tensor,
        af1_y_upper: torch.Tensor,
        af1_y_lower: torch.Tensor,
        af2_x_domain: torch.Tensor,
        af2_y_upper: torch.Tensor,
        af2_y_lower: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute union coordinates of two airfoils for IoU calculation.
            Assumes the x-domains of of the airfoils are for upper and lower surface the same.
        Args:
            af1_x_domain: x-coordinates of first airfoil (tensor of shape (batch_size, n_points_per_side))
            af1_y_upper: upper surface y-coordinates of first airfoil (tensor of shape (batch_size, n_points_per_side))
            af1_y_lower: lower surface y-coordinates of first airfoil (tensor of shape (batch_size, n_points_per_side))
            af2_x_domain: x-coordinates of second airfoil (tensor of shape (batch_size, n_points_per_side))
            af2_y_upper: upper surface y-coordinates of second airfoil (tensor of shape (batch_size, n_points_per_side))
            af2_y_lower: lower surface y-coordinates of second airfoil (tensor of shape (batch_size, n_points_per_side))
        Returns:
            tuple of union x-coordinates and corresponding upper and lower y-coordinates
            (batch_size, n_points_per_side), (batch_size, n_points_per_side), (batch_size, n_points_per_side)
        """
        batch_size, _ = af1_x_domain.shape

        # Find common x-domain for integration - consider all points
        af1_x_min = torch.min(af1_x_domain, dim=1).values.unsqueeze(-1)  # (batch_size, 1)
        af1_x_max = torch.max(af1_x_domain, dim=1).values.unsqueeze(-1)
        af2_x_min = torch.min(af2_x_domain, dim=1).values.unsqueeze(-1)
        af2_x_max = torch.max(af2_x_domain, dim=1).values.unsqueeze(-1)

        # Common domain is the intersection
        x_min = torch.max(af1_x_min, af2_x_min)  # (batch_size, 1)
        x_max = torch.min(af1_x_max, af2_x_max)

        has_overlap = (x_max > x_min).float()  # (batch_size, 1)

        x = self.x_coords.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_points_per_side)
        # strech to intervall [x_min, x_max]
        x_common = x_min + (x_max - x_min) * x  # (batch_size, n_points_per_side)

        # still n_points_per_side points, but only in common domain
        af1_y_upper_common = self.torch_interp(
            x_common, af1_x_domain, af1_y_upper
        )  # (batch_size, n_points_per_side)
        af1_y_lower_common = self.torch_interp(x_common, af1_x_domain, af1_y_lower)
        af2_y_upper_common = self.torch_interp(x_common, af2_x_domain, af2_y_upper)
        af2_y_lower_common = self.torch_interp(x_common, af2_x_domain, af2_y_lower)

        # Compute intersection and union surfaces
        upper_intersection = torch.min(
            af1_y_upper_common, af2_y_upper_common
        )  # (batch_size, n_points_per_side)
        lower_intersection = torch.max(af1_y_lower_common, af2_y_lower_common)

        # If no overlap, set intersection surfaces to zero area
        upper_intersection = upper_intersection * has_overlap  # (batch_size, n_points_per_side)
        lower_intersection = lower_intersection * has_overlap

        return x_common, upper_intersection, lower_intersection

    def _compute_area(
        self, x: torch.Tensor, y_upper: torch.Tensor, y_lower: torch.Tensor
    ) -> torch.Tensor:
        """Compute area between upper and lower surface using trapezoidal rule."""
        dx = x[:, 1:] - x[:, :-1]  # (batch_size, n_points_per_side-1)
        # If y_upper < y_lower (no overlap), height should be 0, not positive.
        hights = F.relu(y_upper - y_lower)  # (batch_size, n_points_per_side)
        area = torch.sum(0.5 * (hights[:, 1:] + hights[:, :-1]) * dx, dim=1)
        return area  # (batch_size,)

    def af_coords_transformed(
        self,
        af_cst: torch.Tensor,
        af_chord: torch.Tensor,
        af_twist: torch.Tensor,
        af_offset: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the rotated and shifted coords of an airfoil defined by CST parameters,
        Chord lengths, and twist angles. The rotation is performed around the spar position (spar_pos, 0)
        This is an approx for the shearcenter, wich cannot be computed here directly.
        Args:
            af_cst: CST parameters of first airfoil (tensor of shape (batch_size, af_cst_length))
            af_chord: chord length of first airfoil (tensor of shape (batch_size, 1))
            af_twist: twist angle of first airfoil (tensor of shape (batch_size, 1))
            af_offset: leading edge offest of first airfoil (tensor of shape (batch_size, 1))
        Returns:
            af_coords_final: Transformed airfoil coordinates (tensor of shape (batch_size, n_points, 2))
        """
        batch_size = af_cst.shape[0]
        af_y_coords = self.cst_to_y_coords(af_cst)  # (batch_size, 2*n_points_per_side-1)
        x_coords = self.x_coords.unsqueeze(0).expand(
            batch_size, -1
        )  # (batch_size, n_points_per_side)

        # Combine surfaces
        x_upper = torch.flip(x_coords, [-1])  # reverse for upper surface
        x_lower = x_coords[:, 1:]  # exclude first point (leading edge)
        x_full = torch.cat([x_upper, x_lower], dim=1)  # (batch_size, 2*n_points_per_side-1)

        # Spar position (rotation center)
        spar_pos = af_chord * self.wing_cfg.fixed.spar_position  # (batch_size, 1)

        # Center coordinates around spar position for rotation
        x_centered = x_full * af_chord - spar_pos  # (batch_size, 2*n_points_per_side-1)
        y_centered = af_y_coords * af_chord

        # Stack coordinates for matrix operations
        coords_centered = torch.stack([x_centered, y_centered], dim=-1)
        # (batch_size, 2*n_points_per_side-1, 2)

        # take negative twist angles as the twist is defined nose up
        twist_rad = -torch.deg2rad(af_twist)  # (batch_size, 1)

        # Create rotation matrix
        cos_twist = torch.cos(twist_rad)  # (batch_size, 1)
        sin_twist = torch.sin(twist_rad)  # (batch_size, 1)
        # Build rotation matrix (batch_size, 2, 2)
        rotation_matrix = torch.zeros(batch_size, 2, 2, device=af_cst.device)
        rotation_matrix[:, 0, 0] = cos_twist.squeeze(-1)  # cos
        rotation_matrix[:, 0, 1] = -sin_twist.squeeze(-1)  # -sin
        rotation_matrix[:, 1, 0] = sin_twist.squeeze(-1)  # sin
        rotation_matrix[:, 1, 1] = cos_twist.squeeze(-1)  # cos

        # as we have points horizontal as (x,y) = x^T we compute x^T @ R^T == R @ x
        coords_rotated = torch.bmm(coords_centered, rotation_matrix.transpose(-2, -1))
        # (batch_size, 2*n_points_per_side-1, 2)

        # now shift to ofsett
        # note: as we take the spar pos ar rot center this shift is basically just the spar sweep
        # as the offset is computed as:
        # offsets = slope * self.ys_over_half_span * (self.span / 2) - chords * self.cfg.fixed.spar_position

        # for completeness shift back the rot center and add offset
        # Create the shift vector more clearly
        batch_size, n_points, _ = coords_rotated.shape

        # Shift only in x-direction (spar position + offset)
        x_shift = spar_pos + af_offset  # (batch_size, 1)
        y_shift = torch.zeros_like(x_shift)  # (batch_size, 1) - no y shift

        # Expand to match coordinate dimensions
        x_shift_expanded = x_shift.unsqueeze(1).expand(-1, n_points, -1)
        # (batch_size, 2*n_points_per_side-1, 1)
        y_shift_expanded = y_shift.unsqueeze(1).expand(-1, n_points, -1)
        # (batch_size, 2*n_points_per_side-1, 1)

        # Combine shifts
        shift_vector = torch.cat([x_shift_expanded, y_shift_expanded], dim=-1)
        # (batch_size, 2*n_points_per_side-1, 2)

        # Apply shift
        af_coords_final = coords_rotated + shift_vector

        return af_coords_final  # (batch_size, n_points, 2)

    def _interpolate_af(
        self, x_rel: torch.Tensor, cst1: torch.Tensor, cst2: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate between two CST parameterizations and chord lengths
        Args:
            x_rel: Relative position between airfoil1 and airfoil2 (0.0 = airfoil1, 1.0 = airfoil2)
            cst1: CST parameters of first airfoil (tensor of shape (batch_size, af_cst_length))
            cst2: CST parameters of second airfoil (tensor of shape (batch_size, af_cst_length))
        Returns:
            tuple of interpolated CST parameters and chord length at x_rel
        """
        cst_interpolated = (1 - x_rel) * cst1 + x_rel * cst2

        return cst_interpolated  # (batch_size, af_cst_length)

    def _offsets_at_y_eval(
        self, spar_sweep: torch.Tensor, chords_at_y: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute leading edge offsets at given spanwise relative positions using
        the slope and chord lengths at those positions.
        Args:
            spar_sweep: sweep angle of the spar in degrees (tensor of shape (batch_size, 1))
            chords_at_y: chord lengths at evaluation spanwise positions
            (tensor of shape (batch_size, n_eval_points))
            y: spanwise relative positions (tensor of shape (n_eval_points,))
        Returns:
            offsets: leading edge offsets at evaluation spanwise positions
            (tensor of shape (batch_size, n_eval_points))
        """
        batch_size, n_eval_points = chords_at_y.shape
        y_expanded = y.unsqueeze(0).expand(batch_size, -1)
        # (batch_size, n_eval_points)

        spar_pos = self.wing_cfg.fixed.spar_position  # scalar
        span = self.wing_cfg.fixed.span  # scalar

        # Ensure slope has correct shape for broadcasting
        slope = torch.tan(torch.deg2rad(spar_sweep))  # (batch_size, 1)
        slope_expanded = slope.expand(-1, n_eval_points)  # (batch_size, n_eval_points)

        # Calculate offsets element-wise
        spar_sweep_term = slope_expanded * y_expanded * (span / 2)
        spar_position_term = chords_at_y * spar_pos

        offsets = spar_sweep_term - spar_position_term  # (batch_size, n_eval_points)

        return offsets  # (batch_size, n_eval_points)

    def _twists_at_y_eval(self, twist_params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute twist angles at given spanwise relative positions using
        the the exponent function defined by the root twist, tip twist, mid twist position.
        Args:
            twist_params: twist parameters (tensor of shape (batch_size, 3))
            y: spanwise relative positions (tensor of shape (n_eval_points,))
        Returns:
            twists_at_y_eval: twist angles at evaluation spanwise positions
            (tensor of shape (batch_size, n_eval_points))
        """
        batch_size = twist_params.shape[0]
        r_twist = twist_params[:, 0:1]  # root twist (batch_size, 1)
        t_twist = twist_params[:, 1:2]  # tip twist (batch_size, 1)
        mean_pos_twist = twist_params[:, 2:3]  # mid twist position (batch_size, 1)

        y_expanded = y.unsqueeze(0).expand(batch_size, -1)
        # (batch_size, n_eval_points)
        # mean position is the point at which the twist is halfway between root and tip
        exponent = np.log(0.5) / torch.log(mean_pos_twist + 1e-8)  # (batch_size, 1)
        twist_at_y_eval = r_twist + (t_twist - r_twist) * (y_expanded**exponent)
        return twist_at_y_eval  # (batch_size, n_eval_points)

    def _chords_at_y_eval(self, chords_params: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute chord lengths at given spanwise relative positions using
        the rational quadratic Bezier curve defined by control points and weights.
        Args:
            chords_params: chord parameters (tensor of shape (batch_size, 3))
            y: spanwise relative positions (tensor of shape (n_eval_points,))
        Returns:
            chords_at_y_eval: chord lengths at evaluation spanwise positions
            (tensor of shape (batch_size, n_eval_points))
        """
        batch_size = chords_params.shape[0]
        device = chords_params.device

        # Extract parameters
        ctrl_pos = chords_params[:, 0:1]  # control point position (batch_size, 1)
        w_ctrl = chords_params[:, 1:2]  # control point weight (batch_size, 1)
        c_tip = chords_params[:, 2:3]  # tip chord length (batch_size, 1)
        c_root = self.root_chord.expand_as(c_tip)  # root chord length (batch_size, 1)

        t = torch.linspace(0, 1, 50, device=device)
        t = t.unsqueeze(0).expand(batch_size, -1)  # (batch_size, n_points)
        # Clip weights to prevent division by zero or extreme values
        w = torch.clamp(w_ctrl, 1e-6, 100.0)  # (batch_size, 1)
        # Control points for each batch element
        # p0 = [0.0, c_root], p1 = [c_pos, c_root], p2 = [1.0, c_tip]
        p0_x = torch.zeros_like(c_root)  # (batch_size, 1)
        p0_y = c_root  # (batch_size, 1)

        p1_x = ctrl_pos  # (batch_size, 1)
        p1_y = c_root  # (batch_size, 1)

        p2_x = torch.ones_like(c_tip)  # (batch_size, 1)
        p2_y = c_tip  # (batch_size, 1)

        # Bernstein polynomials
        b0 = self.bernstein_torch(0, 2, t)  # (batch_size, n_points)
        b1 = self.bernstein_torch(1, 2, t)  # (batch_size, n_points)
        b2 = self.bernstein_torch(2, 2, t)  # (batch_size, n_points)

        # Denominator: Sum of weighted Bernstein polynomials
        d = b0 + b1 * w + b2
        # Numerator: Weighted sum of control points
        # X coordinates
        x_num = b0 * p0_x + b1 * w * p1_x + b2 * p2_x  # (batch_size, 50)
        x = x_num / d  # (batch_size, 50)

        # Y coordinates (chord lengths)
        y_num = b0 * p0_y + b1 * w * p1_y + b2 * p2_y  # (batch_size, 50)
        chords = y_num / d  # (batch_size, 50)

        # Interpolate to find chord lengths at desired y_eval positions
        y_expanded = y.unsqueeze(0).expand(batch_size, -1)
        # (batch_size, n_eval_points)
        chords_at_y_eval = self.torch_interp(y_expanded, x, chords)
        # (batch_size, n_eval_points)
        return chords_at_y_eval  # (batch_size, n_eval_points)

    def torch_interp(self, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """
        1D linear interpolation in PyTorch
        Args:
            x: points to interpolate at sorted (tensor of shape (batch_size, n_points))
            xp: known x-coordinates sorted (tensor of shape (batch_size, n_known))
            fp: known y-coordinates (tensor of shape (batch_size, n_known))
        Returns:
            interpolated values at x (tensor of shape (batch_size, n_points))
        """
        batch_size, n_points = x.shape
        _, n_known = xp.shape

        # Clamp x values to valid range [min(xp), max(xp)]
        x_clamped = torch.clamp(x, xp[:, 0:1], xp[:, -1:])  # (batch_size, n_points)

        # Finds next greater index such that xp[idx-1] <= x < xp[idx]
        idx = torch.searchsorted(xp, x_clamped, right=False)  # (batch_size, n_points)

        # clamp indices to valid range [1, n_known-1]
        idx = torch.clamp(idx, 1, n_known - 1)  # (batch_size, n_points)

        # Create batch indices for advanced indexing
        batch_idx = torch.arange(batch_size, device=x.device, dtype=torch.long)[
            :, None
        ]  # (batch_size, 1)

        x0 = xp[batch_idx, idx - 1]  # (batch_size, n_points)
        x1 = xp[batch_idx, idx]  # (batch_size, n_points)
        y0 = fp[batch_idx, idx - 1]  # (batch_size, n_points)
        y1 = fp[batch_idx, idx]  # (batch_size, n_points)

        # Linear interpolation
        alpha = (x_clamped - x0) / (x1 - x0 + 1e-8)
        f_out = y0 + alpha * (y1 - y0)

        return f_out  # (batch_size, n_points)

    def bernstein_torch(self, i: int, n: int, x: torch.Tensor) -> torch.Tensor:
        """Compute Bernstein polynomial with torch operations"""
        binom_coeff = comb(n, i)  # scipy calculation (scalar)
        binom_coeff_torch = torch.tensor(binom_coeff, dtype=x.dtype, device=x.device)
        return binom_coeff_torch * (x**i) * ((1 - x) ** (n - i))

    def cst_to_y_coords(
        self, cst_params: torch.Tensor, N1: float = 0.5, N2: float = 1.0, dz_te: float = 0.0
    ) -> torch.Tensor:
        """
        Convert CST parameters to airfoil coordinates in PyTorch
        assumees cst_params shape is (batch_size, 1 + n_upper + n_lower)
        where n_upper = n_lower
        Args:
            cst_params: tensor of shape (batch_size, 1 + n_upper + n_lower)
            containing CST parameters
            N1, N2: class function exponents
            dz_te: trailing edge thickness
        Returns:
            y_coords: tensor of shape (batch_size, 2 * n_points_per_side-1)
            containing y coordinates of airfoil
        """
        batch_size = cst_params.shape[0]
        num_surface_w = n_lower = (cst_params.shape[1] - 1) / 2
        assert (
            num_surface_w.is_integer()
        ), "CST parameters must consist of 1 leading edge parameter + n upper + n lower coefficients."
        num_surface_w = int(num_surface_w)
        # Extract parameters
        le = cst_params[:, 0]  # leading edge weight
        Au = cst_params[:, 1 : 1 + num_surface_w]  # upper weights
        Al = cst_params[:, 1 + num_surface_w :]  # lower weights

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

        return y_full

    def af_recon_loss(self, cst: torch.Tensor, cst_recon: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss: proportional root mean square relative error (RMSRE) in y-coordinates
        The values are normalized by the squared true values to avoid bias towards thick airfoils
        or fitting only to the thicker parts of the airfoil.
        Args:
            x: Original input tensor of shape (batch_size * num_secs, af_cst_length)
            x_recon: Reconstructed input tensor of shape (batch_size * num_secs, af_cst_length)
        Returns:
            normalized_mse: tensor of shape (batch_size * num_secs,)
        """
        y_recon = self.cst_to_y_coords(cst_recon)  # (batch_size * num_secs, 2*n_points_per_side-1)
        y_true = self.cst_to_y_coords(cst)

        n_points_per_side = self.n_points_per_side

        y_upper_true = y_true[:, :n_points_per_side]  # (batch_size * num_secs, n_points_per_side)
        y_upper_recon = y_recon[:, :n_points_per_side]

        y_lower_true = y_true[
            :, n_points_per_side:
        ]  # (batch_size * num_secs, n_points_per_side-1)
        y_lower_recon = y_recon[:, n_points_per_side:]

        # shift so we can normalize without numerical instability at trailing edge
        y_upper_true_shifted = y_upper_true + 1.0
        y_upper_recon_shifted = y_upper_recon + 1.0

        y_lower_true_shifted = y_lower_true - 1.0
        y_lower_recon_shifted = y_lower_recon - 1.0

        # normalize by squared true values
        msre_upper = torch.mean(
            ((y_upper_true_shifted - y_upper_recon_shifted) ** 2)
            / (y_upper_true_shifted**2 + 1e-6),
            dim=1,
        )  # (batch_size * num_secs,)
        msre_lower = torch.mean(
            ((y_lower_true_shifted - y_lower_recon_shifted) ** 2)
            / (y_lower_true_shifted**2 + 1e-6),
            dim=1,
        )  # (batch_size * num_secs,)

        # make to root mean square relative error
        normalized_mse = (msre_upper + msre_lower) / 2  # (batch_size * num_secs,)

        return normalized_mse

    def afs_recon_loss(self, x_afs: torch.Tensor, x_recon_afs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss: mean squared error in airfoil y-coordinates over all wing sections
        Args:
            x: Original input tensor of shape (batch_size, af_cst_length * num_secs)
            x_recon: Reconstructed input tensor of shape (batch_size, af_cst_length * num_secs)
        Returns:
            loss: tensor of shape (batch_size,)
        """

        assert (
            x_afs.shape == x_recon_afs.shape
        ), "Input and reconstruction must have the same shape."

        batch_size, total_len = x_afs.shape  # (batch_size, af_cst_length * num_secs)
        num_secs = len(self.wing_cfg.fixed.ys_over_half_span)

        num_cst_params = total_len / num_secs
        assert num_cst_params.is_integer(), "Input length is not divisible by number of sections."
        num_cst_params = int(num_cst_params)

        x_afs_reshaped = x_afs.view(batch_size, num_secs, num_cst_params)
        # (batch_size, num_secs, af_cst_length)
        x_recon_afs_reshaped = x_recon_afs.view(batch_size, num_secs, num_cst_params)

        x_flat = x_afs_reshaped.reshape(batch_size * num_secs, num_cst_params)
        # (batch_size * num_secs, af_cst_length)
        x_recon_flat = x_recon_afs_reshaped.reshape(batch_size * num_secs, num_cst_params)

        loss_flat = self.af_recon_loss(x_flat, x_recon_flat)  # (batch_size * num_secs,)

        loss = loss_flat.view(batch_size, num_secs)  # (batch_size, num_secs)

        return torch.mean(loss, dim=1)  # (batch_size,)

    def geometry_loss(self, x: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """
        Reconstruction loss: calculates the reconstruction loss based on the mse
        of the wing geometry points defined at ys_over_half_span spanwise locations.
        Args:
            x: Original input tensor of shape (batch_size, num_params)
            x_recon: Reconstructed input tensor of shape (batch_size, num_params)
        Returns:
            average_mse: tensor of shape (batch_size,)
        """
        num_secs = len(self.wing_cfg.fixed.ys_over_half_span)
        batch_size, _ = x.shape

        assert x.shape == x_recon.shape, "Input and reconstruction must have the same shape."

        ys_over_half_span = torch.tensor(
            self.wing_cfg.fixed.ys_over_half_span, device=x.device, dtype=x.dtype
        )  # (n_eval_points,)

        num_sampled_params = len(vars(self.wing_cfg.variables).keys())
        x_params = x[:, 0:num_sampled_params]  # (batch_size, num_sampled_params)
        x_recon_params = x_recon[:, 0:num_sampled_params]
        x_afs = x[:, num_sampled_params:]  # (batch_size, af_cst_length * num_secs)
        x_recon_afs = x_recon[:, num_sampled_params:]

        num_cst_params = x_afs.shape[1] / num_secs  # should be 7 for CST representation
        assert num_cst_params.is_integer(), "Input length is not divisible by number of sections."
        num_cst_params = int(num_cst_params)

        x_afs_reshaped = x_afs.view(batch_size, num_secs, num_cst_params)
        # (batch_size, num_secs, af_cst_length)
        x_recon_afs_reshaped = x_recon_afs.view(batch_size, num_secs, num_cst_params)
        x_afs_flat = x_afs_reshaped.reshape(batch_size * num_secs, num_cst_params)
        # (batch_size * num_secs, af_cst_length)
        x_recon_afs_flat = x_recon_afs_reshaped.reshape(batch_size * num_secs, num_cst_params)

        x_chord = x_params[:, 0:3]  # (batch_size, 3)
        x_twist = x_params[:, 3:6]  # (batch_size, 3)
        x_sweep = x_params[:, 6:7]  # (batch_size, 1)
        x_recon_chord = x_recon_params[:, 0:3]
        x_recon_twist = x_recon_params[:, 3:6]
        x_recon_sweep = x_recon_params[:, 6:7]

        chords = self._chords_at_y_eval(x_chord, ys_over_half_span)  # (batch_size, num_secs)
        chords_flat = chords.view(batch_size * num_secs, 1)  # (batch_size * num_secs, 1)
        twists = self._twists_at_y_eval(x_twist, ys_over_half_span)
        twists_flat = twists.view(batch_size * num_secs, 1)
        chords_recon = self._chords_at_y_eval(x_recon_chord, ys_over_half_span)
        chords_recon_flat = chords_recon.view(batch_size * num_secs, 1)
        twists_recon = self._twists_at_y_eval(x_recon_twist, ys_over_half_span)
        twists_recon_flat = twists_recon.view(batch_size * num_secs, 1)

        offsets = self._offsets_at_y_eval(
            x_sweep, chords, ys_over_half_span
        )  # (batch_size, num_secs)
        offsets_flat = offsets.view(batch_size * num_secs, 1)  # (batch_size * num_secs, 1)
        offsets_recon = self._offsets_at_y_eval(x_recon_sweep, chords_recon, ys_over_half_span)
        offsets_recon_flat = offsets_recon.view(batch_size * num_secs, 1)

        coords_flat = self.af_coords_transformed(
            x_afs_flat, chords_flat, twists_flat, offsets_flat
        )
        coords_recon_flat = self.af_coords_transformed(
            x_recon_afs_flat, chords_recon_flat, twists_recon_flat, offsets_recon_flat
        )
        coords = coords_flat.view(
            batch_size, num_secs, -1, 2
        )  # (batch_size, num_secs, n_points, 2)
        coords_recon = coords_recon_flat.view(batch_size, num_secs, -1, 2)

        sec_mse = torch.mean((coords - coords_recon) ** 2, dim=[2, 3])  # (batch_size, num_secs)
        average_mse = torch.mean(sec_mse, dim=1)  # (batch_size,)

        return average_mse  # (batch_size,)

    def param_recon_loss(
        self, x_params: torch.Tensor, x_recon_params: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruction loss: mean squared error in wing parameters.
        Uses the scaled parameters only for the mse calculation.
        Args:
            x_params: Original input parameters tensor of shape (batch_size, num_params)
            x_recon_params: Reconstructed input parameters tensor of shape (batch_size, num_params)
        """
        _, num_sampled_params = x_params.shape
        # compute the mse loss of parameters of similar scale
        # as the decoder outputs unscaled values, rescale them first
        mean = self.scaler_mean[:num_sampled_params]
        scale = self.scaler_scale[:num_sampled_params]
        x_params_norm = (x_params - mean) / scale
        x_recon_params_norm = (x_recon_params - mean) / scale

        mse_loss = F.mse_loss(x_params_norm, x_recon_params_norm)
        return mse_loss

    def _cosine_factor(self) -> float:
        """
        Cosine annealing factor in [0,1] based on current_epoch.
        0 at epoch 0 -> 1 at epoch >= T_max.
        """
        t = float(self.current_epoch)
        T_max = float(self.sched_T_max)
        if T_max <= 0:
            return 1.0
        t = min(t, T_max)
        # 0.5 * (1 - cos(pi * t / T_max))
        return 0.5 * (1.0 - np.cos(np.pi * t / T_max))

    def vae_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        stage: str,
    ) -> torch.Tensor:
        """
        Compute VAE loss: reconstruction loss + KL divergence
        """
        # Reconstruction loss (MSE)
        # first 7 values are parameters, rest are airfoil CST coefficients
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
        kl_div = torch.sum(-0.5 * (1 + logvar - mean.pow(2) - logvar.exp()), dim=1)
        kl_loss = torch.mean(kl_div)

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

        self.log(
            f"{stage}_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

        if self._save_learing_curves:
            path = Path(f"data/raw/learning_curves_wing_vae/seed_{self._seed}")
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

    def _step(self, batch: torch.Tensor, batch_idx: int, stage: str) -> torch.Tensor:
        x_recon, mean, logvar = self.forward(batch)
        loss = self.vae_loss(batch, x_recon, mean, logvar, stage)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def save_learning_curves(self, seed: int) -> None:
        self._save_learing_curves = True
        self._seed = seed

    def _save_learning_curve_csv(self, filepath: Path, stage: str, metrics: dict) -> None:
        """Save a csv file for each stage (train, val, test) with the learning curve."""
        filepath = Path(filepath) / f"{self.id}_{stage}_learning_curve.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not filepath.exists():
            with open(filepath, "w") as f:
                header = "epoch," + ",".join(metrics.keys()) + "\n"
                f.write(header)

        with open(filepath, "a") as f:
            line = f"{self.current_epoch}," + ",".join(f"{v:.6f}" for v in metrics.values()) + "\n"
            f.write(line)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.sched_T_max, eta_min=self.sched_lr_min
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # step each epoch
                "monitor": "val_loss",  # not used by this scheduler, but fine
            },
        }

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
    def load(filepath: Path, add_details: bool = False) -> tuple[Wing_VAE, dict]:
        """
        Load a saved VAE model from a checkpoint file.
        Complete model instance is loaded (not only the weights) for proper id handling.
        Args:
            filepath (Path): Path to the checkpoint file.
            add_details (bool): If True, print model version and full training config.
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
        model.to(device)
        cfg = checkpoint["config"]
        version = checkpoint["version"]

        if hasattr(model, "conditioning_idxs"):
            n_loaded_conditions = len(model.conditioning_idxs)
            expected_conditions = 4
            if n_loaded_conditions != expected_conditions:
                raise ValueError(
                    f"Loaded cVAE checkpoint '{filepath}' exposes {n_loaded_conditions} conditioning "
                    f"metrics, but {expected_conditions} are required. "
                    "This usually means the checkpoint is stale or internally inconsistent."
                )

        details = {}
        if add_details:
            details["version"] = version
            details["config"] = cfg
            details["id"] = id
            print(f"Loaded Wing_VAE model version: {version}, ID: {id}")

        return model, details
