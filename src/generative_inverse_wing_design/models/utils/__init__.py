"""Utilities for cVAE training and latent analysis."""

from generative_inverse_wing_design.models.utils.latent_utils import (
    generate_wing_from_conditions,
    latent_space_from_visualization_dataset,
    pca_latent_cosine_similarity,
    plot_latent_std_diagnostics,
    plot_prior_mean_sweeps_in_latent_pca,
    plot_prior_vs_posterior_pca,
    sample_random_conditioning_inputs,
)
from generative_inverse_wing_design.models.utils.learning_curve_utils import (
    plot_cvae_learning_curves_curated,
    plot_learning_curves_mean_std,
    plot_preprocessing_learning_curves_curated,
)

__all__ = [
    "generate_wing_from_conditions",
    "sample_random_conditioning_inputs",
    "latent_space_from_visualization_dataset",
    "pca_latent_cosine_similarity",
    "plot_latent_std_diagnostics",
    "plot_prior_vs_posterior_pca",
    "plot_prior_mean_sweeps_in_latent_pca",
    "plot_learning_curves_mean_std",
    "plot_preprocessing_learning_curves_curated",
    "plot_cvae_learning_curves_curated",
]
