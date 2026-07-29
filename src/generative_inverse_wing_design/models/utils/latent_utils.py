from __future__ import annotations

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch


def generate_wing_from_conditions(
    model,
    data_module,
    batch_idx: int = 0,
    random_state: int = 42,
):
    batch = next(iter(data_module.test_dataloader()))
    # Split batch: all but last 13 columns are features, last 13 are conditions
    batch = batch.to(device=model.device)
    x = batch[:, :-13]
    c = batch[:, -13:]

    # Select only the conditioning indices
    c_cond = c[batch_idx].unsqueeze(0)[:, model.conditioning_idxs]

    # Embed conditions
    c_embedded = model.embed_conditioning(c_cond)

    # Get prior distribution for these conditions
    prior_mean, prior_logvar = model.learnable_prior(c_cond)

    # Sample latent vector from prior
    generator = torch.Generator(device=model.device)
    generator.manual_seed(random_state)
    z = prior_mean + torch.exp(0.5 * prior_logvar) * torch.randn_like(
        prior_mean, generator=generator
    )

    # Generate wing sample
    decoder_input = torch.cat([z, c_embedded], dim=1)
    x_gen = model.decoder(decoder_input)

    return (
        c_cond.detach().cpu().numpy(),
        prior_mean.detach().cpu().numpy(),
        prior_logvar.detach().cpu().numpy(),
        x_gen.detach().cpu().numpy(),
        x.detach().cpu().numpy(),
    )


def latent_space_from_visualization_dataset(
    model,
    data_module,
    num_samples: int = 2000,
    random_state: int = 42,
):
    """
    Use WingDataModule.visualization_dataloader() to:
      - take a random subset of samples (num_samples),
      - run them through the VAE to get latent means,
      - apply t-SNE to 2D,
      - make multiple scatter plots colored by selected performance metrics.

    Assumes Wing_Dataset.load_visualization_subset() appends, in order:
        ["L","D","CL","CD","CM","CMa","CLa",
         "max_heave_displacement","max_twist_displacement",
         "max_bending_moment","max_von_mises_stress","mass"]
    """

    model.eval()
    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    # --- Get full visualization dataset as a single tensor ---
    data_module.load_labels()
    vis_loader = data_module.test_dataloader()

    # Collect all batches (data already includes features + metrics)
    all_data = []
    with torch.no_grad():
        for batch in vis_loader:
            # batch shape: (B, feat_dim + 12_performance_metrics)
            all_data.append(batch)
    if not all_data:
        raise RuntimeError("Visualization dataloader is empty.")

    data_full = torch.cat(all_data, dim=0)  # (N, D_total)
    N, D_total = data_full.shape

    # Last 12 columns are metrics as defined in Wing_Dataset.load_visualization_subset
    n_perf = 13
    if D_total <= n_perf:
        raise RuntimeError(
            f"Expected data to have > {n_perf} columns (features + metrics), got {D_total}."
        )

    # Random subset indices
    rng = np.random.RandomState(random_state)
    n_take = min(num_samples, N)
    idxs = rng.choice(N, size=n_take, replace=False)
    data_subset = data_full[idxs].to(device)  # (n_take, D_total)

    # Split into model input X and performance metrics Y
    X = data_subset[:, :-n_perf]  # standardized input features
    Y = data_subset[:, -n_perf:]  # performance metrics

    # Metric order must match Wing_Dataset.load_visualization_subset
    perf_names = [
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
    ]

    # Indices of the metrics you want to visualize
    selected_metrics = [
        # "L",
        # "D",
        # "max_heave_displacement",
        # "max_twist_displacement",
        "max_von_mises_stress",
        # "CL",
        # "CD",
        # "CM",
        "CMa",
        "mass",
        "CL/CD",
    ]
    metric_indices = {name: perf_names.index(name) for name in selected_metrics}

    # --- Pass through VAE, collect latent means ---
    latents = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, n_take, batch_size):
            end = min(start + batch_size, n_take)
            x_batch = X[start:end][:, : model.feature_length]  # (shape: (B, feat_dim)
            c_batch = Y[start:end][:, model.conditioning_idxs]  # (shape: (B, cond_dim)
            # model(x) -> (recon, mu, logvar)
            _, mu, _ = model(x_batch, c_batch)
            latents.append(mu.cpu().numpy())

    Z = np.concatenate(latents, axis=0)  # (n_take, latent_dim)

    # --- t-SNE to 2D ---
    pca = PCA(n_components=14, random_state=random_state)
    Z_2d = pca.fit_transform(Z)

    # Plot cumulative explained variance
    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.xlabel("Number of PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Prepare metrics as numpy ---
    Y_np = Y.cpu().numpy()  # (n_take, 12)

    # --- Plot one figure per selected metric ---
    for name in selected_metrics:
        mi = metric_indices[name]
        vals = Y_np[:, mi].astype(float)
        mean = np.mean(vals)
        std = np.std(vals)
        mask = np.abs(vals - mean) <= 3 * std  # keep only within 3 sigma

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            Z_2d[mask, 0], Z_2d[mask, 1], vals[mask], c=vals[mask], cmap="viridis", s=10
        )
        fig.colorbar(sc, ax=ax, label=name)
        ax.set_xlabel("PCA dim 1")
        ax.set_ylabel("PCA dim 2")
        ax.set_zlabel(f"Conditioning metric {name}")
        ax.set_title(
            f"Latent Space PCA (3rd dim) colored by {name}\n({np.sum(mask)} samples, 3σ clipped)"
        )
        ax.view_init(elev=30, azim=135)
        plt.tight_layout()
        plt.show()

        fig2 = plt.figure(figsize=(7, 6))
        ax2 = fig2.add_subplot(111)
        sc2 = ax2.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=vals[mask], cmap="viridis", s=10)
        fig2.colorbar(sc2, ax=ax2, label=name)
        ax2.set_xlabel("PCA dim 1")
        ax2.set_ylabel("PCA dim 2")
        ax2.set_title(
            f"Latent Space PCA (2D) colored by {name}\n({np.sum(mask)} samples, 3σ clipped)"
        )
        plt.tight_layout()
        plt.show()


def pca_latent_cosine_similarity(
    model,
    data_module,
    num_samples: int = 2000,
    n_neighbors: int = 25,
    random_state: int = 42,
):
    """
    Plots a 3D PCA of the latent space, colored by cosine similarity to a single random sample.
    """
    model.eval()
    device = model.device if hasattr(model, "device") else next(model.parameters()).device

    # --- Collect samples ---
    data_module.load_labels()
    vis_loader = data_module.test_dataloader()
    all_data = []
    with torch.no_grad():
        for batch in vis_loader:
            all_data.append(batch)
    data_full = torch.cat(all_data, dim=0)
    N = data_full.shape[0]

    # Random subset
    rng = np.random.RandomState(random_state)
    n_take = min(num_samples, N)
    idxs = rng.choice(N, size=n_take, replace=False)
    data_subset = data_full[idxs].to(device)

    # Split into model input X
    n_perf = 13
    X = data_subset[:, :-n_perf]
    Y = data_subset[:, -n_perf:]  # performance metrics
    Y_cond = Y[:, model.conditioning_idxs]
    # --- Pass through VAE, collect latent means ---
    latents = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, n_take, batch_size):
            end = min(start + batch_size, n_take)
            x_batch = X[start:end][:, : model.feature_length]
            c_batch = Y_cond[start:end]
            _, mu, _ = model(x_batch, c_batch)
            latents.append(mu.cpu().numpy())
    Z = np.concatenate(latents, axis=0)  # (n_take, latent_dim)
    Y_cond_np = Y_cond.cpu().numpy()  # (n_take, len(conditioning_idxs))

    # --- PCA to 3D ---
    pca = PCA(n_components=14, random_state=random_state)
    Z_2d = pca.fit_transform(Z)

    # select a random reference index that is visible in the plot
    n_candidates = min(75, Z_2d.shape[0])
    smallest_indices = np.argsort(Z_2d[:, 0])[:n_candidates]
    ref_idx = rng.choice(smallest_indices)

    # --- Cosine similarity ---
    scaler = StandardScaler()
    Y_cond_np_scaled = scaler.fit_transform(Y_cond_np)
    ref_Y = Y_cond_np_scaled[ref_idx : ref_idx + 1]
    cos_sim = cosine_similarity(Y_cond_np_scaled, ref_Y).flatten()

    # --- Plot: Cosine similarity plot (single reference) ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Z_2d[:, 0], Z_2d[:, 1], cos_sim, c=cos_sim, cmap="viridis", s=10, alpha=0.3)
    ax.scatter(
        Z_2d[ref_idx, 0],
        Z_2d[ref_idx, 1],
        cos_sim[ref_idx],
        c="red",
        s=120,
        label="Reference sample",
        marker="x",
        zorder=10,
        linewidths=3,
    )
    fig.colorbar(sc, ax=ax, label="Cosine similarity to reference")
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
    ax.set_zlabel("Similarity")
    ax.set_title(
        "Latent Space PCA (XY) colored by cosine similarity (Z) \nto a random reference sample"
    )
    ax.legend()
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()

    fig2 = plt.figure(figsize=(7, 6))
    ax2 = fig2.add_subplot(111)
    sc = ax2.scatter(Z_2d[:, 0], Z_2d[:, 1], c=cos_sim, cmap="viridis", s=10, alpha=0.3)
    ax2.scatter(
        Z_2d[ref_idx, 0],
        Z_2d[ref_idx, 1],
        c="red",
        s=120,
        label="Reference sample",
        marker="x",
        zorder=10,
        linewidths=3,
    )
    fig2.colorbar(sc, ax=ax2, label="Cosine similarity to reference")
    ax2.set_xlabel("PCA dim 1")
    ax2.set_ylabel("PCA dim 2")
    ax2.set_title(
        "Latent Space PCA (XY) colored by cosine similarity \nto a random reference sample"
    )
    plt.tight_layout()
    plt.show()

    # --- Plot: Multiple neighbors for the single reference ---
    colors = cm.tab10.colors  # 10 distinct colors

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c="lightgray", s=10, alpha=0.3, label="All samples")

    neighbor_indices = np.argsort(-cos_sim)
    neighbor_indices = neighbor_indices[neighbor_indices != ref_idx][:n_neighbors]
    # Plot reference sample
    ax.scatter(
        Z_2d[ref_idx, 0],
        Z_2d[ref_idx, 1],
        c=[colors[0]],
        s=120,
        marker="x",
        label="Reference sample",
        zorder=10,
        linewidths=3,
    )
    # Plot neighbors
    ax.scatter(
        Z_2d[neighbor_indices, 0],
        Z_2d[neighbor_indices, 1],
        c=[colors[0]],
        s=40,
        marker="o",
        label="Most similar samples",
        alpha=0.8,
    )

    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
    ax.set_title(
        f"Latent Space PCA: reference and their {n_neighbors} nearest neighbors\nby conditioning metrics cosine similarity"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_prior_mean_sweeps_in_latent_pca(
    model,
    data_module,
    n_points: int = 15,
    num_samples: int = 4096,
    show_wings: bool = True,
    random_state: int = 42,
):
    # Get all data
    data_module.load_labels()
    vis_loader = data_module.test_dataloader()
    all_data = []
    for batch in vis_loader:
        all_data.append(batch)
    data_full = torch.cat(all_data, dim=0)

    # Get latent means for all samples
    n_perf = 13
    X = data_full[:, :-n_perf]
    Y = data_full[:, -n_perf:]

    # Prepare sweeps with the entire test dataset
    cond_idxs = model.conditioning_idxs
    C = Y[:, cond_idxs].cpu().numpy()
    cond_names = [model.available_metrics[i] for i in cond_idxs]
    C_mean = np.mean(C, axis=0)
    C_min = np.min(C, axis=0)
    C_max = np.max(C, axis=0)
    all_prior_means = []
    all_prior_stds = []
    all_sweep_conditions = []
    for i in range(C.shape[1]):
        sweep = np.linspace(C_min[i], C_max[i], n_points)
        C_sweep = np.tile(C_mean, (n_points, 1))
        C_sweep[:, i] = sweep
        C_sweep_tensor = torch.tensor(C_sweep, dtype=torch.float32, device=model.device)
        with torch.no_grad():
            prior_mean, prior_logvar = model.learnable_prior(C_sweep_tensor)
            prior_std = torch.exp(0.5 * prior_logvar)
        all_prior_means.append(prior_mean.cpu().numpy())
        all_prior_stds.append(prior_std.cpu().numpy())
        all_sweep_conditions.append(C_sweep)
    all_prior_means_flat = np.concatenate(
        all_prior_means, axis=0
    )  # (n_points * n_cond, latent_dim)
    all_prior_stds_flat = np.concatenate(  # noqa: F841
        all_prior_stds, axis=0
    )  # (n_points * n_cond, latent_dim)

    N = data_full.shape[0]
    rng = np.random.RandomState(random_state)
    n_take = min(num_samples, N)
    idxs = rng.choice(N, size=n_take, replace=False)
    data_subset = data_full[idxs].to(model.device)

    # Get latent means for subset of samples
    X = data_subset[:, :-n_perf]
    Y = data_subset[:, -n_perf:]

    latents = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            x_batch = X[start:end][:, : model.feature_length]
            c_batch = Y[start:end][:, model.conditioning_idxs]
            _, mu, _ = model(x_batch, c_batch)
            latents.append(mu.cpu().numpy())
    Z_latent = np.concatenate(latents, axis=0)  # (N, latent_dim)

    # Fit PCA on both latent means and prior means
    pca = PCA(n_components=3, random_state=random_state)
    pca.fit(np.concatenate([Z_latent, all_prior_means_flat], axis=0))
    Z_2d = pca.transform(Z_latent)
    Z_pca_cond_priors = pca.transform(all_prior_means_flat)

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    # exclude yellow as i dont like it
    colors = plt.cm.viridis(np.linspace(0, 1, len(cond_names) + 1))
    linestyles = ["-", "--", "-.", ":"]
    idx = 0
    ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c="lightgray", s=8, alpha=0.7, label="All samples")
    ax.scatter(
        [],
        [],
        marker="s",
        s=100,
        color="gray",
        label="Min value",
        edgecolors="black",
        linewidths=1.5,
    )
    ax.scatter(
        [],
        [],
        marker="^",
        s=100,
        color="gray",
        label="Max value",
        edgecolors="black",
        linewidths=1.5,
    )
    for i, prior_means in enumerate(all_prior_means):
        Z_curve = Z_pca_cond_priors[idx : idx + n_points]
        # min point as rectangle, max point as triangle
        # Plot the line connecting all points
        ax.plot(
            Z_curve[:, 0],
            Z_curve[:, 1],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i % len(colors)],
            linewidth=2,
            label=cond_names[i],
        )
        ax.scatter(
            Z_curve[1:-1, 0],
            Z_curve[1:-1, 1],
            marker="o",
            s=30,
            color=colors[i % len(colors)],
            zorder=5,
        )
        # plot min
        ax.scatter(
            Z_curve[0, 0], Z_curve[0, 1], marker="s", s=50, color=colors[i % len(colors)], zorder=6
        )
        # plot max
        ax.scatter(
            Z_curve[-1, 0],
            Z_curve[-1, 1],
            marker="^",
            s=50,
            color=colors[i % len(colors)],
            zorder=6,
        )

        idx += n_points
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")
    ax.set_title("Learned Prior Mean Sweeps in Latent PCA Space")
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="black", framealpha=1)
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.3, color="black", linewidth=0.5)
    plt.tight_layout()
    # plt.savefig("references/wing_sweeps/Prior_Mean_Sweeps_in_Latent_PCA.svg", dpi=600, format="svg")
    plt.show()
    return
    # Generate and visualize wings at min and max for each conditioning metric
    # if show_wings:
    #     wing_assembly = WingAssembly(model.wing_cfg)

    #     for i, cond_name in enumerate(cond_names):
    #         print(f"\n{'='*60}")
    #         print(f"Conditioning Metric: {cond_name}")
    #         print(f"{'='*60}")

    #         # Get conditions for min and max
    #         C_sweep = all_sweep_conditions[i]
    #         c_min = C_sweep[0]  # First point in sweep (min value)
    #         c_max = C_sweep[-1]  # Last point in sweep (max value)

    #         # Convert to tensors
    #         c_min_tensor = torch.tensor(c_min, dtype=torch.float32, device=model.device).unsqueeze(
    #             0
    #         )
    #         c_max_tensor = torch.tensor(c_max, dtype=torch.float32, device=model.device).unsqueeze(
    #             0
    #         )

    #         # Generate wings
    #         model.eval()
    #         with torch.no_grad():
    #             x_gen_min = model.generate_deterministic(c_min_tensor)
    #             x_gen_max = model.generate_deterministic(c_max_tensor)

    #         x_gen_min_np = x_gen_min.detach().cpu().numpy()[0]
    #         x_gen_max_np = x_gen_max.detach().cpu().numpy()[0]

    #         # Assemble and visualize min wing
    #         try:
    #             print(f"\n--- MIN {cond_name}: {c_min[i]:.4f} ---")
    #             print(f"Conditioning values: {dict(zip(cond_names, c_min))}")
    #             wing_min = wing_assembly.assemble_wing(x_gen_min_np)
    #             wing_min.mechanical_wing.solve_for_properties()
    #             wing_min.solve_all_my_problems()
    #             print(f"Actual performance: {wing_min.performance}")
    #             if "/" in cond_name:
    #                 adj_cond_name = cond_name.replace("/", "_")
    #             wing_min.plot_mechanics_3D(
    #                 include_sweep=True,
    #                 include_deformations=True,
    #                 include_twist=True,
    #                 scale=True,
    #                 pretty_plot=True,
    #                 save_path=f"references/wing_sweeps/MIN_{adj_cond_name}_{c_min[i]:.4f}.svg",
    #             )
    #         except Exception as e:
    #             print(f"Error assembling/visualizing MIN wing for {cond_name}: {e}")

    #         # Assemble and visualize max wing
    #         try:
    #             print(f"\n--- MAX {cond_name}: {c_max[i]:.4f} ---")
    #             print(f"Conditioning values: {dict(zip(cond_names, c_max))}")
    #             wing_max = wing_assembly.assemble_wing(x_gen_max_np)
    #             wing_max.mechanical_wing.solve_for_properties()
    #             wing_max.solve_all_my_problems()
    #             print(f"Actual performance: {wing_max.performance}")
    #             if "/" in cond_name:
    #                 adj_cond_name = cond_name.replace("/", "_")
    #             wing_max.plot_mechanics_3D(
    #                 include_sweep=True,
    #                 include_deformations=True,
    #                 include_twist=True,
    #                 scale=True,
    #                 pretty_plot=True,
    #                 save_path=f"references/wing_sweeps/MAX_{adj_cond_name}_{c_max[i]:.4f}.svg",
    #             )
    #         except Exception as e:
    #             print(f"Error assembling/visualizing MAX wing for {cond_name}: {e}")


def sample_random_conditioning_inputs(
    model,
    data_module,
    n_samples: int = 5,
    random_state: int = 42,
):
    """
    Samples random conditioning inputs from a normal distribution
    using the mean and std of the conditioning metrics in the dataset.

    Returns:
        cond_inputs: torch.Tensor of shape (n_samples, n_cond_inputs)
        cond_names: list of conditioning metric names
        cond_mean: np.ndarray of shape (n_cond_inputs,)
        cond_std: np.ndarray of shape (n_cond_inputs,)
    """

    # Get all conditioning data from the dataset
    data_module.load_labels()
    vis_loader = data_module.test_dataloader()
    all_data = []
    for batch in vis_loader:
        all_data.append(batch)
    data_full = torch.cat(all_data, dim=0)
    n_perf = 13
    Y = data_full[:, -n_perf:]  # (N, n_perf)
    cond_idxs = model.conditioning_idxs
    C = Y[:, cond_idxs].cpu().numpy()  # (N, n_cond)
    cond_names = [model.available_metrics[i] for i in cond_idxs]

    # Compute mean and std for each conditioning variable
    cond_mean = np.mean(C, axis=0)
    cond_std = np.std(C, axis=0)

    # Sample random configurations
    rng = np.random.RandomState(random_state)
    samples = rng.multivariate_normal(mean=cond_mean, cov=np.diag(cond_std**2), size=n_samples)

    cond_inputs = torch.tensor(samples, dtype=torch.float32, device=model.device)

    return cond_inputs, cond_names, cond_mean, cond_std


def plot_latent_std_diagnostics(stats):
    """
    Posterior vs prior std diagnostics using KL(q||p) w.r.t. the learnable prior p(z|c).
    """
    Z_pca = stats["Z_pca"]  # noqa: F841
    Z_std = stats["Z_std"]
    P_std = stats["P_std"]
    Z_mu = stats["Z_mu"]  # noqa: F841
    P_mu = stats["P_mu"]  # noqa: F841
    mean_std = stats["mean_std"]  # noqa: F841
    kl_per_dim = stats["kl_per_dim"]
    kl_per_sample = stats["kl_per_sample"]
    kl_per_sample_per_dim = stats["kl_per_sample_per_dim"]  # noqa: F841
    n_take = stats["n_take"]  # noqa: F841
    latent_dim = Z_std.shape[1]

    # Relative std: σ_q / σ_p  (<1 = encoder is more certain than prior → actively encoding)
    rel_std = Z_std / (P_std + 1e-8)  # (n_take, latent_dim)
    mean_rel_std_per_dim = rel_std.mean(axis=0)  # (latent_dim,)
    active_mask = mean_rel_std_per_dim < 0.95

    # --- 1. Per-dim: posterior std vs prior std + relative std + KL ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 4))

    x = np.arange(latent_dim)
    bar_colors = ["steelblue" if a else "salmon" for a in active_mask]

    axes[0].bar(
        x - 0.2, Z_std.mean(axis=0), width=0.4, label="Posterior σ_q", color=bar_colors, alpha=0.85
    )
    axes[0].bar(
        x + 0.2,
        P_std.mean(axis=0),
        width=0.4,
        label="Prior σ_p",
        color="lightgray",
        alpha=0.85,
        edgecolor="gray",
    )
    axes[0].set_xlabel("Latent dimension", fontsize=12)
    axes[0].set_ylabel("Mean std", fontsize=12)
    axes[0].set_title(
        "Posterior σ_q vs Prior σ_p per dim\n(blue=active: σ_q < 0.95·σ_p)", fontsize=11
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, mean_rel_std_per_dim, color=bar_colors, alpha=0.85)
    axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="σ_q = σ_p")
    axes[1].set_xlabel("Latent dimension", fontsize=12)
    axes[1].set_ylabel("Mean σ_q / σ_p", fontsize=12)
    axes[1].set_title(
        "Relative posterior std (σ_q / σ_p)\n(<1 → encoder adds info beyond prior)", fontsize=11
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    axes[2].bar(x, kl_per_dim, color="darkorange", alpha=0.85)
    axes[2].set_xlabel("Latent dimension", fontsize=12)
    axes[2].set_ylabel("KL(q||p) per dim", fontsize=12)
    axes[2].set_title(
        "KL(q||p) per latent dim w.r.t. learnable prior\n(high KL = dim carries info beyond prior)",
        fontsize=11,
    )
    axes[2].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    print(f"Active dims (σ_q/σ_p < 0.95): {active_mask.sum()} / {latent_dim}")
    print(f"Mean posterior std:      {Z_std.mean():.4f}")
    print(f"Mean prior    std:       {P_std.mean():.4f}")
    print(f"Mean rel std (σ_q/σ_p): {mean_rel_std_per_dim.mean():.4f}")
    print(
        f"Total KL(q||p):         {kl_per_dim.sum():.4f} nats  (mean per sample: {kl_per_sample.mean():.4f})"
    )

    # --- 2. Histograms: std distributions + relative std + per-sample KL ---
    fig, axes = plt.subplots(1, 3, figsize=(19, 4))

    axes[0].hist(Z_std.flatten(), bins=80, color="steelblue", alpha=0.7, label="Posterior σ_q")
    axes[0].hist(P_std.flatten(), bins=80, color="gray", alpha=0.5, label="Prior σ_p")
    axes[0].set_xlabel("std value", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Posterior vs Prior std distribution\n(all dims, all samples)", fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(rel_std.flatten(), bins=80, color="teal", alpha=0.7)
    axes[1].axvline(1.0, color="red", linestyle="--", label="σ_q = σ_p")
    axes[1].set_xlabel("σ_q / σ_p", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title(
        "Relative std (σ_q / σ_p)\n(<1 = encoder more certain than prior)", fontsize=11
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(kl_per_sample, bins=80, color="darkorange", alpha=0.7)
    axes[2].axvline(
        kl_per_sample.mean(), color="red", linestyle="--", label=f"mean={kl_per_sample.mean():.3f}"
    )
    axes[2].set_xlabel("KL(q||p) per sample (summed over dims)", fontsize=12)
    axes[2].set_ylabel("Count", fontsize=12)
    axes[2].set_title("Per-sample total KL(q||p)\nw.r.t. learnable prior", fontsize=11)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_prior_vs_posterior_pca(stats, model, random_state: int = 42):
    """
    3-row, 1-col figure (3D scatter):
      1. Encoder posterior μ_q in PCA space, Z=alpha, colored by alpha
      2. Prior μ_p in PCA space, Z=alpha, colored by alpha
      3. Prior μ_p in PCA space, Z=mean prior σ_p, colored by mean prior σ_p
    + bonus 2D scatter: alpha (x) vs mean prior σ_p (y)
    """
    Z_mu = stats["Z_mu"]
    P_mu = stats["P_mu"]
    P_std = stats["P_std"]
    alpha_ranges = stats["alpha_ranges"]

    mean_prior_std = P_std.mean(axis=1)  # (n_take,)

    # PCA on encoder posterior
    pca_q = PCA(n_components=2, random_state=random_state)
    Z_pca = pca_q.fit_transform(Z_mu)

    # PCA on prior means
    pca_p = PCA(n_components=2, random_state=random_state)
    P_pca = pca_p.fit_transform(P_mu)

    # --- 3-row, 1-col 3D figure ---
    fig = plt.figure(figsize=(10, 28))

    # Row 1: Encoder posterior μ_q, Z=alpha
    ax0 = fig.add_subplot(3, 1, 1, projection="3d")
    sc0 = ax0.scatter(
        Z_pca[:, 0], Z_pca[:, 1], alpha_ranges, c=alpha_ranges, cmap="plasma", s=8, alpha=0.7
    )
    fig.colorbar(sc0, ax=ax0, pad=0.1).set_label("Alpha (OOD-ness)", fontsize=10)
    ax0.set_xlabel("PCA dim 1", fontsize=10)
    ax0.set_ylabel("PCA dim 2", fontsize=10)
    ax0.set_zlabel("Alpha (OOD-ness)", fontsize=10)
    ax0.set_title(
        f"Encoder posterior μ_q\ncolored & Z = alpha (OOD-ness)\n"
        f"(PCA explains {pca_q.explained_variance_ratio_.sum()*100:.1f}% variance)",
        fontsize=11,
    )
    ax0.view_init(elev=25, azim=135)

    # Row 2: Prior μ_p, Z=alpha
    ax1 = fig.add_subplot(3, 1, 2, projection="3d")
    sc1 = ax1.scatter(
        P_pca[:, 0], P_pca[:, 1], alpha_ranges, c=alpha_ranges, cmap="plasma", s=8, alpha=0.7
    )
    fig.colorbar(sc1, ax=ax1, pad=0.1).set_label("Alpha (OOD-ness)", fontsize=10)
    ax1.set_xlabel("PCA dim 1", fontsize=10)
    ax1.set_ylabel("PCA dim 2", fontsize=10)
    ax1.set_zlabel("Alpha (OOD-ness)", fontsize=10)
    ax1.set_title(
        f"Prior μ_p\ncolored & Z = alpha (OOD-ness)\n"
        f"(PCA explains {pca_p.explained_variance_ratio_.sum()*100:.1f}% variance)",
        fontsize=11,
    )
    ax1.view_init(elev=25, azim=135)

    # Row 3: Prior μ_p, Z=mean prior σ_p
    ax2 = fig.add_subplot(3, 1, 3, projection="3d")
    sc2 = ax2.scatter(
        P_pca[:, 0], P_pca[:, 1], mean_prior_std, c=mean_prior_std, cmap="viridis", s=8, alpha=0.7
    )
    fig.colorbar(sc2, ax=ax2, pad=0.1).set_label("Mean prior σ_p", fontsize=10)
    ax2.set_xlabel("PCA dim 1", fontsize=10)
    ax2.set_ylabel("PCA dim 2", fontsize=10)
    ax2.set_zlabel("Mean prior σ_p", fontsize=10)
    ax2.set_title(
        "Prior μ_p\ncolored & Z = mean prior σ_p\n" "(high = prior more uncertain here)",
        fontsize=11,
    )
    ax2.view_init(elev=25, azim=135)

    plt.suptitle(
        "Encoder posterior vs Prior network in latent PCA space\n"
        f"n={Z_mu.shape[0]} test samples",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()

    # --- Bonus: alpha (x) vs mean prior σ_p (y) 2D scatter ---
    corr = np.corrcoef(alpha_ranges, mean_prior_std)[0, 1]
    fig2, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(alpha_ranges, mean_prior_std, s=5, alpha=0.3, c=alpha_ranges, cmap="plasma")
    fig2.colorbar(sc, ax=ax).set_label("Alpha (OOD-ness)", fontsize=10)
    ax.set_xlabel("Alpha contour level (OOD-ness)", fontsize=12)
    ax.set_ylabel("Mean prior σ_p", fontsize=12)
    ax.set_title(
        f"Alpha vs mean prior σ_p  (corr={corr:.3f})\n"
        f"{'→ prior WIDER for OOD ✓' if corr > 0.1 else '→ prior NARROWER for OOD ✗ (overconfident extrapolation)' if corr < -0.1 else '→ no clear trend ~'}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Pearson corr(alpha, mean prior σ_p): {corr:.4f}")
