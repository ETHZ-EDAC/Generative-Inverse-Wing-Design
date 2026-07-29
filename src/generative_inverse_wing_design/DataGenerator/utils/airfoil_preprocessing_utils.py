from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from scipy.special import comb
from shapely.geometry import Polygon


def normalize_airfoil(coords: np.ndarray) -> np.ndarray:
    """Normalize airfoil coordinates to a common reference frame.

    Assumes points are ordered TE upper -> LE -> TE lower.
    Output is scaled to unit chord, translated to LE at x=0, and rotated so the
    chord line is horizontal.
    """
    x_points = coords[:, 0]
    y_points = coords[:, 1]

    dist_to_le = np.sqrt((x_points - x_points[0]) ** 2 + (y_points - y_points[0]) ** 2)
    le_idx = np.argmax(dist_to_le)
    chord_length = dist_to_le[le_idx]

    x_points = x_points / chord_length
    y_points = y_points / chord_length

    x_points = x_points - x_points[le_idx]
    y_points = y_points - y_points[le_idx]

    angle = np.arctan2(y_points[0] - y_points[le_idx], x_points[0] - x_points[le_idx])
    coords_3d = np.column_stack([x_points, y_points, np.zeros_like(x_points)])
    rotated = np.array(
        [R.from_euler("z", -angle, degrees=False).apply(point) for point in coords_3d]
    )

    x_points_norm = np.clip(rotated[:, 0], 0, 1)
    y_points_norm = rotated[:, 1]
    return np.column_stack([x_points_norm, y_points_norm])


def is_trailing_edge_closed(points: np.ndarray, tolerance: float = 0.01) -> bool:
    """Return True if normalized trailing-edge endpoints are within tolerance."""
    points = normalize_airfoil(points)

    x_points = points[:, 0]
    te_mask = np.abs(x_points - 1.0) < 0.001
    te_points = points[te_mask]

    if len(te_points) < 2:
        return False

    te_distance = np.linalg.norm(te_points[0] - te_points[-1])
    return bool(te_distance <= tolerance)


def calculate_point_grid_iou_polygon(
    points1: np.ndarray,
    points2: np.ndarray,
    buffer_radius: float = 0.001,
) -> float:
    """Compute IoU between two airfoil point clouds via polygon approximation."""
    try:
        points1 = normalize_airfoil(points1)
        points2 = normalize_airfoil(points2)

        poly1 = Polygon(points1).buffer(buffer_radius)
        poly2 = Polygon(points2).buffer(buffer_radius)

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        if union == 0:
            return 0.0
        return float(intersection / union)
    except Exception:
        return 0.0


def cst_airfoil(
    x: np.ndarray,
    Au: np.ndarray,
    Al: np.ndarray,
    le: float = 0,
    N1: float = 0.5,
    N2: float = 1.0,
    dz_te: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate upper/lower surfaces using CST parameterization."""
    n_upper = len(Au)
    n_lower = len(Al)

    class_fn = x**N1 * (1 - x) ** N2

    def bernstein(i: int, n: int, xvals: np.ndarray) -> np.ndarray:
        return comb(n, i) * xvals**i * (1 - xvals) ** (n - i)

    shape_upper = np.zeros_like(x)
    for i in range(n_upper):
        shape_upper += Au[i] * bernstein(i, n_upper - 1, x)

    yu = class_fn * shape_upper + x * dz_te / 2
    yu += le * x * (1 - x) ** (n_upper + 0.5)

    shape_lower = np.zeros_like(x)
    for i in range(n_lower):
        shape_lower += Al[i] * bernstein(i, n_lower - 1, x)

    yl = class_fn * shape_lower + x * dz_te / 2
    yl += le * x * (1 - x) ** (n_lower + 0.5)

    return yu, yl


def fit_cst_to_airfoil(
    airfoil_points: np.ndarray, n_cst: int = 3
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit CST coefficients and LE weight to an airfoil point cloud."""
    airfoil_points = normalize_airfoil(airfoil_points)
    x_points = airfoil_points[:, 0]
    y_points = airfoil_points[:, 1]
    le_idx = np.argmin(x_points)

    x_upper = x_points[: le_idx + 1][::-1]
    y_upper = y_points[: le_idx + 1][::-1]

    x_lower = x_points[le_idx:]
    y_lower = y_points[le_idx:]

    def objective(params: np.ndarray) -> np.ndarray:
        Au = params[:n_cst]
        Al = params[n_cst:-1]
        le = params[-1]

        yu_fit, _ = cst_airfoil(x_upper, Au, Al, le)
        _, yl_fit = cst_airfoil(x_lower, Au, Al, le)

        residual_upper = yu_fit - y_upper
        residual_lower = yl_fit - y_lower
        return np.concatenate([residual_upper, residual_lower])

    initial_params = np.zeros(2 * n_cst + 1)
    result = least_squares(objective, initial_params)

    Au = result.x[:n_cst]
    Al = result.x[n_cst:-1]
    le = float(result.x[-1])
    return Au, Al, le


def assess_cluster_quality_simple(airfoil_df, cluster_labels, dim_per_side: int):
    """Compute average within-cluster pairwise distances for geometry/aero/performance features."""
    from scipy.spatial.distance import pdist

    geom_cols = (
        ["asb_leading_edge_weight"]
        + [f"asb_upper_{i}" for i in range(dim_per_side)]
        + [f"asb_lower_{i}" for i in range(dim_per_side)]
    )
    aero_cols = ["CL", "CD", "CM"]
    perf_cols = ["CL/CD"]

    all_geom_distances = []
    all_aero_distances = []
    all_perf_distances = []

    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_data = airfoil_df[cluster_mask]

        if len(cluster_data) < 2:
            continue

        all_geom_distances.extend(pdist(cluster_data[geom_cols].values))
        all_aero_distances.extend(pdist(cluster_data[aero_cols].values))
        all_perf_distances.extend(pdist(cluster_data[perf_cols].values))

    avg_cst_distance = float(np.mean(all_geom_distances)) if all_geom_distances else 0.0
    avg_aero_distance = float(np.mean(all_aero_distances)) if all_aero_distances else 0.0
    avg_perf_distance = float(np.mean(all_perf_distances)) if all_perf_distances else 0.0

    return avg_cst_distance, avg_aero_distance, avg_perf_distance


def read_dat_airfoil_file(filepath) -> np.ndarray | None:
    """Read a .dat airfoil file and return Nx2 coordinates."""
    try:
        coordinates = []
        with open(filepath, "r") as file:
            lines = file.readlines()

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    coordinates.append([float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                continue

        return np.array(coordinates)
    except Exception:
        return None


def load_all_dat_files(folder_path: str | Path):
    """Load all `.dat` airfoil files from a folder and return coordinates and names."""
    current_dir = Path.cwd()
    folder_path = current_dir.parent / Path(folder_path)

    dat_files = list(folder_path.glob("*.dat"))
    if not dat_files:
        return np.array([], dtype=object), np.array([])

    airfoils_coordinates = []
    airfoil_names = []

    for dat_file in dat_files:
        coordinates = read_dat_airfoil_file(dat_file)
        if coordinates is not None and len(coordinates) > 0:
            airfoils_coordinates.append(coordinates)
            airfoil_names.append(dat_file.stem)

    return np.array(airfoils_coordinates, dtype=object), np.array(airfoil_names)


def assign_hex_clusters(
    x_points: np.ndarray, y_points: np.ndarray, hex_centers: np.ndarray
) -> np.ndarray:
    """Assign each point to the nearest hex-center index."""
    hex_clusters = []
    for x, y in zip(x_points, y_points):
        distances = np.sqrt((hex_centers[:, 0] - x) ** 2 + (hex_centers[:, 1] - y) ** 2)
        hex_clusters.append(np.argmin(distances))
    return np.array(hex_clusters)


def map_dataset_to_latent_space(model, scaler, dataset_df, feature_columns):
    """Map dataset rows to VAE latent means/logvars."""
    import torch

    model.eval()
    data = dataset_df[feature_columns].values.astype("float32")
    scaled_data = scaler.transform(data)
    data_tensor = torch.tensor(scaled_data).to(model.device)

    with torch.no_grad():
        latent_means, latent_logvars = model.encode(data_tensor)
        latent_means_np = latent_means.cpu().numpy()
        latent_logvars_np = latent_logvars.cpu().numpy()

    return latent_means_np, latent_logvars_np, scaled_data


def visualize_latent_space(
    latent_means: np.ndarray, dataset_df, title_prefix: str = "VAE Latent Space"
):
    """Create standard latent-space diagnostic scatter plots."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].scatter(latent_means[:, 0], latent_means[:, 1], alpha=0.6, s=10)
    axes[0, 0].set_title(f"{title_prefix} - All Points")
    axes[0, 0].set_xlabel("Latent Dimension 1")
    axes[0, 0].set_ylabel("Latent Dimension 2")
    axes[0, 0].grid(True, alpha=0.3)

    if "CL" in dataset_df.columns:
        scatter1 = axes[0, 1].scatter(
            latent_means[:, 0],
            latent_means[:, 1],
            c=dataset_df["CL"],
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        axes[0, 1].set_title(f"{title_prefix} - Colored by CL")
        axes[0, 1].set_xlabel("Latent Dimension 1")
        axes[0, 1].set_ylabel("Latent Dimension 2")
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0, 1], label="CL")

    if "CD" in dataset_df.columns:
        scatter2 = axes[0, 2].scatter(
            latent_means[:, 0],
            latent_means[:, 1],
            c=dataset_df["CD"],
            cmap="plasma",
            s=10,
            alpha=0.7,
        )
        axes[0, 2].set_title(f"{title_prefix} - Colored by CD")
        axes[0, 2].set_xlabel("Latent Dimension 1")
        axes[0, 2].set_ylabel("Latent Dimension 2")
        axes[0, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[0, 2], label="CD")

    if "CL" in dataset_df.columns and "CD" in dataset_df.columns:
        ld_ratio = dataset_df["CL"] / np.where(dataset_df["CD"] == 0, 1e-6, dataset_df["CD"])
        scatter3 = axes[1, 0].scatter(
            latent_means[:, 0], latent_means[:, 1], c=ld_ratio, cmap="coolwarm", s=10, alpha=0.7
        )
        axes[1, 0].set_title(f"{title_prefix} - Colored by L/D")
        axes[1, 0].set_xlabel("Latent Dimension 1")
        axes[1, 0].set_ylabel("Latent Dimension 2")
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[1, 0], label="L/D")

    if "asb_leading_edge_weight" in dataset_df.columns:
        scatter4 = axes[1, 1].scatter(
            latent_means[:, 0],
            latent_means[:, 1],
            c=dataset_df["asb_leading_edge_weight"],
            cmap="RdYlBu",
            s=10,
            alpha=0.7,
        )
        axes[1, 1].set_title(f"{title_prefix} - Colored by LE Weight")
        axes[1, 1].set_xlabel("Latent Dimension 1")
        axes[1, 1].set_ylabel("Latent Dimension 2")
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=axes[1, 1], label="LE Weight")

    if "cluster" in dataset_df.columns:
        scatter5 = axes[1, 2].scatter(
            latent_means[:, 0],
            latent_means[:, 1],
            c=dataset_df["cluster"],
            cmap="viridis",
            s=10,
            alpha=0.7,
        )
        axes[1, 2].set_title(f"{title_prefix} - Colored by Clusters")
        axes[1, 2].set_xlabel("Latent Dimension 1")
        axes[1, 2].set_ylabel("Latent Dimension 2")
        axes[1, 2].grid(True, alpha=0.3)
        plt.colorbar(scatter5, ax=axes[1, 2], label="Cluster ID")
    else:
        axes[1, 2].hexbin(latent_means[:, 0], latent_means[:, 1], gridsize=30, cmap="Blues")
        axes[1, 2].set_title(f"{title_prefix} - Density")
        axes[1, 2].set_xlabel("Latent Dimension 1")
        axes[1, 2].set_ylabel("Latent Dimension 2")

    plt.tight_layout()
    plt.show()


def analyze_latent_distributions_l2(latent_means: np.ndarray, latent_logvars: np.ndarray):
    """Plot latent distance/variance histograms and return >3σ outlier masks."""
    import matplotlib.pyplot as plt

    centroid = np.mean(latent_means, axis=0)
    l2_distances = np.linalg.norm(latent_means - centroid, axis=1)

    mean_l2_distance = np.mean(l2_distances)
    std_l2_distance = np.std(l2_distances)

    var_magnitudes = np.exp(latent_logvars)
    var_magnitudes = np.sqrt(np.sum(var_magnitudes**2, axis=1))
    mean_var = np.mean(var_magnitudes)
    std_var = np.std(var_magnitudes)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].hist(l2_distances, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0].axvline(
        mean_l2_distance,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_l2_distance:.3f}",
    )
    axes[0].axvline(
        mean_l2_distance + 3 * std_l2_distance,
        color="blue",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"+3σ: {mean_l2_distance + 3 * std_l2_distance:.3f}",
    )
    axes[0].set_title("Distance to Centroid Distribution of Latent embeddings", fontsize=14)
    axes[0].set_xlabel("Distance to Centroid (L2 Norm)")
    axes[0].set_ylabel("Samples")
    axes[0].legend(loc="upper right", bbox_to_anchor=(0.95, 0.75))
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(var_magnitudes, bins=50, alpha=0.7, color="lightgreen", edgecolor="black")
    axes[1].axvline(
        mean_var, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_var:.3f}"
    )
    axes[1].axvline(
        mean_var + 3 * std_var,
        color="blue",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"+3σ: {mean_var + 3 * std_var:.3f}",
    )
    axes[1].set_title("Var Magnitude Distribution of Latent Embeddings", fontsize=14)
    axes[1].set_xlabel("Var Magnitude (L2 Norm)")
    axes[1].set_ylabel("Samples")
    axes[1].legend(loc="upper right", bbox_to_anchor=(0.95, 0.75))
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    distance_outlier_mask = l2_distances > (mean_l2_distance + 3 * std_l2_distance)
    variance_outlier_mask = var_magnitudes > (mean_var + 3 * std_var)

    return distance_outlier_mask, variance_outlier_mask


def sample_from_latent_grid(model, scaler, feature_columns, grid_size: int = 5):
    """Decode a regular grid in latent space back to feature space."""
    import torch

    model.eval()
    latent_range = 3.0
    x = np.linspace(-latent_range, latent_range, grid_size)
    y = np.linspace(-latent_range, latent_range, grid_size)
    xx, yy = np.meshgrid(x, y)

    grid_points = np.column_stack([xx.ravel(), yy.ravel()])
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(model.device)

    with torch.no_grad():
        decoded = model.decode(grid_tensor)
        decoded_np = decoded.cpu().numpy()
        decoded_original = scaler.inverse_transform(decoded_np)

    return grid_points, decoded_original, xx, yy


def visualize_latent_grid_sampling(
    grid_points, decoded_data, xx, yy, feature_columns, grid_size: int = 5
):
    """Visualize decoded airfoils on a latent-grid layout."""
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    feature_to_idx = {name: idx for idx, name in enumerate(feature_columns)}

    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            data_point = decoded_data[idx]
            try:
                uw = [data_point[feature_to_idx[f"asb_upper_{k}"]] for k in range(3)]
                lw = [data_point[feature_to_idx[f"asb_lower_{k}"]] for k in range(3)]
                le = data_point[feature_to_idx["asb_leading_edge_weight"]]

                coords = get_kulfan_coordinates(
                    lower_weights=np.array(lw),
                    upper_weights=np.array(uw),
                    leading_edge_weight=le,
                    n_points_per_side=96,
                    TE_thickness=0,
                )

                axes[i, j].plot(coords[:, 0], coords[:, 1], "b-", linewidth=1)
                axes[i, j].fill(coords[:, 0], coords[:, 1], alpha=0.3)
                axes[i, j].set_xlim(-0.1, 1.1)
                axes[i, j].set_ylim(-0.3, 0.3)
                axes[i, j].set_aspect("equal")
                axes[i, j].grid(True, alpha=0.3)

                z1, z2 = grid_points[idx]
                axes[i, j].set_title(f"({z1:.1f}, {z2:.1f})", fontsize=8)

                cl = data_point[feature_to_idx["CL"]]
                cd = data_point[feature_to_idx["CD"]]
                ld_ratio = cl / cd if cd != 0 else 0

                axes[i, j].text(
                    0.02,
                    0.98,
                    f"L/D:{ld_ratio:.1f}",
                    transform=axes[i, j].transAxes,
                    fontsize=6,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )
            except Exception:
                axes[i, j].text(0.5, 0.5, "Error", ha="center", va="center")
                axes[i, j].set_title(
                    f"Error at ({grid_points[idx][0]:.1f}, {grid_points[idx][1]:.1f})"
                )

    plt.suptitle("Airfoils Sampled from 2D Latent Space Grid", fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_outlier_airfoils(
    df,
    outlier_mask,
    latent_means,
    dim_per_side: int = 3,
    n_points_per_side: int = 96,
    max_airfoils: int = 10,
):
    """Plot outlier airfoils from a boolean outlier mask."""
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
    import matplotlib.pyplot as plt

    outliers_df = df[outlier_mask].copy()
    if len(outliers_df) == 0:
        return outliers_df

    n_samples = min(max_airfoils, len(outliers_df))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes if n_samples > 1 else [axes]
    else:
        axes = axes.flatten()

    centroid = np.mean(latent_means, axis=0)
    l2_distances = np.linalg.norm(latent_means - centroid, axis=1)
    mean_distance = np.mean(l2_distances)
    std_distance = np.std(l2_distances)

    for i in range(n_samples):
        outlier_data = outliers_df.iloc[i]
        original_idx = outliers_df.index[i]
        uw = [outlier_data[f"asb_upper_{j}"] for j in range(dim_per_side)]
        lw = [outlier_data[f"asb_lower_{j}"] for j in range(dim_per_side)]
        le = outlier_data["asb_leading_edge_weight"]

        coords = get_kulfan_coordinates(
            lower_weights=np.array(lw),
            upper_weights=np.array(uw),
            leading_edge_weight=le,
            TE_thickness=0,
            n_points_per_side=n_points_per_side,
        )

        axes[i].plot(coords[:, 0], coords[:, 1], "b-", linewidth=2)
        axes[i].fill(coords[:, 0], coords[:, 1], alpha=0.3, color="lightblue")
        axes[i].set_xlim(-0.1, 1.1)
        axes[i].set_ylim(-0.4, 0.4)
        axes[i].set_aspect("equal")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(
            f"Outlier Airfoil {original_idx}, Latent: {latent_means[original_idx].round(2)}",
            fontsize=12,
        )
        axes[i].set_xlabel("x/c")
        axes[i].set_ylabel("y/c")

    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"Outlier Airfoils | μ={mean_distance:.3f}, σ={std_distance:.3f}, +3σ={mean_distance + 3 * std_distance:.3f}",
        fontsize=14,
        y=0.93,
    )
    plt.tight_layout()
    plt.show()

    return outliers_df


def visualize_airfoil_reconstruction(
    model, scaler, airfoil_data, title: str = "Airfoil Reconstruction"
):
    """Plot original CST airfoil and reconstruction from VAE latent mean."""
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
    import matplotlib.pyplot as plt
    import torch

    if hasattr(airfoil_data, "iloc"):
        le = airfoil_data["asb_leading_edge_weight"]
        uw = [airfoil_data[f"asb_upper_{i}"] for i in range(3)]
        lw = [airfoil_data[f"asb_lower_{i}"] for i in range(3)]
    else:
        le = airfoil_data["asb_leading_edge_weight"]
        uw = [airfoil_data[f"asb_upper_{i}"] for i in range(3)]
        lw = [airfoil_data[f"asb_lower_{i}"] for i in range(3)]

    coords_original = get_kulfan_coordinates(
        lower_weights=np.array(lw),
        upper_weights=np.array(uw),
        leading_edge_weight=le,
        TE_thickness=0,
        n_points_per_side=96,
    )

    model.eval()
    features = np.array([le] + uw + lw).reshape(1, -1).astype("float32")
    features_scaled = scaler.transform(features)
    features_tensor = torch.tensor(features_scaled).to(model.device)

    with torch.no_grad():
        latent_mean, _ = model.encode(features_tensor)
        reconstructed = model.decode(latent_mean)
        reconstructed_np = reconstructed.cpu().numpy()
        reconstructed_original = scaler.inverse_transform(reconstructed_np)[0]

    le_rec = reconstructed_original[0]
    uw_rec = reconstructed_original[1:4]
    lw_rec = reconstructed_original[4:7]

    coords_reconstructed = get_kulfan_coordinates(
        lower_weights=lw_rec,
        upper_weights=uw_rec,
        leading_edge_weight=le_rec,
        TE_thickness=0,
        n_points_per_side=96,
    )

    mse = np.mean((coords_original[:, 1] - coords_reconstructed[:, 1]) ** 2)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(12, 8))
    plt.plot(
        coords_original[:, 0],
        coords_original[:, 1],
        "b-",
        linewidth=2,
        label="Original",
        alpha=0.8,
    )
    plt.plot(
        coords_reconstructed[:, 0],
        coords_reconstructed[:, 1],
        "r--",
        linewidth=2,
        label="VAE Reconstruction",
        alpha=0.8,
    )
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    plt.title(f"{title}\nRMSE: {rmse:.6f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return rmse, latent_mean.cpu().numpy()[0]


def compute_p_x_given_k(gm, X: np.ndarray) -> np.ndarray:
    """Compute p(x|k) values for each sample under each GMM component."""
    from scipy.stats import multivariate_normal

    n_samples, _ = X.shape
    n_comp = gm.means_.shape[0]
    p_x_given_k = np.zeros((n_samples, n_comp))
    for k in range(n_comp):
        mu = gm.means_[k]
        cov = gm.covariances_[k]
        p_x_given_k[:, k] = multivariate_normal.pdf(X, mean=mu, cov=cov)
    return p_x_given_k
