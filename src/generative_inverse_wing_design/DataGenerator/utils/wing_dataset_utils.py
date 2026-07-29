from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def _airfoil_family_key(df: pd.DataFrame) -> pd.Series:
    return df["airfoil_familys"].apply(lambda s: tuple(ast.literal_eval(s)))



def _drop_internal_duplicates_by_key(df: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    if key.empty:
        return df.copy()

    df_work = df.copy()
    df_work["_temp_tuple"] = key
    df_clean = df_work.drop_duplicates(subset=["_temp_tuple"], keep="first")
    return df_clean.drop(columns=["_temp_tuple"])


def _append_new_rows_by_key(
    combined_df: pd.DataFrame,
    df_clean: pd.DataFrame,
    key: pd.Series,
    seen_keys: set[tuple[Any, ...]],
) -> tuple[pd.DataFrame, set[tuple[Any, ...]], pd.DataFrame]:
    if key.empty:
        return combined_df, seen_keys, pd.DataFrame(columns=df_clean.columns)

    df_work = df_clean.copy()
    df_work["_temp_tuple"] = key
    mask_new = ~df_work["_temp_tuple"].isin(seen_keys)
    df_new = df_work[mask_new].copy()

    new_keys = set(df_new["_temp_tuple"].tolist())
    seen_keys.update(new_keys)

    combined_df = pd.concat(
        [combined_df, df_new.drop(columns=["_temp_tuple"])],
        ignore_index=True,
    )
    return combined_df, seen_keys, df_new.drop(columns=["_temp_tuple"])


def map_afs_cst_params(col: pd.Series) -> np.ndarray:
    parsed = col.map(json.loads)

    # extract cst params for each aifoil
    def extract_cst_params(aero_wing_dict: dict[str, Any]) -> list[float]:
        row_afs_cst_params: list[float] = []
        afs_dict = aero_wing_dict.get("airfoils", {})

        for i in range(len(afs_dict.keys())):
            af_dict = afs_dict.get(f"airfoil_{i}", {})
            leading_edge_weight = af_dict.get("leading_edge_w", 0.0)
            row_afs_cst_params.append(leading_edge_weight)

            upper_cst = af_dict.get("upper_surface", [])
            row_afs_cst_params.extend(upper_cst)

            lower_cst = af_dict.get("lower_surface", [])
            row_afs_cst_params.extend(lower_cst)

        return row_afs_cst_params

    features_lists = parsed.map(extract_cst_params).to_list()
    return np.array(features_lists, dtype="float32")


def remove_internal_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    key = _airfoil_family_key(df)
    return _drop_internal_duplicates_by_key(df, key)


def clean_and_combine_datasets(
    base_wings_path: str | Path,
    output_filename: str | Path = "combined_wings_data.csv",
    save_cleaned_csvs: bool = True,
) -> pd.DataFrame:
    base_path = Path(base_wings_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Path does not exist: {base_path}")

    # Find tip_fam CSVs sorted by number (with run_* folder)
    tip_fam_csvs = sorted(
        base_path.glob("tip_fam_*/wings/run_*/wings_data.csv"),
        key=lambda p: int(p.parent.parent.parent.name.split("_")[-1]),
    )
    # Find completion_set CSVs sorted by number (with run_* folder)
    completion_csvs = sorted(
        base_path.glob("completion_set_*/wings/run_*/wings_data.csv"),
        key=lambda p: int(p.parent.parent.parent.name.split("_")[-1]),
    )
    # Hydra's default generation path is usually .../wings/data/run_*/wings_data.csv.
    direct_run_csvs = sorted(
        base_path.glob("run_*/wings_data.csv"),
        key=lambda p: int(p.parent.name.split("_")[-1]),
    )
    hydra_run_csvs = sorted(
        base_path.glob("**/wings/data/run_*/wings_data.csv"),
        key=lambda p: int(p.parent.name.split("_")[-1]),
    )

    # Combine in order: curated groups first, then plain Hydra datagen outputs.
    csv_files = tip_fam_csvs + completion_csvs + direct_run_csvs + hydra_run_csvs

    if not csv_files:
        raise FileNotFoundError(f"No 'wings_data.csv' files found in {base_path}")

    print(f"Found {len(csv_files)} files:")
    combined_df = pd.DataFrame()
    combined_set: set[tuple[int, ...]] = set()  # Track combinations already in combined_df

    for csv_idx, csv_path in enumerate(csv_files):
        run_name = csv_path.parent.name
        print(f"[{csv_idx + 1}/{len(csv_files)}] Processing {run_name}...")
        # Load dataset
        df = pd.read_csv(csv_path)
        print(f"  Original rows: {len(df)}")

        # Step 1: Remove internal duplicates
        df_clean = remove_internal_duplicates(df)

        # Step 2: Check against combined dataset
        key = _airfoil_family_key(df_clean)
        combined_df, combined_set, df_new = _append_new_rows_by_key(
            combined_df=combined_df,
            df_clean=df_clean,
            key=key,
            seen_keys=combined_set,
        )

        print(f"  Final rows after cleaning: {len(df_new)}")
        print(f"  Combined dataset now has: {len(combined_df)} rows\n")
        # Save cleaned individual CSV (overwrite original)
        if save_cleaned_csvs:
            df_clean.to_csv(csv_path, index=False)

    output_path = base_path / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    return combined_df



def symmetric_quantile_clip(
    df: pd.DataFrame,
    metrics: list[str],
    data_range: float = 0.999,
    drop_statically_unstable: bool = True,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if not 0 < data_range <= 1:
        raise ValueError("data_range must be in (0, 1].")

    df_work = df.copy()
    n_before_stability = len(df_work)
    n_removed_stability = 0

    if drop_statically_unstable and "is_statically_stable" in df_work.columns:
        df_work = df_work[df_work["is_statically_stable"] == True].copy()  # noqa: E712
        n_removed_stability = n_before_stability - len(df_work)

    lower_quantile = (1 - data_range) / 2
    upper_quantile = 1 - lower_quantile

    valid_mask = np.ones(len(df_work), dtype=bool)

    for metric in metrics:
        if metric != "CL/CD":
            metric_values = df_work[metric].astype(float)
        else:
            metric_values = (df_work["CL"] / df_work["CD"]).astype(float)

        metric_values = metric_values[np.isfinite(metric_values)]
        lower = metric_values.quantile(lower_quantile)
        upper = metric_values.quantile(upper_quantile)

        if metric != "CL/CD":
            metric_mask = (df_work[metric].astype(float) >= lower) & (
                df_work[metric].astype(float) <= upper
            )
        else:
            cl_cd = (df_work["CL"] / df_work["CD"]).astype(float)
            metric_mask = (cl_cd >= lower) & (cl_cd <= upper)

        valid_mask &= metric_mask

    df_clipped = df_work[valid_mask].copy()
    original_count = len(df_work)
    final_count = len(df_clipped)
    total_removed = original_count - final_count

    summary: dict[str, float | int] = {
        "rows_before_stability": n_before_stability,
        "rows_removed_stability": n_removed_stability,
        "rows_before_clipping": original_count,
        "rows_after_clipping": final_count,
        "rows_removed_clipping": total_removed,
        "removal_pct_clipping": (total_removed / original_count) * 100 if original_count else 0.0,
        "retention_pct_clipping": (final_count / original_count) * 100 if original_count else 0.0,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
    }
    return df_clipped, summary


def nn_distance_metrics(samples: np.ndarray) -> dict[str, float]:
    nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(samples)
    dists, _ = nbrs.kneighbors(samples)
    # dists[:, 0] is the distance to itself, use nearest other point
    d1 = dists[:, 1]
    return {
        "nn_mean": float(d1.mean()),
        "nn_std": float(d1.std()),
    }


def pairwise_distance_metrics(samples: np.ndarray) -> dict[str, float]:
    dists = pdist(samples, metric="euclidean")
    return {
        "pairwise_mean": float(dists.mean()),
        "pairwise_std": float(dists.std()),
    }


def pca_explained_variance(X: np.ndarray, n_components: int) -> np.ndarray:
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return np.cumsum(pca.explained_variance_ratio_)


def get_af_cst_at_idx(afs: Any, sec_idx: int, num_cst_params: int) -> Any:
    start_idx = sec_idx * num_cst_params
    end_idx = start_idx + num_cst_params
    return afs[:, start_idx:end_idx]


def interpolate_cst_at_y(model: Any, x_afs: Any, y_val: float) -> Any:
    ys = model.wing_cfg.fixed.ys_over_half_span
    num_secs = len(ys)
    num_cst_params = x_afs.shape[1] // num_secs

    idx = int(np.searchsorted(ys, y_val))

    if idx == 0:
        return get_af_cst_at_idx(x_afs, 0, num_cst_params)
    if idx >= len(ys):
        return get_af_cst_at_idx(x_afs, len(ys) - 1, num_cst_params)

    y0 = ys[idx - 1]
    y1 = ys[idx]
    y_frac = (y_val - y0) / (y1 - y0)

    cst1 = get_af_cst_at_idx(x_afs, idx - 1, num_cst_params)
    cst2 = get_af_cst_at_idx(x_afs, idx, num_cst_params)
    return model._interpolate_af(y_frac, cst1, cst2)


def collect_loss_distribution(
    model: Any,
    dataloader: Any,
    num_batches: int = 5,
) -> pd.DataFrame:
    import torch

    model.eval()
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    rows: list[dict[str, float | int]] = []

    with torch.no_grad():
        for b_idx, batch in enumerate(dataloader):
            if b_idx >= num_batches:
                break

            x = batch.to(device)
            recon, _, _ = model(x)

            num_sampled_params = len(vars(model.wing_cfg.variables).keys())
            x_params = x[:, 0:num_sampled_params]
            x_afs = x[:, num_sampled_params:]
            recon_params = recon[:, 0:num_sampled_params]
            recon_afs = recon[:, num_sampled_params:]

            batch_size = x.shape[0]

            iou_loss = model.IoU(x, recon).detach().cpu().view(-1)
            geometry_loss = model.geometry_loss(x, recon).detach().cpu().view(-1)
            afs_recon_loss = model.afs_recon_loss(x_afs, recon_afs).detach().cpu().view(-1)

            param_recon_loss_scalar = model.param_recon_loss(x_params, recon_params).item()
            param_recon_loss = torch.full((batch_size,), param_recon_loss_scalar)

            for i in range(batch_size):
                rows.append(
                    {
                        "batch_idx": b_idx,
                        "sample_idx": i,
                        "iou_loss": float(iou_loss[i].item()),
                        "geometry_loss": float(geometry_loss[i].item()),
                        "afs_recon_loss": float(afs_recon_loss[i].item()),
                        "param_recon_loss": float(param_recon_loss[i].item()),
                    }
                )

    return pd.DataFrame(rows)


def find_ld_outliers(data_module: Any, ld_threshold: float = 200.0) -> list[Any]:
    vis_loader = data_module.test_dataloader()
    outlier_wing_params = []

    import torch

    with torch.no_grad():
        for batch in vis_loader:
            # batch shape: (B, feat_dim + 13_performance_metrics)
            perf_metrics = batch[:, -13:]  # Last 13 columns are performance metrics
            LD = perf_metrics[:, -1]  # L/D is the last metric

            for i in range(LD.shape[0]):
                if LD[i].item() > ld_threshold:
                    outlier_wing_params.append(batch[i, :-13].cpu().numpy())

    return outlier_wing_params


def plot_loss_histogram(
    df_losses: pd.DataFrame,
    metric: str = "iou_loss",
    bins: int = 50,
) -> None:
    """
    Plot a histogram of a chosen loss metric column in df_losses,
    with vertical lines for mean and ±2*std.
    """
    import matplotlib.pyplot as plt

    if metric not in df_losses.columns:
        raise ValueError(
            f"DataFrame must contain column '{metric}', got {list(df_losses.columns)}"
        )

    vals = np.asarray(df_losses[metric], dtype=float)
    mean = float(vals.mean())
    std = float(vals.std())

    plt.figure(figsize=(7, 5))
    plt.hist(vals, bins=bins, alpha=0.8, edgecolor="black", label="samples")

    # Mean line
    plt.axvline(mean, color="red", linestyle="-", linewidth=2, label=f"mean = {mean:.4f}")

    # ±2 sigma lines
    plt.axvline(mean - 2 * std, color="red", linestyle="--", linewidth=1.5, label="mean ± 2σ")
    plt.axvline(mean + 2 * std, color="red", linestyle="--", linewidth=1.5)

    plt.xlabel(metric)
    plt.ylabel("Count")
    plt.title(f"Distribution of per-sample {metric}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def debug_iou_computation(
    model: Any,
    x_batch: Any,
    x_recon_batch: Any,
    batch_idx: int = 0,
    y_eval_idx: int = 0,
    save_path: str | None = None,
) -> None:
    """
    Debug IoU computation by visualizing airfoil shapes and intersection/union areas.

    Args:
        model: The Wing_VAE model
        x_batch: Original input batch (batch_size, input_length)
        x_recon_batch: Reconstructed input batch (batch_size, input_length)
        batch_idx: Which sample in batch to debug (default: 0)
        y_eval_idx: Which spanwise position to debug (default: 0)
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    model.eval()

    with torch.no_grad():
        # Extract single sample
        x = x_batch[batch_idx : batch_idx + 1]  # Keep batch dimension
        x_recon = x_recon_batch[batch_idx : batch_idx + 1]

        num_sampled_params = len(vars(model.wing_cfg.variables).keys())
        x_sampled = x[:, 0:num_sampled_params]
        x_recon_sampled = x_recon[:, 0:num_sampled_params]

        # Extract wing parameters
        x_chord = x_sampled[:, 0:3]
        x_twist = x_sampled[:, 3:6]
        x_sweep = x_sampled[:, 6:7]
        x_recon_chord = x_recon_sampled[:, 0:3]
        x_recon_twist = x_recon_sampled[:, 3:6]
        x_recon_sweep = x_recon_sampled[:, 6:7]

        y_eval = model.y_eval  # (num_y_eval,)

        # Get parameters at evaluation point
        chords_at_y_eval = model._chords_at_y_eval(x_chord, y_eval)
        twists_at_y_eval = model._twists_at_y_eval(x_twist, y_eval)
        offsets_at_y_eval = model._offsets_at_y_eval(x_sweep, chords_at_y_eval, y_eval)
        recon_chords_at_y_eval = model._chords_at_y_eval(x_recon_chord, y_eval)
        recon_twists_at_y_eval = model._twists_at_y_eval(x_recon_twist, y_eval)
        recon_offsets_at_y_eval = model._offsets_at_y_eval(
            x_recon_sweep, recon_chords_at_y_eval, y_eval
        )

        # Get airfoil CST parameters at specific spanwise location
        x_afs = x[:, num_sampled_params:]
        x_recon_afs = x_recon[:, num_sampled_params:]

        num_secs = len(model.wing_cfg.fixed.ys_over_half_span)
        num_cst_params = x_afs.shape[1] // num_secs

        ys = model.wing_cfg.fixed.ys_over_half_span
        y = float(model.y_eval[y_eval_idx])

        idx = int(np.searchsorted(ys, y))
        if idx == 0:
            cst_interpolated = get_af_cst_at_idx(x_afs, 0, num_cst_params)
            cst_recon_interpolated = get_af_cst_at_idx(x_recon_afs, 0, num_cst_params)
        elif idx >= len(ys):
            cst_interpolated = get_af_cst_at_idx(x_afs, len(ys) - 1, num_cst_params)
            cst_recon_interpolated = get_af_cst_at_idx(x_recon_afs, len(ys) - 1, num_cst_params)
        else:
            y0 = ys[idx - 1]
            y1 = ys[idx]
            y_frac = (y - y0) / (y1 - y0)

            cst1 = get_af_cst_at_idx(x_afs, idx - 1, num_cst_params)
            cst2 = get_af_cst_at_idx(x_afs, idx, num_cst_params)
            cst_interpolated = model._interpolate_af(y_frac, cst1, cst2)

            cst1_recon = get_af_cst_at_idx(x_recon_afs, idx - 1, num_cst_params)
            cst2_recon = get_af_cst_at_idx(x_recon_afs, idx, num_cst_params)
            cst_recon_interpolated = model._interpolate_af(y_frac, cst1_recon, cst2_recon)

        # Get transformed coordinates
        af1_coords = model.af_coords_transformed(
            cst_interpolated,
            chords_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
            twists_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
            offsets_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
        )

        af2_coords = model.af_coords_transformed(
            cst_recon_interpolated,
            recon_chords_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
            recon_twists_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
            recon_offsets_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
        )

        # Convert to numpy for plotting
        af1_coords_np = af1_coords[0].cpu().numpy()  # (2*n_points_per_side-1, 2)
        af2_coords_np = af2_coords[0].cpu().numpy()

        # Compute intersection and union areas
        intersection_area, union_area = model.intersection_and_union(
            cst_interpolated,
            chords_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
            twists_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
            offsets_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
            cst_recon_interpolated,
            recon_chords_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
            recon_twists_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
            recon_offsets_at_y_eval[:, y_eval_idx : y_eval_idx + 1],
        )

        # --- REPLICATE MODEL LOGIC FOR VISUALIZATION ---
        # Prepare domains exactly as in intersection_and_union
        af1_x = af1_coords[:, :, 0]
        af1_y = af1_coords[:, :, 1]
        af2_x = af2_coords[:, :, 0]
        af2_y = af2_coords[:, :, 1]

        af1_x_domain = torch.flip(af1_x[:, : model.n_points_per_side], [-1])
        af1_y_upper = torch.flip(af1_y[:, : model.n_points_per_side], [-1])
        af1_y_lower = af1_y[:, model.n_points_per_side - 1 :]

        af2_x_domain = torch.flip(af2_x[:, : model.n_points_per_side], [-1])
        af2_y_upper = torch.flip(af2_y[:, : model.n_points_per_side], [-1])
        af2_y_lower = af2_y[:, model.n_points_per_side - 1 :]

        # 1. Get Intersection Geometry using model method
        # This returns the exact grid and y-values used for intersection area calculation
        x_common, upper_intersection, lower_intersection = model._get_union_coords(
            af1_x_domain, af1_y_upper, af1_y_lower, af2_x_domain, af2_y_upper, af2_y_lower
        )

        # 2. Get Union Geometry (Mimicking model logic for visualization)
        # The model calculates Union Area = Area1 + Area2 - Intersection Area
        # To visualize the "Union Shape", we construct a grid covering both airfoils
        # and interpolate using the model's torch_interp method

        # Find global min/max x
        af1_x_min = torch.min(af1_x_domain, dim=1).values.unsqueeze(-1)
        af1_x_max = torch.max(af1_x_domain, dim=1).values.unsqueeze(-1)
        af2_x_min = torch.min(af2_x_domain, dim=1).values.unsqueeze(-1)
        af2_x_max = torch.max(af2_x_domain, dim=1).values.unsqueeze(-1)

        x_min_union = torch.min(af1_x_min, af2_x_min)
        x_max_union = torch.max(af1_x_max, af2_x_max)

        # Create union grid
        x_ref = model.x_coords.unsqueeze(0).expand(x_common.shape[0], -1)
        x_union_grid = x_min_union + (x_max_union - x_min_union) * x_ref

        # Interpolate using model's interpolator
        af1_y_upper_u = model.torch_interp(x_union_grid, af1_x_domain, af1_y_upper)
        af1_y_lower_u = model.torch_interp(x_union_grid, af1_x_domain, af1_y_lower)
        af2_y_upper_u = model.torch_interp(x_union_grid, af2_x_domain, af2_y_upper)
        af2_y_lower_u = model.torch_interp(x_union_grid, af2_x_domain, af2_y_lower)

        # Compute union bounds
        upper_union = torch.max(af1_y_upper_u, af2_y_upper_u)
        lower_union = torch.min(af1_y_lower_u, af2_y_lower_u)

        # Convert to numpy for plotting
        x_common_np = x_common[0].cpu().numpy()
        upper_int_np = upper_intersection[0].cpu().numpy()
        lower_int_np = lower_intersection[0].cpu().numpy()

        x_union_np = x_union_grid[0].cpu().numpy()
        upper_union_np = upper_union[0].cpu().numpy()
        lower_union_np = lower_union[0].cpu().numpy()

        # --- VERIFICATION CALCULATIONS ---
        # Calculate Intersection Area using Trapezoidal Rule on the visualization points
        # Ensure height is non-negative (ReLU logic)
        h_int = np.maximum(0, upper_int_np - lower_int_np)
        area_int_trapz = np.abs(np.trapezoid(h_int, x_common_np))

        # Calculate Union Area using Trapezoidal Rule on the visualization points
        h_union = np.maximum(0, upper_union_np - lower_union_np)
        area_union_trapz = np.abs(np.trapezoid(h_union, x_union_np))

        # Create debug plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Airfoil comparison
        ax1.plot(
            af1_coords_np[:, 0], af1_coords_np[:, 1], color="grey", label="Original", linewidth=2
        )
        ax1.plot(
            af2_coords_np[:, 0],
            af2_coords_np[:, 1],
            color="blue",
            linestyle="--",
            label="Reconstructed",
            linewidth=2,
        )
        # Fill Union Area (Red)
        ax1.fill_between(
            x_union_np, lower_union_np, upper_union_np, color="red", alpha=0.5, label="Union Area"
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title(f"Airfoil Comparison at y={y:.3f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect("equal")

        # Plot 2: Raw CST parameters comparison
        cst_orig = cst_interpolated[0].cpu().numpy()
        cst_recon = cst_recon_interpolated[0].cpu().numpy()
        ax2.plot(cst_orig, "b-o", label="Original CST", markersize=4)
        ax2.plot(cst_recon, "r--s", label="Reconstructed CST", markersize=4)
        ax2.set_xlabel("CST Parameter Index")
        ax2.set_ylabel("Parameter Value")
        ax2.set_title("CST Parameters Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Intersection and Union Visualization
        ax3.plot(
            af1_coords_np[:, 0], af1_coords_np[:, 1], color="grey", label="Original", linewidth=2
        )
        ax3.plot(
            af2_coords_np[:, 0],
            af2_coords_np[:, 1],
            color="blue",
            linestyle="--",
            label="Reconstructed",
            linewidth=2,
        )

        # Fill Intersection Area (Green)
        valid_int = upper_int_np >= lower_int_np
        ax3.fill_between(
            x_common_np,
            lower_int_np,
            upper_int_np,
            where=valid_int,
            alpha=0.5,
            color="green",
            label="Intersection Area",
        )

        ax3.set_xlabel("X (Physical)")
        ax3.set_ylabel("Y (Physical)")
        ax3.set_title("Intersection & Union Visualization")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect("equal")

        # Plot 4: Area calculation details
        intersect_height = np.maximum(0, upper_int_np - lower_int_np)
        union_height = np.maximum(0, upper_union_np - lower_union_np)

        ax4.plot(x_common_np, intersect_height, "g-", label="Intersection Height", linewidth=2)
        ax4.plot(x_union_np, union_height, "r-", label="Union Height", linewidth=2)
        ax4.set_xlabel("X (Physical)")
        ax4.set_ylabel("Height")
        ax4.set_title("Height Distribution")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add text with computed values
        iou_computed = intersection_area.item() / (union_area.item() + 1e-8)

        fig.suptitle(
            f"Batch {batch_idx}, Y-eval {y_eval_idx} at y={y:.3f}\n"
            f"Model Intersection: {intersection_area.item():.6f} | Model Union: {union_area.item():.6f}\n"
            f"Area IoU: {iou_computed:.6f}"
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

        # Print debug information
        print("Debug Information:")
        print(f"Y evaluation position: {y:.6f}")
        print(f"Model Intersection area: {intersection_area.item():.6f}")
        print(f"Trapz Intersection area: {area_int_trapz:.6f}")
        print(f"Model Union area: {union_area.item():.6f}")
        print(f"Trapz Union area: {area_union_trapz:.6f}")
        print(f"IoU: {iou_computed:.6f}")


def visualize_iou_components(
    model: Any, batch: Any, sample_idx: int = 0, resolution: int = 100
) -> None:
    """
    Visualizes the Intersection and Union Area distributions separately.
    This verifies if the Gaussian Quadrature accurately integrates the components
    used for the global Volume IoU calculation.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    model.eval()

    # Get single sample
    x = batch[sample_idx : sample_idx + 1]

    with torch.no_grad():
        x_recon, _, _ = model(x)

        # Extract parameters
        num_sampled_params = len(vars(model.wing_cfg.variables).keys())
        x_params = x[:, 0:num_sampled_params]
        x_recon_params = x_recon[:, 0:num_sampled_params]
        x_afs = x[:, num_sampled_params:]
        x_recon_afs = x_recon[:, num_sampled_params:]

        # 1. Generate dense evaluation points
        y_dense = torch.linspace(0, 1, resolution, device=x.device)
        # 2. Get Quadrature points
        y_quad = model.y_eval

        # Helper to compute Intersection and Union Areas at points
        def compute_areas_at_points(y_points):
            chords = model._chords_at_y_eval(x_params[:, 0:3], y_points)
            twists = model._twists_at_y_eval(x_params[:, 3:6], y_points)
            offsets = model._offsets_at_y_eval(x_params[:, 6:7], chords, y_points)

            recon_chords = model._chords_at_y_eval(x_recon_params[:, 0:3], y_points)
            recon_twists = model._twists_at_y_eval(x_recon_params[:, 3:6], y_points)
            recon_offsets = model._offsets_at_y_eval(
                x_recon_params[:, 6:7], recon_chords, y_points
            )

            intersections = []
            unions = []
            y_np = y_points.cpu().numpy()

            for i, y_val in enumerate(y_np):
                cst = interpolate_cst_at_y(model, x_afs, float(y_val))
                cst_recon = interpolate_cst_at_y(model, x_recon_afs, float(y_val))

                inter, union = model.intersection_and_union(
                    cst,
                    chords[:, i : i + 1],
                    twists[:, i : i + 1],
                    offsets[:, i : i + 1],
                    cst_recon,
                    recon_chords[:, i : i + 1],
                    recon_twists[:, i : i + 1],
                    recon_offsets[:, i : i + 1],
                )
                intersections.append(inter.item())
                unions.append(union.item())

            return np.array(intersections), np.array(unions)

        # Compute Areas
        I_dense, U_dense = compute_areas_at_points(y_dense)
        I_quad, U_quad = compute_areas_at_points(y_quad)

        y_dense_np = y_dense.cpu().numpy()
        y_quad_np = y_quad.cpu().numpy()
        poly_degree = len(y_quad_np) - 1

        # --- Helper to fit poly and integrate ---
        def fit_and_integrate(y_pts, vals, y_dense_pts):
            coeffs = np.polyfit(y_pts, vals, poly_degree)
            poly_vals = np.polyval(coeffs, y_dense_pts)
            integ_coeffs = np.polyint(coeffs)
            integral = np.polyval(integ_coeffs, 1.0) - np.polyval(integ_coeffs, 0.0)
            return poly_vals, integral

        # Fit polynomials
        I_poly_vals, int_I_poly = fit_and_integrate(y_quad_np, I_quad, y_dense_np)
        U_poly_vals, int_U_poly = fit_and_integrate(y_quad_np, U_quad, y_dense_np)

        # Calculate True Integrals (Trapezoidal)
        int_I_true = np.trapezoid(I_dense, y_dense_np)
        int_U_true = np.trapezoid(U_dense, y_dense_np)

        # Calculate Volume IoUs
        iou_vol_true = int_I_true / (int_U_true + 1e-8)
        iou_vol_poly = int_I_poly / (int_U_poly + 1e-8)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Intersection Area
        ax1.plot(y_dense_np, I_dense, "b-", label="True Area")
        ax1.plot(y_quad_np, I_quad, "ro", label="Quad Points")
        ax1.plot(y_dense_np, I_poly_vals, "g--", alpha=0.7, label="Poly Fit")
        ax1.set_title(
            f"Intersection Area Distribution\nInt True: {int_I_true:.4f} | Int Poly: {int_I_poly:.4f}"
        )
        ax1.set_xlabel("Spanwise Position (y)")
        ax1.set_ylabel("Area")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Union Area
        ax2.plot(y_dense_np, U_dense, "b-", label="True Area")
        ax2.plot(y_quad_np, U_quad, "ro", label="Quad Points")
        ax2.plot(y_dense_np, U_poly_vals, "g--", alpha=0.7, label="Poly Fit")
        ax2.set_title(
            f"Union Area Distribution\nInt True: {int_U_true:.4f} | Int Poly: {int_U_poly:.4f}"
        )
        ax2.set_xlabel("Spanwise Position (y)")
        ax2.set_ylabel("Area")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.suptitle(
            f"Volume IoU Comparison (Sample {sample_idx})\n"
            f"True Volume IoU: {iou_vol_true:.5f}  vs  Quad/Poly Volume IoU: {iou_vol_poly:.5f}",
            fontsize=14,
        )

        plt.tight_layout()
        plt.show()


def debug_iou_example(model: Any, dataloader: Any) -> None:
    import torch

    # Get a batch from your dataloader
    batch = next(iter(dataloader))
    batch = batch.to(model.device)

    # Forward pass
    with torch.no_grad():
        x_recon, mean, logvar = model.forward(batch)

    # Debug IoU computation
    debug_iou_computation(
        model=model,
        x_batch=batch,
        x_recon_batch=x_recon,
        batch_idx=0,  # First sample in batch
        y_eval_idx=5,  # Middle evaluation point
    )

    # Run the visualization
    visualize_iou_components(model, batch)


def tsne_latent_space_from_visualization_dataset(
    model: Any,
    data_module: Any,
    num_samples: int = 2000,
    random_state: int = 42,
) -> None:
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
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import torch

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
        "max_van_mises_stress",
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
        "CM",
        "CMa",
        "mass",
        "CL/CD",
    ]
    metric_indices = {name: perf_names.index(name) for name in selected_metrics}

    # --- Pass through VAE, collect latent means ---
    latents = []
    batch_size = 512
    with torch.no_grad():
        for start in range(0, n_take, batch_size):
            end = min(start + batch_size, n_take)
            x_batch = X[start:end][:, : model.feature_length].to(device)
            # model(x) -> (recon, mu, logvar)
            _, mu, _ = model(x_batch)
            latents.append(mu.cpu().numpy())

    Z = np.concatenate(latents, axis=0)  # (n_take, latent_dim)

    # --- t-SNE to 2D ---
    pca = PCA(n_components=2, random_state=random_state)
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

        plt.figure(figsize=(7, 6))
        sc = plt.scatter(Z_2d[mask, 0], Z_2d[mask, 1], c=vals[mask], cmap="viridis", s=10)
        plt.colorbar(sc, label=name)
        plt.xlabel("PCA dim 1")
        plt.ylabel("PCA dim 2")
        plt.title(f"Latent Space PCA colored by {name}\n({np.sum(mask)} samples, 3σ clipped)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
