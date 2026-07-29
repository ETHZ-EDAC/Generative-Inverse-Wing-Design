from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import random
from typing import Any

import aerosandbox.numpy as np
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import torch
import yaml

from generative_inverse_wing_design.DataGenerator.appconfig import AppConfig
from generative_inverse_wing_design.models.VAE.cVAE import cVAE
from generative_inverse_wing_design.models.VAE.preprocessing_VAE import VAE
from generative_inverse_wing_design.models.VAE.preprocessing_VAE_data_module import (
    AirfoilDataModule,
)
from generative_inverse_wing_design.models.VAE.wing_dataset_VAE import Wing_VAE
from generative_inverse_wing_design.models.VAE.wing_dataset_VAE_data_module import (
    WingDataModule,
)


def _find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def _resolve_wing_config_path(config_path: str, project_root: Path, config_dir: Path) -> Path:
    raw_path = Path(config_path)
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                project_root / raw_path,
                config_dir / raw_path,
                project_root / "configs" / raw_path,
            ]
        )

    candidates.append(project_root / "configs" / "dataset_config.yaml")

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        resolved_str = str(resolved)
        if resolved_str not in seen:
            seen.add(resolved_str)
            unique_candidates.append(resolved)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    attempted = "\n".join(f"- {path}" for path in unique_candidates)
    raise FileNotFoundError(
        "Could not locate wing config file from cfg.data.wing_config_path. "
        f"Attempted:\n{attempted}"
    )


def train_cvae_with_hydra(
    seed: int | None = None,
    save_curves: bool = False,
    tracking_uri: str | None = None,
    cfg_override: DictConfig | None = None,
):
    """
    Train cVAE using Hydra configuration from configs/wing_cVAE_configs/config.yaml.

    Returns:
        model, trainer, cfg, wing_cfg, data_module
    """
    GlobalHydra.instance().clear()

    project_root = _find_project_root()
    config_dir = project_root / "configs" / "wing_cVAE_configs"

    if tracking_uri is None:
        mlflow_tracking_dir = project_root / "outputs" / "cVAE_training_runs"
        mlflow_tracking_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = mlflow_tracking_dir.resolve().as_uri()

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="config")
        eval_n_clusters = 6
        if cfg_override is not None:
            cfg_override = OmegaConf.create(OmegaConf.to_container(cfg_override, resolve=True))
            eval_n_clusters = int(OmegaConf.select(cfg_override, "eval.n_clusters", default=6))
            if "eval" in cfg_override:
                del cfg_override["eval"]
            cfg = OmegaConf.merge(cfg, cfg_override)

        wing_cfg_file = _resolve_wing_config_path(
            config_path=str(cfg.data.wing_config_path),
            project_root=project_root,
            config_dir=config_dir,
        )

        run_dir = OmegaConf.select(cfg, "hydra.run.dir", default=os.getcwd())
        txt = wing_cfg_file.read_text().replace("${hydra:run.dir}", str(run_dir))
        wing_cfg = AppConfig(OmegaConf.create(yaml.safe_load(txt)))

        if seed is None:
            seed = (
                wing_cfg.run.seed
                if hasattr(wing_cfg.run, "seed") and wing_cfg.run.seed is not None
                else 42
            )

        L.seed_everything(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("Initializing data module...")
        data_module = WingDataModule(cfg.data, seed=seed, load_labels=True)

        print("Initializing VAE model...")
        model = cVAE(cfg.model, scaler_aero=data_module.scaler, wing_cfg=wing_cfg)
        model.eval_n_clusters = eval_n_clusters
        if save_curves:
            model.save_learning_curves(seed=seed)

        run_name = f"VAE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlf_logger = MLFlowLogger(
            experiment_name=cfg.experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            log_model=False,
        )
        mlf_logger.log_hyperparams(cfg)

        callbacks: list[Any] = []
        if cfg.runs.early_stopping_enabled:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=cfg.runs.early_stopping_min_delta,
                    patience=cfg.runs.early_stopping_patience,
                    verbose=False,
                    mode="min",
                )
            )

        trainer = L.Trainer(
            max_epochs=cfg.runs.trainer.max_epochs,
            logger=mlf_logger,
            callbacks=callbacks,
            accelerator=cfg.runs.trainer.accelerator,
            devices=cfg.runs.trainer.devices,
            precision=cfg.runs.trainer.precision,
            gradient_clip_val=cfg.runs.trainer.gradient_clip_val,
            log_every_n_steps=cfg.runs.trainer.log_every_n_steps,
            check_val_every_n_epoch=cfg.runs.trainer.check_val_every_n_epoch,
            enable_progress_bar=False,
            enable_model_summary=True,
        )

        print("Starting training on ...")
        print(f"trainer max epochs: {trainer.max_epochs}")
        print(f"early stopping enabled: {cfg.runs.early_stopping_enabled}")
        print(f"early stopping patience: {cfg.runs.early_stopping_patience}")

        trainer.fit(model, data_module)

        print("Running test...")
        trainer.test(model, data_module)

        if trainer.callback_metrics:
            final_metrics = {
                key: value.item() if torch.is_tensor(value) else value
                for key, value in trainer.callback_metrics.items()
            }
            mlf_logger.log_metrics(final_metrics)

        print("Training completed!")
        print(f"MLFlow Run ID: {mlf_logger.run_id}")
        print(f"MLFlow Experiment ID: {mlf_logger.experiment_id}")

    return model, trainer, cfg, wing_cfg, data_module


def train_wing_vae_with_hydra(
    seed: int | None = None,
    save_curves: bool = False,
    tracking_uri: str | None = None,
    cfg_override: DictConfig | None = None,
):
    """
    Train the wing VAE using configs/wing_dataset_VAE_configs/config.yaml.

    Returns:
        model, trainer, cfg, wing_cfg, data_module
    """
    GlobalHydra.instance().clear()

    project_root = _find_project_root()
    config_dir = project_root / "configs" / "wing_dataset_VAE_configs"

    if tracking_uri is None:
        mlflow_tracking_dir = project_root / "outputs" / "wing_VAE_training_runs"
        mlflow_tracking_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = mlflow_tracking_dir.resolve().as_uri()

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="config")
        if cfg_override is not None:
            cfg = OmegaConf.merge(cfg, cfg_override)

        wing_cfg_file = _resolve_wing_config_path(
            config_path=str(cfg.data.wing_config_path),
            project_root=project_root,
            config_dir=config_dir,
        )

        run_dir = OmegaConf.select(cfg, "hydra.run.dir", default=os.getcwd())
        txt = wing_cfg_file.read_text().replace("${hydra:run.dir}", str(run_dir))
        wing_cfg = AppConfig(OmegaConf.create(yaml.safe_load(txt)))

        if seed is None:
            seed = (
                wing_cfg.run.seed
                if hasattr(wing_cfg.run, "seed") and wing_cfg.run.seed is not None
                else 42
            )

        L.seed_everything(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("Initializing data module...")
        data_module = WingDataModule(cfg.data, seed=seed, load_labels=False)

        print("Initializing VAE model...")
        model = Wing_VAE(cfg.model, scaler_aero=data_module.scaler, wing_cfg=wing_cfg)
        if save_curves:
            model.save_learning_curves(seed=seed)

        run_name = f"WingVAE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlf_logger = MLFlowLogger(
            experiment_name=cfg.experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            log_model=False,
        )
        mlf_logger.log_hyperparams(cfg)

        callbacks: list[Any] = []
        if cfg.runs.early_stopping_enabled:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=cfg.runs.early_stopping_min_delta,
                    patience=cfg.runs.early_stopping_patience,
                    verbose=False,
                    mode="min",
                )
            )

        trainer = L.Trainer(
            max_epochs=cfg.runs.trainer.max_epochs,
            logger=mlf_logger,
            callbacks=callbacks,
            accelerator=cfg.runs.trainer.accelerator,
            devices=cfg.runs.trainer.devices,
            precision=cfg.runs.trainer.precision,
            gradient_clip_val=cfg.runs.trainer.gradient_clip_val,
            log_every_n_steps=cfg.runs.trainer.log_every_n_steps,
            val_check_interval=cfg.runs.trainer.val_check_interval,
            enable_progress_bar=False,
            enable_model_summary=True,
        )

        print("Starting training...")
        trainer.fit(model, data_module)

        print("Running test...")
        trainer.test(model, data_module)

        if trainer.callback_metrics:
            final_metrics = {
                key: value.item() if torch.is_tensor(value) else value
                for key, value in trainer.callback_metrics.items()
            }
            mlf_logger.log_metrics(final_metrics)

        print("Training completed!")
        print(f"MLFlow Run ID: {mlf_logger.run_id}")
        print(f"MLFlow Experiment ID: {mlf_logger.experiment_id}")

    return model, trainer, cfg, wing_cfg, data_module


def train_preprocessing_vae_with_hydra(
    seed: int | None = None,
    save_curves: bool = False,
    tracking_uri: str | None = None,
    cfg_override: DictConfig | None = None,
):
    """
    Train the airfoil preprocessing VAE using
    configs/preprocessing_VAE_configs/config.yaml.

    Returns:
        model, trainer, cfg, data_module
    """
    GlobalHydra.instance().clear()

    project_root = _find_project_root()
    config_dir = project_root / "configs" / "preprocessing_VAE_configs"

    if tracking_uri is None:
        mlflow_tracking_dir = project_root / "outputs" / "preprocessing_VAE_training_runs"
        mlflow_tracking_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = mlflow_tracking_dir.resolve().as_uri()

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="config")
        eval_n_clusters = 6
        if cfg_override is not None:
            cfg_override = OmegaConf.create(OmegaConf.to_container(cfg_override, resolve=True))
            eval_n_clusters = int(OmegaConf.select(cfg_override, "eval.n_clusters", default=6))
            if "eval" in cfg_override:
                del cfg_override["eval"]
            cfg = OmegaConf.merge(cfg, cfg_override)

        print("Configuration:")
        print(cfg)

        if seed is None:
            seed = cfg.seed if hasattr(cfg, "seed") else 42

        L.seed_everything(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("Initializing data module...")
        data_module = AirfoilDataModule(cfg.data)

        print("Initializing VAE model...")
        model = VAE(cfg.model, scaler_aero=data_module.scaler)
        model.eval_n_clusters = eval_n_clusters
        if save_curves:
            model.save_learning_curves(seed=seed)

        run_name = f"VAE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlf_logger = MLFlowLogger(
            experiment_name=cfg.experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            log_model=False,
        )
        mlf_logger.log_hyperparams(cfg)

        callbacks: list[Any] = []
        if cfg.runs.early_stopping_enabled:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=cfg.runs.early_stopping_min_delta,
                    patience=cfg.runs.early_stopping_patience,
                    verbose=False,
                    mode="min",
                )
            )

        trainer = L.Trainer(
            max_epochs=cfg.runs.trainer.max_epochs,
            logger=mlf_logger,
            callbacks=callbacks,
            accelerator=cfg.runs.trainer.accelerator,
            devices=cfg.runs.trainer.devices,
            precision=cfg.runs.trainer.precision,
            gradient_clip_val=cfg.runs.trainer.gradient_clip_val,
            log_every_n_steps=cfg.runs.trainer.log_every_n_steps,
            check_val_every_n_epoch=cfg.runs.trainer.check_val_every_n_epoch,
            enable_progress_bar=False,
            enable_model_summary=True,
        )

        print("Starting training...")
        trainer.fit(model, data_module)

        print("Running test...")
        trainer.test(model, data_module)

        if trainer.callback_metrics:
            final_metrics = {
                key: value.item() if torch.is_tensor(value) else value
                for key, value in trainer.callback_metrics.items()
            }
            mlf_logger.log_metrics(final_metrics)

        print("Training completed!")
        print(f"MLFlow Run ID: {mlf_logger.run_id}")
        print(f"MLFlow Experiment ID: {mlf_logger.experiment_id}")

    return model, trainer, cfg, data_module
