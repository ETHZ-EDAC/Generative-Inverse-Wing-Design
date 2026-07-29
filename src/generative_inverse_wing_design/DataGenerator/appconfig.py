# app_config.py
""""
Contains the AppConfig class for managing application configurations using OmegaConf.
Includes classes for fixed wing parameters, variable wing parameter distributions,
operating points, and data runner settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Dict, Optional, Tuple, Union, cast

from aerosandbox.numpy import np
import hydra
from omegaconf import DictConfig, OmegaConf
from sectionproperties.pre import Material


@dataclass(frozen=True)
class OperatingPoint:
    """Operating point parameters for the aeroelastic analysis."""

    mach: float  # [dimensionless]
    velocity: float  # [m/s]
    chordwise_resolution: int  # [num of panels along chord]
    spanwise_resolution: int  # [num of panels along span, per corss section]

class FixedWingParameters:
    """Fixed wing parameters for the wing design."""

    def __init__(self, cfg: DictConfig):
        self.span: float = float(cfg.span)  # [m]
        self.root_chord: float = float(cfg.root_chord)  # [m]
        self.ys_over_half_span: np.ndarray = np.array(cfg.ys_over_half_span)
        # relative positions along half-span [0, 1]
        self.rib_positions: np.ndarray = np.array(cfg.rib_positions)
        # relative positions along half-span [0, 1]

        self.material: Material = Material(
            name=cfg.material_name,
            elastic_modulus=float(cfg.material_elastic_modulus),  # [Pa]
            poissons_ratio=float(cfg.material_poissons_ratio),  # [dimensionless]
            density=float(cfg.material_density),  # [kg/m^3]
            yield_strength=float(cfg.material_yield_strength),  # [Pa]
            color="lightgrey"
        )

        self.skin_thickness: float = float(cfg.skin_thickness) # [m]
        self.spar_position: float = float(cfg.spar_position) # relative position along chord [0, 1]
        self.spar_web_flange_thickness: float = float(cfg.spar_web_flange_thickness) # [m]
        self.spar_flange_width: float = float(cfg.spar_flange_width) # [m]
        self.rib_thickness: float = float(cfg.rib_thickness) # [m]

        self.rib_hole_positions: np.ndarray[float] = np.array(cfg.rib_hole_positions)
        # relative positions along chord [0, 1]
        self.rib_min_border: float = cfg.rib_min_border # [m]


class DistributionSpec:
    """
    Distribution specification for a variable wing parameter:
      - kind: "uniform" or "normal"
      - params: list of floats (for "uniform": [lo, hi]; for "normal": [mu, var])
      - bounds: [min, max] hard bounds for resampleing
    """
    def __init__(self, dist: Dict[str, Any]):

        self._kind: str = dist["kind"]
        self._params: np.ndarray = np.array(dist["params"])
        self._bounds: Tuple[float, float] = tuple(dist["bounds"])
        self.validate()

    def validate(self) -> None:
        assert self._kind in ("uniform", "normal"), \
            f"DistributionSpec.kind must be 'uniform' or 'normal', got '{self._kind}'."
        if self._kind == "uniform":
            assert len(self._params) == 2, "For 'uniform' distribution, params must be [lo, hi]."
        if self._kind == "normal":
            assert len(self._params) == 2, "For 'normal' distribution, params must be [mu, var]."
        assert self._bounds[0] <= self._bounds[1], \
            "DistributionSpec.bounds must be [min, max] with finite values."

    def sample(self, n_samples: Optional[int] = None) -> Union[float, np.ndarray]:
        """Sample from the distribution, resampling if out of bounds.
        Args:
            n_samples: Number of samples to draw. If None, draw a single sample.
        Returns:
            A single float sample or an array of samples.
        """
        if n_samples is None:
            return self._single_sample()
        else:
            samples = np.array([self._single_sample() for _ in range(n_samples)])
            return samples

    def _single_sample(self) -> float:
        if self._kind == "uniform":
            lo, hi = self._params
            i = 0
            while i < 10000: # to prevent infinite loops
                sample = np.random.uniform(lo, hi)
                if self._bounds[0] <= sample <= self._bounds[1]:
                    return sample
                i += 1
            return (self._bounds[0] + self._bounds[1]) / 2.0  # fallback to midpoint of bounds
        else:
            mu, var = self._params
            i = 0
            while i < 10000:
                sample = np.random.normal(mu, np.sqrt(var))
                if self._bounds[0] <= sample <= self._bounds[1]:
                    return sample
                i += 1
            return (self._bounds[0] + self._bounds[1]) / 2.0  # fallback to midpoint of bounds

class VariableWingParameterDist:
    """
    Variable wing parameter distributions for sampling:
    -c_tip: chordlength at wing tip
    -x_ctrl: relative x-coordinate of control point along half-span
    -weight_ctrl: relative weight of control point in rational bezier curve for chord lengths
    -midchord_sweep: midchord sweep angle in degrees
    -twist_root: twist angle at wing root in degrees
    -twist_tip: twist angle at wing tip in degrees
    """

    def __init__(self, cfg: DictConfig):

        self.chord_tip_dist: DistributionSpec = DistributionSpec(cfg.chord_tip)
        self.chord_ctrl_pos_dist: DistributionSpec = DistributionSpec(cfg.chord_ctrl_pos)
        self.chord_weight_ctrl_dist: DistributionSpec = DistributionSpec(cfg.chord_weight_ctrl)
        self.spar_sweep_dist: DistributionSpec = DistributionSpec(cfg.spar_sweep)
        self.twist_root_dist: DistributionSpec = DistributionSpec(cfg.twist_root)
        self.twist_tip_dist: DistributionSpec = DistributionSpec(cfg.twist_tip)
        self.twist_mean_pos_dist: DistributionSpec = DistributionSpec(cfg.twist_mean_pos)

    def __len__(self) -> int:
        return len(vars(self))

class DataRunnerSettings:
    """Settings for the data runner."""

    def __init__(self, cfg: DictConfig):
        self.seed: Optional[int] = cfg.seed if "seed" in cfg else None
        self.num_workers: int = cfg.num_workers
        self.output_folder_path: Path = Path(cfg.output_folder_path)
        self.airfoils_csv_path: Path = Path(cfg.airfoils_csv_path)
        self.num_samples: Optional[int] = cfg.num_samples if "num_samples" in cfg else None
        self.tip_airfoil_family: Optional[int] = cfg.tip_airfoil_family if\
            "tip_airfoil_family" in cfg else None

class AppConfig:
    """
    Config built from OmegaConf DictConfig. Contains:
      - fixed_wing_parameters
      - variable_wing_parameters
      - operating_point
      - datarunner_settings
    """
    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg
        self.fixed: FixedWingParameters = FixedWingParameters(cfg.fixed_wing_parameters)
        self.variables: VariableWingParameterDist = VariableWingParameterDist(cfg.variable_wing_parameters)
        self.operating_point: OperatingPoint = OperatingPoint(**cfg.operating_point)
        self.run: DataRunnerSettings = DataRunnerSettings(cfg.datarunner_settings)

    def to_yaml(self) -> str:
        return OmegaConf.to_yaml(self.cfg)

    def to_dict(self) -> dict:
        return cast(dict, OmegaConf.to_container(self.cfg, resolve=True))

    def __repr__(self) -> str:
        return self.to_yaml()

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state: dict):
        cfg = OmegaConf.create(state)
        self.__init__(cfg)

    @staticmethod
    def from_dict(cfg_dict: dict) -> "AppConfig":
        cfg = OmegaConf.create(cfg_dict)
        return AppConfig(cfg)


@hydra.main(version_base=None, config_path="../../../configs/", config_name="dataset_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    config = AppConfig(cfg)
    picked = pickle.dumps(config)
    unpickled = pickle.loads(picked)
    print(unpickled)
    print(config)
    print(type(config.fixed.ys_over_half_span))
    print(len(config.variables))
    print(vars(config.variables).keys())


if __name__ == "__main__":
    main()
