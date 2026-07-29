
from pathlib import Path

from aerosandbox.numpy import np
import hydra
from omegaconf import DictConfig

from generative_inverse_wing_design.DataGenerator.aeroelasticwing import AeroelasticWing
from generative_inverse_wing_design.DataGenerator.airfoilrepository import (
    AirfoilRepository,
)
from generative_inverse_wing_design.DataGenerator.appconfig import AppConfig


class WingSampler:
    """"A class to sample AeroelasticWings, based on the configuration.
    """

    def __init__(self, cfg: AppConfig, airfoil_repo: AirfoilRepository):
        self.cfg: AppConfig = cfg
        self.airfoil_repo: AirfoilRepository = airfoil_repo

    def sample_wing(self, airfoil_familys: np.ndarray) -> AeroelasticWing:
        """Samples a aeroelastic_wing based on the configuration and given airfoil families.
        Args:
            airfoil_familys: An array of Airfoil_families given by the generator.

        Returns:
            An AeroelasticWing instance representing the sampled wing.
        """
        # sample variable wing parameters
        sampled_params = {}
        sampled_params["chord_ctrl_pos"] = self.cfg.variables.chord_ctrl_pos_dist.sample()
        sampled_params["chord_weight_ctrl"] = self.cfg.variables.chord_weight_ctrl_dist.sample()
        sampled_params["chord_tip"] = self.cfg.variables.chord_tip_dist.sample()
        sampled_params["twist_root"] = self.cfg.variables.twist_root_dist.sample()
        sampled_params["twist_tip"] = self.cfg.variables.twist_tip_dist.sample()
        sampled_params["twist_mean_pos"] = self.cfg.variables.twist_mean_pos_dist.sample()
        sampled_params["spar_sweep"] = self.cfg.variables.spar_sweep_dist.sample()

        # sample airfoils for root and tip from given airfoil families
        airfoils = self.airfoil_repo.sample_airfoils(airfoil_familys)

        return AeroelasticWing(cfg=self.cfg, sampled_params=sampled_params, airfoils=airfoils)

@hydra.main(version_base=None, config_path="../../../configs/", config_name="dataset_config")
def main(cfg: DictConfig):
    config = AppConfig(cfg)
    path = Path("data/processed/airfoil_with_polars.csv")
    airfoilrepo = AirfoilRepository(path)

    wing_sampler = WingSampler(config, airfoilrepo)
    airfoil_familys = np.array([0,1,2,3,4])
    wing = wing_sampler.sample_wing(airfoil_familys)
    wing.mechanical_wing.solve_for_properties()
    wing.mechanical_wing.plot_at_rib_positions()
    wing.mechanical_wing.plot_mesh_at_rib_position(rib_index=2)
    wing.set_asb_wing()
    wing.plot_asb_wing()
    wing.solve_all_my_problems()
    wing.plot_comparison()

if __name__ == "__main__":
    main()
