"""Contains the Airfoil and AirfoilRepository classes for managing airfoil data."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import aerosandbox as asb
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_coordinates
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon


class Airfoil:
    """
    Represents an airfoil using Kulfan CST parameters with a leading edge modification.
    Assuming class function exponents N1, N2: (0.5, 1.0), and trailing edge thickness = 0.
    """

    def __init__(self,
                  leading_edge_w: float,
                  upper_surface: np.ndarray,
                  lower_surface: np.ndarray,
                  airfoil_id: int = -1):

        #double typing I know but if someone changes it manual later
        self.leading_edge_w: float = leading_edge_w  # leading edge wedge weight
        self.upper_surface: np.ndarray = upper_surface  # upper surface weights
        self.lower_surface: np.ndarray = lower_surface  # lower surface weights
        self.airfoil_id: int = airfoil_id  # idx of dataframe

    def to_coordinates(self, chord_length: float = 1, n_points_per_side: int = 96) -> np.ndarray:
        """Returns the airfoil coordinates as an (N, 2) array of (x, y) points.

        Args:
            n_points_per_side: Number of points to generate along each side of the airfoil.
            chord_length: Scale factor for coordinates (1.0 = unscaled unit chord).
        Returns:
            An (N, 2) array of (x, y) points scaled to chord_length.
        """
        coords = get_kulfan_coordinates(
            lower_weights=self.lower_surface,
            upper_weights=self.upper_surface,
            leading_edge_weight=self.leading_edge_w,
            TE_thickness=0.0,
            n_points_per_side=n_points_per_side
        )
        return coords * chord_length

    def to_polygon(self, chord_length: float = 1) -> Polygon:
        """Turns the Airfoil into a Polygon."""
        coords = self.to_coordinates(chord_length=chord_length)
        polygon = Polygon(coords)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_valid:
            raise ValueError("Airfoil cant be converted to valid polygon.")
        return polygon

    def to_rotated_coordinates(self, chord_length: float = 1,
                               incidence_angle: float = 0,
                               center_of_rotation: np.ndarray = np.array([0, 0])
                               ) -> np.ndarray:
        """Returns the airfoil coordinates rotated by the given incidence angle around the center of rotation.
        Incidence angle is defined clockwise positive.
        Args:
            chord_length: Scale factor for coordinates (1.0 = unscaled unit chord).
            incidence_angle: Angle of incidence in degrees to rotate the airfoil coordinates.
            center_of_rotation: The point about which to rotate the airfoil.
        Returns:
            An (N, 2) array of rotated (x, y) points scaled to chord_length.
        """
        assert center_of_rotation.shape == (2,),\
            "center_of_rotation must be of shape (2,)."

        coords = self.to_coordinates(chord_length=chord_length)

        if incidence_angle != 0:
            # rotate clockwise as incidence is nose up
            theta = -np.radians(incidence_angle)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta),  np.cos(theta)]])
            coords = (coords - center_of_rotation) @ rotation_matrix.T + center_of_rotation

        return coords

    def min_max_y_coords(self, chord_length: float = 1,
                         incidence_angle: float = 0,
                         center_of_rotation: np.ndarray = np.array([0, 0])
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the minimum and maximum coordinates of the airfoil. Considering the rotation.
        Incidence angle is defined clockwise positive.
        Args:
            chord_length: Scale factor for coordinates (1.0 = unscaled unit chord).
            incidence_angle: Angle of incidence in degrees to rotate the airfoil coordinates.
            center_of_rotation: The point about which to rotate the airfoil.
        Returns:
            A tuple of (min_y_point, max_y_point), each an array of shape (2,).
        """
        coords = self.to_rotated_coordinates(
            chord_length=chord_length,
            incidence_angle=incidence_angle,
            center_of_rotation=center_of_rotation
        )

        min_y_idx = np.argmin(coords[:, 1])
        max_y_idx = np.argmax(coords[:, 1])

        min_y_point = coords[min_y_idx]
        max_y_point = coords[max_y_idx]

        return min_y_point, max_y_point

    def to_asb_airfoil(self) -> asb.Airfoil:
        """Converts to an Aerosandbox Airfoil object."""

        kulfan_airfoil = asb.KulfanAirfoil(
            name=f"Airfoil_{self.airfoil_id if self.airfoil_id is not None else 'N/A'}",
            lower_weights=self.lower_surface,
            upper_weights=self.upper_surface,
            leading_edge_weight=self.leading_edge_w,
            TE_thickness=0.0
        )
        return kulfan_airfoil.to_airfoil()

    def to_dict(self) -> dict:
        """Converts the Airfoil to a dictionary representation."""
        return {
            "leading_edge_w": float(self.leading_edge_w),
            "upper_surface": self.upper_surface.tolist(),
            "lower_surface": self.lower_surface.tolist(),
            "airfoil_id": float(self.airfoil_id)
        }

    @staticmethod
    def from_dict(d: dict) -> "Airfoil":
        """Creates an Airfoil instance from a dictionary representation."""
        return Airfoil(
            leading_edge_w=d["leading_edge_w"],
            upper_surface=np.array(d["upper_surface"]),
            lower_surface=np.array(d["lower_surface"]),
            airfoil_id=d.get("airfoil_id", -1)
        )

    def plot(self, n_points_per_side: int = 96) -> None:
        """Plots the airfoil using matplotlib.

        Args:
            n_points_per_side: Number of points to generate along each side of the airfoil.
        """

        coords = self.to_coordinates(n_points_per_side=n_points_per_side)
        plt.plot(coords[:, 0], coords[:, 1],
                label=f'Airfoil ID: {self.airfoil_id if self.airfoil_id is not None else "N/A"}')
        plt.axis('equal')
        plt.title('Airfoil Shape')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def interpolate_airfoil(x_rel: float,
                            airfoil1: Tuple[Airfoil, float],
                            airfoil2: Tuple[Airfoil, float]) -> Tuple[Airfoil, float]:
        """Interpolates between two airfoils and their chord lengths at a given relative position.

        Args:
            x_rel: Relative position between airfoil1 and airfoil2 (0.0 = airfoil1, 1.0 = airfoil2).
            airfoil1: Tuple of (Airfoil instance, chord length).
            airfoil2: Tuple of (Airfoil instance, chord length).
        Returns:
            A Tuple of Airfoil instance with the interpolated weights and chord length.
        """
        assert len(airfoil1[0].lower_surface) == len(airfoil2[0].lower_surface) and \
            len(airfoil1[0].upper_surface) == len(airfoil2[0].upper_surface), \
            "Airfoils must have the same number of surface weights to interpolate."
        assert (0.0 <= x_rel <= 1.0), "x_rel must be between 0.0 and 1.0."

        leading_edge_w = (1 - x_rel) * airfoil1[0].leading_edge_w + x_rel * airfoil2[0].leading_edge_w
        upper_surface = (1 - x_rel) * airfoil1[0].upper_surface + x_rel * airfoil2[0].upper_surface
        lower_surface = (1 - x_rel) * airfoil1[0].lower_surface + x_rel * airfoil2[0].lower_surface
        chord = (1 - x_rel) * airfoil1[1] + x_rel * airfoil2[1]

        return Airfoil(
            airfoil_id=-1,  # New airfoil ID will be assigned later
            leading_edge_w=leading_edge_w,
            upper_surface=upper_surface,
            lower_surface=lower_surface
        ), chord



class AirfoilRepository:
    """
    Repository for airfoils stored in a CSV file.
    The CSV is expected to have columns for leading edge weight,
    upper surface weights, lower surface weights, and cluster probabilities.
    """

    def __init__(self, csv_path: Path):
        self.path: Path = csv_path
        self.airfoils_df: pd.DataFrame = pd.read_csv(self.path)
        self._leading_edge_w_col: str = "asb_leading_edge_weight"
        self._upper_surface_cols: List[str] = \
            [col for col in self.airfoils_df.columns if col.startswith("asb_upper_")]
        self._lower_surface_cols: List[str] = \
            [col for col in self.airfoils_df.columns if col.startswith("asb_lower_")]
        self._cluster_prob_cols: List[str] = \
            [col for col in self.airfoils_df.columns if col.startswith("gmm_prob_cluster_")]
        self.validate_dataframe()
        self.n_clusters: int = len(self._cluster_prob_cols)
        self.n_upper_weights: int = len(self._upper_surface_cols)
        self.n_lower_weights: int = len(self._lower_surface_cols)


    def validate_dataframe(self) -> None:
        assert self._leading_edge_w_col in self.airfoils_df.columns, \
        f"DataFrame must contain leading edge weight column '{self._leading_edge_w_col}'."
        assert self._upper_surface_cols, \
        "DataFrame must contain upper surface weight columns starting with 'asb_upper_'."
        assert self._lower_surface_cols, \
            "DataFrame must contain lower surface weight columns starting with 'asb_lower_'."
        assert self._cluster_prob_cols, \
            "DataFrame must contain cluster probability columns starting with 'gmm_prob_cluster_'."

    def get_airfoil_by_id(self, airfoil_id: int) -> Airfoil:
        """Retrieves an Airfoil instance by its ID.

        Args:
            airfoil_id: The ID of the airfoil to retrieve.

        Returns:
            An Airfoil instance.
        """
        row: pd.Series = self.airfoils_df.iloc[airfoil_id]
        leading_edge_w = row[self._leading_edge_w_col]
        upper_surface = row[self._upper_surface_cols].to_numpy()
        lower_surface = row[self._lower_surface_cols].to_numpy()

        return Airfoil(
            airfoil_id=airfoil_id,
            leading_edge_w=leading_edge_w,
            upper_surface=upper_surface,
            lower_surface=lower_surface
        )

    def get_airfoils_by_ids(self, airfoil_ids: np.ndarray[int]) -> np.ndarray[Airfoil]:
        """Retrieves multiple Airfoil instances by their IDs."""
        return np.array([self.get_airfoil_by_id(airfoil_id) for airfoil_id in airfoil_ids])

    def sample_airfoil(self, cluster_id: int) -> Airfoil:
        """Samples an Airfoil instance from a specific cluster given the cluster probabilities.

        Args:
            cluster_id: The ID of the cluster to sample from.

        Returns:
            An Airfoil instance.
        """
        if cluster_id < 0 or cluster_id >= self.n_clusters:
            raise ValueError(f"cluster_id must be between 0 and {self.n_clusters - 1}.")

        cluster_col = self._cluster_prob_cols[cluster_id]
        cluster_probs = self.airfoils_df[cluster_col].to_numpy()

        #drawing form the csv probs p(x|k), as k is the cluster id
        #normalize first per cluster to sample from
        sampled_idx = np.random.choice(len(self.airfoils_df), p=cluster_probs / np.sum(cluster_probs))

        return self.get_airfoil_by_id(sampled_idx)

    def sample_airfoils(self, cluster_ids: np.ndarray[int]) -> np.ndarray[Airfoil]:
        """Samples multiple Airfoils given a list of cluster IDs."""
        return np.array([self.sample_airfoil(cluster_id) for cluster_id in cluster_ids])

def main():
    airfoilrepo = AirfoilRepository("data/processed/airfoil_with_polars.csv")
    airfoil = airfoilrepo.sample_airfoil(cluster_id=0)
    airfoil.plot()
    airfoil2 = airfoilrepo.sample_airfoil(cluster_id=0)
    airfoil2.plot()
    airfoil_interp, _ = airfoil.interpolate_airfoil(0, (airfoil,1), (airfoil2,1))
    airfoil_interp.plot()


if __name__ == "__main__":
    main()
