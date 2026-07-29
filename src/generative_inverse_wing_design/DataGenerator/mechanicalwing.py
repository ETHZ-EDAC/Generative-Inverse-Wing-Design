"""Contains the MechanicalWing class for modeling aeroelastic effects on wings."""

from __future__ import annotations

from typing import Optional, Tuple

from aerosandbox.numpy import np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sectionproperties.analysis import Section
from sectionproperties.pre import Geometry
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.ops import split

from generative_inverse_wing_design.DataGenerator.airfoilrepository import Airfoil
from generative_inverse_wing_design.DataGenerator.appconfig import AppConfig
from generative_inverse_wing_design.DataGenerator.utils import (
    ChordFunction,
    DummyFunction,
    Function,
    TwistFunction,
)


class MechanicalWing:
    """Mechanical wing model (placeholder)."""

    def __init__(
        self,
        cfg: AppConfig,
        chord_function: ChordFunction,
        twist_function: TwistFunction,
        airfoils_at_ys: np.ndarray[Airfoil],
    ):
        self.cfg: AppConfig = cfg
        self.chord_function: ChordFunction = chord_function
        self.twist_function: TwistFunction = twist_function
        self.ys_over_half_span: np.ndarray = cfg.fixed.ys_over_half_span
        self.rib_positions: np.ndarray = cfg.fixed.rib_positions

        assert len(self.ys_over_half_span) == len(
            airfoils_at_ys
        ), "Number of airfoils must match number of spanwise stations."

        self.solved: bool = False
        # The mechanical Wing treats its properties as unrotated.
        # The public Values are rotated according to the wing twist values.
        # Deformation caused twist is neglected in this model.
        self._airfoils_at_ribs: np.ndarray[Tuple[Airfoil, float]] = self._get_airfoils_at_ribs(
            airfoils_at_ys
        )
        # these funcions are smoothed by an area preserving boxcar at rib positions
        # to avoid discontinuities in properties at rib positions
        self._EI: Optional[Function] = None
        self._GJ: Optional[Function] = None
        self._shear_center: Optional[Function] = None
        self._centroid: Optional[Function] = None
        # the area is left unsmoothed as smoothing is not meaningful for area
        self._area: Optional[Function] = None
        # mass of the entire wing
        self._mass: Optional[float] = None

    @property
    def EI(self) -> Function:
        """Bending stiffness function along the span."""
        if not self.solved:
            self.solve_for_properties()
        assert isinstance(self._EI, Function), "EI has not been calculated."
        rotated_EI = self._rotate_EI()
        return rotated_EI

    @property
    def GJ(self) -> Function:
        """Torsional stiffness function along the span."""
        if not self.solved:
            self.solve_for_properties()
        assert isinstance(self._GJ, Function), "GJ has not been calculated."
        # st. vernant constant is already calculated with respect to
        # the shear center and not affected by rotation?
        return self._GJ

    @property
    def shear_center(self) -> Function:
        """Shear center function along the span."""
        if not self.solved:
            self.solve_for_properties()
        assert isinstance(self._shear_center, Function), "Shear center has not been calculated."
        # does not need to be rotated bc shear center is defined in unrotated cross-section
        return self._shear_center

    @property
    def centroid(self) -> Function:
        """Centroid function along the span."""
        if not self.solved:
            self.solve_for_properties()
        assert isinstance(self._centroid, Function), "Centroid has not been calculated."
        assert isinstance(self._shear_center, Function), "Shear center has not been calculated."

        twist_angles = self.twist_function.evaluate(self.rib_positions)
        shear_centers = self._shear_center.evaluate(self.rib_positions)
        # rotate centroid about the negative twist as twist is defined positive nose up
        centroid_rotated = self._centroid.rotate(angle_deg=-twist_angles, rot_center=shear_centers)
        return centroid_rotated

    @property
    def area(self) -> Function:
        """Cross-sectional area function along the span."""
        if not self.solved:
            self.solve_for_properties()
        assert isinstance(self._area, Function), "Area has not been calculated."
        return self._area

    @property
    def mass(self) -> float:
        """Mass of the wing."""
        if not self.solved:
            self.solve_for_properties()
        assert isinstance(self._mass, float), "Mass has not been calculated."
        return self._mass

    def _rotate_EI(self) -> Function:
        """Rotates the bending stiffness matrix EI
        according to the twist angles around the shear center.
        It uses the parallel axis theorem (I' = I + A*d^2)
        to shift to the shear center, then rotates the EI matrix, and shifts back.
        """
        # help the type checker
        if not self.solved:
            self.solve_for_properties()
        assert isinstance(self._EI, Function), "EI has not been calculated."
        assert isinstance(self._centroid, Function), "Centroid has not been calculated."
        assert isinstance(self._shear_center, Function), "Shear center has not been calculated."
        assert isinstance(self._area, Function), "Area has not been calculated."

        original_EI = self._EI.evaluate(self.rib_positions)
        young_modulus = self.cfg.fixed.material.elastic_modulus
        centroids = self._centroid.evaluate(self.rib_positions)
        shear_centers = self._shear_center.evaluate(self.rib_positions)

        # rotate by the negative twist angle as twist is defined positive nose up
        twist_angles = self.twist_function.evaluate(self.rib_positions)
        twist_angles = -np.radians(twist_angles)

        # use smoothed area values as the Inertia values are smoothed
        area_func = self._area.area_preserving_boxcar(box_centers=self.rib_positions)
        area = area_func.evaluate(self.rib_positions)

        transformed_EI = []
        for i, y in enumerate(self.rib_positions):

            # EI_centroid = [[EIxx,   -EIxy],
            #               [-EIxy,   EIyy]]
            EI_c = np.array(
                [[original_EI[i][0], -original_EI[i][2]], [-original_EI[i][2], original_EI[i][1]]]
            )

            # Rotation matrix = [[cos(theta), -sin(theta)],
            #                   [sin(theta),  cos(theta)]]
            Rot = np.array(
                [
                    [np.cos(twist_angles[i]), -np.sin(twist_angles[i])],
                    [np.sin(twist_angles[i]), np.cos(twist_angles[i])],
                ]
            )
            # d_f = [[dx],
            #        [dy]]
            d_f = np.array(
                [[shear_centers[i][0] - centroids[i][0]], [shear_centers[i][1] - centroids[i][1]]]
            )

            # D = [[dy**2,   -dx*dy],
            #     [-dx*dy,   dx**2]]
            D_f = d_f.T @ d_f * np.eye(2) - d_f @ d_f.T

            # with this we can compute the parallel axis theorem term as
            # EI_shift = EI_c + E * A * D

            # forward shift
            EI_sc = EI_c + young_modulus * area[i] * D_f

            # rotate
            EI_sc_rot = Rot @ EI_sc @ Rot.T

            # compute d_b for backward shift
            # as the controid is rotated aswell we rotate the offset vecotor
            d_b = Rot @ d_f

            D_b = d_b.T @ d_b * np.eye(2) - d_b @ d_b.T

            # backward shift as I' = I + A*d^2  --> I = I' - A*d^2
            EI_final = EI_sc_rot - young_modulus * area[i] * D_b

            transformed_EI.append((EI_final[0, 0], EI_final[1, 1], -EI_final[0, 1]))

        transformed_EI_array = np.array(transformed_EI)
        return Function(x=self.rib_positions, y=transformed_EI_array)

    def solve_for_properties(self):
        """Solves for the wing structural properties at rib positions.
        The calculated Properties are with respect to the chord length at each rib position.
        """
        EI_list = []
        GJ_list = []
        shear_center_list = []
        centroid_list = []
        area_list = []
        property_lists = [shear_center_list, centroid_list, EI_list, GJ_list, area_list]
        y = []

        normalized_half_rib_thickness = self.cfg.fixed.rib_thickness / self.cfg.fixed.span

        def append_properties(y_list, pos, prop_lists, values):
            y_list.append(pos)
            for prop_list, value in zip(prop_lists, values):
                prop_list.append(value)

        def scaling_fallback(previous_chord, current_chord, previous_props):
            if previous_props is None or previous_chord is None:
                raise ValueError("No previous properties to scale from.")

            # Shear center and centroid scale linearly with chord
            lin_factor = current_chord / previous_chord
            shear_center = (previous_props[0][0] * lin_factor, previous_props[0][1] * lin_factor)
            centroid = (previous_props[1][0] * lin_factor, previous_props[1][1] * lin_factor)

            # Bending stiffness scales with chord^4
            bend_factor = (current_chord / previous_chord) ** 4
            bending_stiffness = tuple(b * bend_factor for b in previous_props[2])
            # Torsional stiffness scales with chord^4
            torsion_stiffness = previous_props[3] * bend_factor

            # Area scales with chord^2
            area_factor = (current_chord / previous_chord) ** 2
            area = previous_props[4] * area_factor

            return shear_center, centroid, bending_stiffness, torsion_stiffness, area

        previous_rib_props = None
        previous_spar_props = None
        previous_chord = None
        for i, pos in enumerate(self.rib_positions):
            # get airfoil and chord at rib position
            airfoil = self._airfoils_at_ribs[i][0]
            chord = self._airfoils_at_ribs[i][1]

            try:
                rib_props = self.get_rib_properties(airfoil, chord)
            except Exception as e:
                print(
                    f"Rib property calculation failed at rib index {i}, position {pos}: {e}\
                      Using scaling fallback."
                )
                rib_props = scaling_fallback(previous_chord, chord, previous_rib_props)
            try:
                spar_props = self.get_spar_properties(airfoil, chord)
            except Exception as e:
                print(
                    f"Spar property calculation failed at rib index {i}, position {pos}: {e}\
                      Using scaling fallback."
                )
                spar_props = scaling_fallback(previous_chord, chord, previous_spar_props)

            previous_rib_props = rib_props
            previous_spar_props = spar_props
            previous_chord = chord

            # for each rib position, add spar, rib, rib, spar properties
            # with rib thickness spacing
            if pos - normalized_half_rib_thickness > 0:
                spar_pos = pos - normalized_half_rib_thickness - normalized_half_rib_thickness / 10
                rib_pos = pos - normalized_half_rib_thickness + normalized_half_rib_thickness / 10
                append_properties(y, spar_pos, property_lists, spar_props)
                append_properties(y, rib_pos, property_lists, rib_props)
            else:
                append_properties(y, pos, property_lists, rib_props)

            if pos + normalized_half_rib_thickness < 1:
                spar_pos = pos + normalized_half_rib_thickness + normalized_half_rib_thickness / 10
                rib_pos = pos + normalized_half_rib_thickness - normalized_half_rib_thickness / 10
                append_properties(y, rib_pos, property_lists, rib_props)
                append_properties(y, spar_pos, property_lists, spar_props)
            else:
                append_properties(y, pos, property_lists, rib_props)

        y = np.array(y)
        shear_center_array = np.array(shear_center_list)
        centroid_array = np.array(centroid_list)
        EI_array = np.array(EI_list)
        GJ_array = np.array(GJ_list)
        area_array = np.array(area_list)

        self._shear_center = Function(x=y, y=shear_center_array)
        self._centroid = Function(x=y, y=centroid_array)
        self._EI = Function(x=y, y=EI_array)
        self._GJ = Function(x=y, y=GJ_array)
        self._area = Function(x=y, y=area_array)

        # smooth properties at rib positions
        self._shear_center = self._shear_center.area_preserving_boxcar(
            box_centers=self.rib_positions
        )
        self._centroid = self._centroid.area_preserving_boxcar(box_centers=self.rib_positions)
        self._EI = self._EI.area_preserving_boxcar(box_centers=self.rib_positions)
        self._GJ = self._GJ.area_preserving_boxcar(box_centers=self.rib_positions)

        # calculate mass
        # integrate over normalized func and multiply with half span and double
        # for full wing material volume
        volume = self._area.integrate_between(0, 1) * (self.cfg.fixed.span)
        self._mass = volume * self.cfg.fixed.material.density

        self.solved = True

    def get_rib_properties(self, airfoil: Airfoil, chord_length: float) -> Tuple:
        """Calculates rib properties using sectionproperties.
        Args:
            airfoil: Airfoil object representing the rib airfoil shape.
            chord_length: Chord length at the rib position.
        Returns:
            shear_center: Shear center location (x, y) [m].
            centroid: Centroid location (x, y) [m].
            bending_stiffness: Bending stiffness matrix (EIxx, EIyy, EIxy) [Nm^2].
            torsion_stiffness: Torsional stiffness [Nm^2].
        """
        rib_polygon = self._create_rib_polygon(airfoil, chord_length)

        # load geometry into sectionproperties
        rib_geom = Geometry(geom=rib_polygon)

        # Create mesh, and scale with chord length
        rib_geom.create_mesh(mesh_sizes=[0.001 * chord_length])

        # Create section
        rib_sec = Section(geometry=rib_geom)

        # Calculate properties
        rib_sec.calculate_geometric_properties()
        rib_sec.calculate_warping_properties()

        # Get results
        shear_center = rib_sec.get_sc()
        centroid = rib_sec.get_c()
        bending_moment_of_inertia = rib_sec.get_ic()
        polar_moment_of_inertia = rib_sec.get_j()
        area = rib_sec.get_area()

        # Calculate EI
        bending_stiffness = tuple(
            moment_direction * self.cfg.fixed.material.elastic_modulus
            for moment_direction in bending_moment_of_inertia
        )

        # Calculate GJ
        shear_modulus = self.cfg.fixed.material.elastic_modulus / (
            2 * (1 + self.cfg.fixed.material.poissons_ratio)
        )
        torsion_stiffness = polar_moment_of_inertia * shear_modulus

        return shear_center, centroid, bending_stiffness, torsion_stiffness, area

    def get_spar_properties(self, airfoil: Airfoil, chord_length: float) -> Tuple:
        """Calculates spar properties using sectionproperties.
        Args:
            airfoil: Airfoil object representing the spar airfoil shape.
            chord_length: Chord length at the spar position.
        Returns:
            shear_center: Shear center location (x, y) [m].
            centroid: Centroid location (x, y) [m].
            bending_stiffness: Bending stiffness matrix (EIxx, EIyy, EIxy) [Nm^2].
            torsion_stiffness: Torsional stiffness [Nm^2].
        """
        spar_polygon, _ = self._create_spar_polygon(airfoil, chord_length)

        # load geometry into sectionproperties
        spar_geom = Geometry(geom=spar_polygon)

        # Create mesh, and scale with chord length
        spar_geom.create_mesh(mesh_sizes=[0.001 * chord_length])

        # Create section
        spar_sec = Section(geometry=spar_geom)

        # Calculate properties
        spar_sec.calculate_geometric_properties()
        spar_sec.calculate_warping_properties()

        # Get results
        shear_center = spar_sec.get_sc()
        centroid = spar_sec.get_c()
        bending_moment_of_inertia = spar_sec.get_ic()
        polar_moment_of_inertia = spar_sec.get_j()
        area = spar_sec.get_area()

        # Calculate EI
        bending_stiffness = tuple(
            moment_direction * self.cfg.fixed.material.elastic_modulus
            for moment_direction in bending_moment_of_inertia
        )

        # Calculate GJ
        shear_modulus = self.cfg.fixed.material.elastic_modulus / (
            2 * (1 + self.cfg.fixed.material.poissons_ratio)
        )
        torsion_stiffness = polar_moment_of_inertia * shear_modulus

        return shear_center, centroid, bending_stiffness, torsion_stiffness, area

    def _get_airfoils_at_ribs(
        self, airfoils_at_ys: np.ndarray[Airfoil]
    ) -> np.ndarray[Airfoil, float]:
        """Gets the airfoils at the rib positions by interpolating between the given airfoils.

        Args:
            airfoils: Array of Airfoil objects at spanwise stations.

        Returns:
            Array of Airfoil objects at rib positions.
        """
        rib_airfoils = []
        chord_lengths_at_ys = self.chord_function.evaluate(self.ys_over_half_span)

        # help the type checker
        assert isinstance(
            chord_lengths_at_ys, np.ndarray
        ), "Chord function evaluation did not return np.ndarray"

        for rib_pos in self.rib_positions:
            # Find the two spanwise stations surrounding the rib position
            idx = np.searchsorted(self.ys_over_half_span, rib_pos)
            if idx == 0:
                rib_airfoils.append((airfoils_at_ys[0], chord_lengths_at_ys[0]))
            elif idx == len(self.ys_over_half_span):
                rib_airfoils.append((airfoils_at_ys[-1], chord_lengths_at_ys[-1]))
            else:
                # get cross-sectional position left and right of rib position
                y0 = self.ys_over_half_span[idx - 1]
                y1 = self.ys_over_half_span[idx]
                # get airfoils at those positions
                af1 = airfoils_at_ys[idx - 1]
                af2 = airfoils_at_ys[idx]
                # interpolate airfoil at rib position
                x_rel = (rib_pos - y0) / (y1 - y0)
                rib_airfoil, chord_length = Airfoil.interpolate_airfoil(
                    x_rel,
                    airfoil1=(af1, chord_lengths_at_ys[idx - 1]),
                    airfoil2=(af2, chord_lengths_at_ys[idx]),
                )
                rib_airfoils.append((rib_airfoil, chord_length))
        return np.array(rib_airfoils)

    def _create_rib_polygon(self, airfoil: Airfoil, chord_length: float) -> Polygon:
        """Creates a rib polygon with holes cut out based on the rib hole positions.
            The holes are circular and are located at specified relative positions along the chord.
            the radius is determined based on the airfoil thickness at that position
            while ensuring a minimum border around the holes and avoiding intersection with the spar web.
            The Holes have a minimum radius is also deterimened by the minimum border.
        Args:
            airfoil: The Airfoil object representing the airfoil shape.
            chord_length: The chord length at the rib position.
        Returns:
            A shapely Polygon representing the rib with holes.
        """

        outer_polygon = airfoil.to_polygon(chord_length=chord_length)
        _, miny, _, maxy = outer_polygon.bounds
        # relative positions along chord [0, 1]
        hole_positions = self.cfg.fixed.rib_hole_positions
        # min border around holes and also min radius of holes
        min_border = self.cfg.fixed.rib_min_border
        # to ensure no intersection with I-beam web and holes
        beam_web_thickness = self.cfg.fixed.spar_web_flange_thickness

        # need to adjust
        spar_x = self.cfg.fixed.spar_position * chord_length

        rib_polygon = outer_polygon
        try:
            cutouts = None
            for pos in hole_positions:
                x_pos = pos * chord_length
                vertical_line = LineString([(x_pos, miny - 0.1), (x_pos, maxy + 0.1)])
                intersections = outer_polygon.boundary.intersection(vertical_line)

                intersection_points = self._extract_intersection_points(intersections)
                if len(intersection_points) < 2:
                    continue
                # Sort intersection points by y-coordinate
                intersection_points.sort(key=lambda p: p[1])
                top_point = intersection_points[-1]
                bottom_point = intersection_points[0]

                airfoil_width = top_point[1] - bottom_point[1]

                radius = min(
                    (airfoil_width - 2 * min_border) / 2,
                    abs(x_pos - spar_x) - beam_web_thickness / 2,
                )
                if radius <= min_border:
                    continue  # Skip if radius is non-positive
                rib_circle = Point(x_pos, (top_point[1] + bottom_point[1]) / 2).buffer(
                    radius, resolution=64
                )
                # init cutouts
                if cutouts is None:
                    cutouts = rib_circle
                else:
                    # add to cutouts
                    cutouts = cutouts.union(rib_circle)
            # only cut if cutouts were created and are valid
            if cutouts is not None and cutouts.is_valid:
                rib_polygon = rib_polygon.difference(cutouts)
        except Exception as e:
            print(f"Error creating rib holes: {e}. Proceeding without holes.")

        return rib_polygon

    def _create_spar_polygon(self, airfoil: Airfoil, chord_length: float) -> Polygon:
        """Creates a spar polygon based on the airfoil shape and spar dimensions.
        Args:
            airfoil: The Airfoil object representing the airfoil shape.
            chord_length: The chord length at the spar position.
        Returns:
            A shapely Polygon representing the spar. With an integrated I beam.
            The beam has an adaptive height based on the airfoil cavity at the spar position.
            The rest or the beam parameters are taken from the config.
        """

        outer_polygon = airfoil.to_polygon(chord_length=chord_length)
        spar_flange_width = self.cfg.fixed.spar_flange_width
        skin_thickness = self.cfg.fixed.skin_thickness
        spar_web_flange_thickness = self.cfg.fixed.spar_web_flange_thickness

        spar_x = self.cfg.fixed.spar_position * chord_length

        inner_polygon = outer_polygon.buffer(-skin_thickness, join_style=2)
        spar_polygon = outer_polygon.difference(inner_polygon)
        spar_only_polygon = spar_polygon
        try:
            # Calculate beam x-bounds
            beam_left_x = spar_x - spar_flange_width / 2
            beam_right_x = spar_x + spar_flange_width / 2
            minx, miny, maxx, maxy = inner_polygon.bounds
            # Create vertical lines at beam boundaries
            left_line = LineString([(beam_left_x, miny - 0.1), (beam_left_x, maxy + 0.1)])
            right_line = LineString([(beam_right_x, miny - 0.1), (beam_right_x, maxy + 0.1)])

            # Find intersections with the inner polygon boundary
            left_intersections = inner_polygon.boundary.intersection(left_line)
            right_intersections = inner_polygon.boundary.intersection(right_line)

            left_points = self._extract_intersection_points(left_intersections)
            right_points = self._extract_intersection_points(right_intersections)

            if len(left_points) < 2 or len(right_points) < 2:
                raise ValueError(
                    f"Beam position {spar_x} doesn't intersect properly with airfoil cavity"
                )

            # Sort points by y-coordinate
            left_points.sort(key=lambda p: p[1])
            right_points.sort(key=lambda p: p[1])

            # Take the top and bottom intersection points
            left_bottom = left_points[0]  # Bottom point of the left flange
            left_top = left_points[-1]  # Top point of the left flange
            right_bottom = right_points[0]  # Bottom point of the right flange
            right_top = right_points[-1]  # Top point of the right flange

            max_bottom = max(left_bottom[1], right_bottom[1])
            min_top = min(left_top[1], right_top[1])

            half_flange = (spar_flange_width - spar_web_flange_thickness) / 2

            # Follow the contour of the I-beam shape
            spar_points = [
                # bottom left flange with overshoot
                (left_bottom[0], miny - 0.1),
                # bottom right flange with overshoot
                (right_bottom[0], miny - 0.1),
                # bottom right flange inner
                (right_bottom[0], max_bottom + spar_web_flange_thickness),
                # bottom right web inner
                (right_bottom[0] - half_flange, max_bottom + spar_web_flange_thickness),
                # top right web inner
                (right_top[0] - half_flange, min_top - spar_web_flange_thickness),
                # top right flange inner
                (right_top[0], min_top - spar_web_flange_thickness),
                # top right flange with overshoot
                (right_top[0], maxy + 0.1),
                # top left flange with overshoot
                (left_top[0], maxy + 0.1),
                # top left flange inner
                (left_top[0], min_top - spar_web_flange_thickness),
                # top left web inner
                (left_top[0] + half_flange, min_top - spar_web_flange_thickness),
                # bottom left web inner
                (left_bottom[0] + half_flange, max_bottom + spar_web_flange_thickness),
                # bottom left flange inner
                (left_bottom[0], max_bottom + spar_web_flange_thickness),
            ]

            # Create spar polygon, that overshoots so no cavity is left
            spar_only_polygon = Polygon(spar_points)
            bounding_box = Polygon(
                [
                    (minx - 0.15, miny - 0.15),
                    (maxx + 0.15, miny - 0.15),
                    (maxx + 0.15, maxy + 0.15),
                    (minx - 0.15, maxy + 0.15),
                ]
            )
            # buffer for numeric stability
            spar_only_polygon = spar_only_polygon.buffer(0)
            # check if spar_only_polygon is valid, else return spar_polygon without modifications
            if not spar_only_polygon.is_valid:
                raise ValueError("Spar polygon is not valid.")
            # generate negative buffer to create hole in spar where skin is
            negative = bounding_box.difference(outer_polygon)
            # buffer inwards into the middle of skin thickness
            negative_buffered = negative.buffer(skin_thickness / 2, join_style=2)
            # subtract the overshoot to leave the spar with correct flange curvature
            spar_only_polygon = spar_only_polygon.difference(negative_buffered)
            # buffer for numeric stability
            spar_only_polygon = spar_only_polygon.buffer(0)
            # union with existing spar polygon
            spar_polygon = spar_only_polygon.union(spar_polygon)
        except Exception as e:
            print(f"Error occurred while creating spar polygon: {e}, proceeding without spar.")

        return spar_polygon, spar_only_polygon

    @staticmethod
    def _extract_intersection_points(geometry) -> list[tuple[float, float]]:
        """Extract point-like coordinates from shapely intersections.

        Intersections against the airfoil boundary can produce points, line segments,
        or geometry collections depending on local tangencies and overlap. For line-like
        results, the endpoints are sufficient because downstream code only needs the
        lowest and highest intersection locations.
        """
        points: list[tuple[float, float]] = []

        def _collect(geom) -> None:
            if isinstance(geom, Point):
                points.append((geom.x, geom.y))
            elif isinstance(geom, MultiPoint):
                for point in geom.geoms:
                    points.append((point.x, point.y))
            elif isinstance(geom, LineString):
                coords = list(geom.coords)
                if coords:
                    points.append(tuple(coords[0]))
                    points.append(tuple(coords[-1]))
            elif isinstance(geom, MultiLineString):
                for line in geom.geoms:
                    _collect(line)
            elif isinstance(geom, GeometryCollection):
                for sub_geom in geom.geoms:
                    _collect(sub_geom)

        _collect(geometry)
        return list(dict.fromkeys(points))

    def reset(self) -> None:
        """Resets the solved properties."""
        self.solved = False
        self._EI = None
        self._GJ = None
        self._shear_center = None
        self._centroid = None
        self._area = None
        self._mass = None

        self.solved = False

    def plot_at_rib_positions(self) -> None:
        """Plots the rib and spar cross-sections at each rib position."""

        n = len(self.rib_positions)
        fig, axes = plt.subplots(n, 2, figsize=(8, 1 * n))
        fig.suptitle("Rib and Spar Cross-Sections at Rib Positions", fontsize=16)

        root_chord = self._airfoils_at_ribs[0][1]
        plot_color = "C0"
        sc_color = "C1"
        centroid_color = "C2"
        x_spar = self.cfg.fixed.spar_position * root_chord

        for i in range(n):
            airfoil, chord_length = self._airfoils_at_ribs[i]

            rib_polygon = self._create_rib_polygon(airfoil, chord_length)
            spar_polygon, _ = self._create_spar_polygon(airfoil, chord_length)

            x, y = rib_polygon.exterior.xy
            # shift so spar is alsway at same x-position
            shift = x_spar - self.cfg.fixed.spar_position * chord_length
            x = x + shift
            if i == 0:
                y_max = max(y)
                y_min = min(y)
            axes[i, 0].plot(x, y, color=plot_color, label=f"Rib-crosssection at pos {i}")
            # plot interiors --> rib holes
            for interior in rib_polygon.interiors:
                xi, yi = interior.xy
                xi = xi + shift
                axes[i, 0].plot(xi, yi, color=plot_color)
            # Shear center and centroid
            if self.solved:
                assert isinstance(self._centroid, Function), "Centroid function not calculated."

                sc = self.shear_center.evaluate(self.rib_positions[i])
                centroid = self._centroid.evaluate(self.rib_positions[i])
                axes[i, 0].scatter(
                    sc[0] + shift,
                    sc[1],
                    s=5,
                    color=sc_color,
                    label="Shear Center, after smoothing",
                )
                axes[i, 0].scatter(
                    centroid[0] + shift,
                    centroid[1],
                    s=5,
                    color=centroid_color,
                    label="Centroid, after smoothing",
                )
            axes[i, 0].set_xlim([-0.1 * root_chord, 1.1 * root_chord])
            axes[i, 0].set_ylim([y_min - 0.01, y_max + 0.01])
            axes[i, 0].set_aspect("equal", adjustable="box")

            x, y = spar_polygon.exterior.xy
            x = x + shift
            axes[i, 1].plot(x, y, color=plot_color, label=f"Spar-crosssection at pos {i}")
            # plot interiors --> spar
            for interior in spar_polygon.interiors:
                xi, yi = interior.xy
                xi = xi + shift
                axes[i, 1].plot(xi, yi, color=plot_color)
            # Shear center and centroid
            if self.solved:
                assert isinstance(self._centroid, Function), "Centroid function not calculated."

                sc = self.shear_center.evaluate(self.rib_positions[i])
                centroid = self._centroid.evaluate(self.rib_positions[i])
                axes[i, 1].scatter(
                    sc[0] + shift,
                    sc[1],
                    s=5,
                    color=sc_color,
                    label="Shear Center, after smoothing",
                )
                axes[i, 1].scatter(
                    centroid[0] + shift,
                    centroid[1],
                    s=5,
                    color=centroid_color,
                    label="Centroid, after smoothing",
                )
            axes[i, 1].set_xlim([-0.1 * root_chord, 1.1 * root_chord])
            axes[i, 1].set_ylim([y_min - 0.01, y_max + 0.01])
            axes[i, 1].set_aspect("equal", adjustable="box")

        axes[0, 0].set_title("Rib Cross-Sections")
        axes[0, 1].set_title("Spar Cross-Sections")

        # Get handles and labels from the last row
        handles, labels = axes[-1, 0].get_legend_handles_labels()

        # Place legend below the last row, centered
        fig.legend(
            handles[1:],
            labels[1:],
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=3,
            frameon=False,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)  # Make room for legend
        # plt.savefig("/home/jona/bachelor_thesis/GenerativeInverseWingDesign/references/mech_wing_cross_sections.svg", format="svg")
        plt.show()

    def plot_mesh_at_rib_position(self, rib_index: int) -> None:
        """Plots the rib and spar cross-sections at a specific rib position.
        Args:
            rib_index: Index of the rib position to plot.
        """
        assert 0 <= rib_index < len(self.rib_positions), "Rib index out of range."

        airfoil, chord_length = self._airfoils_at_ribs[rib_index]

        rib_polygon = self._create_rib_polygon(airfoil, chord_length)
        spar_polygon, _ = self._create_spar_polygon(airfoil, chord_length)

        rib_geom = Geometry(geom=rib_polygon, material=self.cfg.fixed.material)
        spar_geom = Geometry(geom=spar_polygon, material=self.cfg.fixed.material)

        # Create mesh
        rib_geom.create_mesh(mesh_sizes=[0.001 * chord_length])
        spar_geom.create_mesh(mesh_sizes=[0.001 * chord_length])

        # Create section and calculate properties
        rib_sec = Section(geometry=rib_geom)
        spar_sec = Section(geometry=spar_geom)

        # Calculate properties
        rib_sec.calculate_geometric_properties()
        rib_sec.calculate_warping_properties()
        spar_sec.calculate_geometric_properties()
        spar_sec.calculate_warping_properties()

        rib_sec.plot_centroids(title=f"Rib Cross-Section at Rib Index {rib_index}")
        # fig1 = axs1.get_figure()
        # fig1.savefig("/home/argos/Documents/jona_bachelor_thesis/GenerativeInverseWingDesign/references/rib_cross_section_mesh.svg", format="svg")
        spar_sec.plot_centroids(title=f"Spar Cross-Section at Rib Index {rib_index}")
        # fig2 = axs2.get_figure()
        # fig2.savefig("/home/argos/Documents/jona_bachelor_thesis/GenerativeInverseWingDesign/references/spar_cross_section_mesh.svg", format="svg")

    def _transform_coords(
        self,
        coords: np.ndarray,
        rot_center: np.ndarray,
        rotation_y: float,
        translation: np.ndarray,
    ) -> np.ndarray:
        """Applies rotation around the y-axis and translation to a set of 3D coordinates.
        Args:
            coords: Nx3 array of 3D coordinates.
            rotation_y: Rotation angle around the y-axis in degrees.
                note: the rotations is defined nose up so in positive y-direction
            translation: 1x3 array representing the translation vector.
        Returns:
            Transformed Nx3 array of 3D coordinates.
        """
        # Convert rotation angle from degrees to radians
        # not we use the positive angle here as the rotation is defined nose up
        # and we use the euler rotation matirx around y axis for that
        # (at other places we used the negative angle for the x axis rotation matrix)
        theta = np.radians(rotation_y)

        assert coords.shape[1] == 3, "Coordinates must be Nx3 array."
        assert translation.shape == (3,), "Translation must be a 1x3 array."
        assert rot_center.shape == (3,), "Rotation center must be a 1x3 array."

        # shift coords to rotation center
        shifted_coords = coords - rot_center
        # Rotation matrix around y-axis
        R = np.rotation_matrix_3D(theta, axis="y")
        # Apply rotation
        rotated_coords = shifted_coords @ R.T
        # Shift back and apply translation
        transformed_coords = rotated_coords + rot_center + translation

        return transformed_coords

    def plot_3D(
        self,
        spar_sweep: Optional[float] = None,
        include_twist: bool = False,
        heave_displacements: Optional[np.ndarray] = None,
        twist_displacements: Optional[np.ndarray] = None,
        scale: bool = False,
        pretty_plot: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """Plots a 3D representation of the wing structure.
        Args:
            spar_sweep: Sweep angle of the spar in degrees. If None, no sweep is applied.
            include_twist: Whether to include base twist in the plot.
            heave_displacement: Vertical displacement to apply to the wing structure.
            twist_displacement: Additional twist angle caused by aeroelastic deformation
                to apply along the span in degrees.
            scale: Whether to scale the displacements for better visualization.
            save_path: Optional path to save the plot. If None, the plot is not saved.
        Note:
            The plot only renders the correct surface order, for the set angles.
            Automatic ordering is disabled. No thanks to Matplotlib.
        """

        bluergba = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 0.7)
        orangergba = (1.0, 0.4980392156862745, 0.054901960784313725, 0.7)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        chords = self._airfoils_at_ribs[:, 1]

        xz_width = 3 * max(chords)
        ax.set_xlim([-xz_width / 2, xz_width / 2])
        ax.set_ylim([0, self.cfg.fixed.span / 2])
        ax.set_zlim([-xz_width / 2, xz_width / 2])

        if heave_displacements is not None:
            assert isinstance(
                heave_displacements, np.ndarray
            ), "Heave displacement must be a numpy array."
            assert len(heave_displacements) == len(
                self.ys_over_half_span
            ), "Heave displacement array length must match spanwise stations."
            heave_displacements = np.interp(
                self.rib_positions, self.ys_over_half_span, heave_displacements
            )
        else:
            heave_displacements = np.zeros_like(self.rib_positions)

        if include_twist:
            twists = (
                self.twist_function.evaluate(self.rib_positions)
                if include_twist
                else np.zeros_like(self.rib_positions)
            )
            if twist_displacements is not None:
                assert isinstance(
                    twist_displacements, np.ndarray
                ), "Twist displacement must be a numpy array."
                assert len(twist_displacements) == len(
                    self.ys_over_half_span
                ), "Tswist displacement array length must match spanwise stations."
                twist_disp_at_ribs = np.interp(
                    self.rib_positions, self.ys_over_half_span, twist_displacements
                )
                twists += twist_disp_at_ribs
        else:
            twists = np.zeros_like(self.rib_positions)

        if scale:
            twist_scale = 3
            twists = twist_scale * twists
            heave_displacement_scale = 2 * xz_width / self.cfg.fixed.span
            heave_displacements = heave_displacement_scale * heave_displacements

        shear_center = self.shear_center.evaluate(self.rib_positions)
        shear_center_coords = np.stack(
            (shear_center[:, 0], self.rib_positions * self.cfg.fixed.span / 2, shear_center[:, 1]),
            axis=-1,
        )

        slope = np.tan(np.radians(spar_sweep if spar_sweep is not None else 0))
        # calculate offsets, scale to true lengths and subtract spar position offset
        offsets = (
            slope * self.rib_positions * (self.cfg.fixed.span / 2)
            - chords * self.cfg.fixed.spar_position
        )

        # note the offsets are calculated for a straight spar but applied after rotateing
        # the offsets iself are not rotated, this only holds for small twist angles
        translation_coords = np.stack(
            (
                np.zeros_like(self.rib_positions) + offsets,
                np.zeros_like(self.rib_positions),
                heave_displacements,
            ),
            axis=-1,
        )

        all_faces = []
        face_colors = []
        edge_colors = []

        last_spar_coords_connect_transformed = None
        last_le_te_coords_transformed = None

        for i in range(len(self.rib_positions)):
            airfoil, chord_length = self._airfoils_at_ribs[i]
            rib_polygon = self._create_rib_polygon(airfoil, chord_length)
            _, spar_only_polygon = self._create_spar_polygon(airfoil, chord_length)

            # Get exterior coordinates
            x_spar, z_spar = spar_only_polygon.exterior.xy
            x_rib, z_rib = rib_polygon.exterior.xy

            rib_pos = self.rib_positions[i] * self.cfg.fixed.span / 2  # spanwise position

            spar_coords = np.stack((x_spar, np.full_like(x_spar, rib_pos), z_spar), axis=-1)
            rib_coords = np.stack((x_rib, np.full_like(x_rib, rib_pos), z_rib), axis=-1)

            le_coords = rib_coords[np.argmax(rib_coords[:, 0]), :]
            te_coords = rib_coords[np.argmin(rib_coords[:, 0]), :]
            le_te_coords = np.stack((le_coords, te_coords), axis=0)

            le_the_coords_transformed = self._transform_coords(
                coords=le_te_coords,
                rot_center=shear_center_coords[i, :],
                rotation_y=twists[i],
                translation=translation_coords[i, :],
            )

            # Apply twist and heave displacement
            rib_coords_transformed = self._transform_coords(
                coords=rib_coords,
                rot_center=shear_center_coords[i, :],
                rotation_y=twists[i],
                translation=translation_coords[i, :],
            )

            # Connect spar points between ribs, find relevant points fist
            # get the flange outer points first ot the I beam (4 points)
            spar_x_max = np.max(spar_coords[:, 0])
            spar_x_min = np.min(spar_coords[:, 0])
            x_max_idxs = [
                j for j, x in enumerate(spar_coords[:, 0]) if np.isclose(x, spar_x_max, atol=1e-6)
            ]
            x_min_idxs = [
                j for j, x in enumerate(spar_coords[:, 0]) if np.isclose(x, spar_x_min, atol=1e-6)
            ]

            spar_coords_x_min = spar_coords[x_min_idxs, :]
            spar_coords_x_max = spar_coords[x_max_idxs, :]

            spar_z_max_left = np.max(spar_coords_x_min[:, 2])
            spar_z_min_left = np.min(spar_coords_x_min[:, 2])
            spar_z_max_right = np.max(spar_coords_x_max[:, 2])
            spar_z_min_right = np.min(spar_coords_x_max[:, 2])

            max_bottom = max(spar_z_min_left, spar_z_min_right)
            min_top = min(spar_z_max_left, spar_z_max_right)

            spar_web_flange_thickness = self.cfg.fixed.spar_web_flange_thickness
            half_flange = (self.cfg.fixed.spar_flange_width - spar_web_flange_thickness) / 2

            spar_points = [
                # top left flange
                (spar_x_min, rib_pos, spar_z_max_left),
                # top left flange inner
                (spar_x_min, rib_pos, min_top - spar_web_flange_thickness),
                # top left web inner
                (spar_x_min + half_flange, rib_pos, min_top - spar_web_flange_thickness),
                # bottom left web inner
                (spar_x_min + half_flange, rib_pos, max_bottom + spar_web_flange_thickness),
                # bottom left flange inner
                (spar_x_min, rib_pos, max_bottom + spar_web_flange_thickness),
                # bottom left flange
                (spar_x_min, rib_pos, spar_z_min_left),
                # bottom right flange
                (spar_x_max, rib_pos, spar_z_min_right),
                # bottom right flange inner
                (spar_x_max, rib_pos, max_bottom + spar_web_flange_thickness),
                # bottom right web inner
                (spar_x_max - half_flange, rib_pos, max_bottom + spar_web_flange_thickness),
                # top right web inner
                (spar_x_max - half_flange, rib_pos, min_top - spar_web_flange_thickness),
                # top right flange inner
                (spar_x_max, rib_pos, min_top - spar_web_flange_thickness),
                # top right flange
                (spar_x_max, rib_pos, spar_z_max_right),
                # top left flange (to close the loop)
                (spar_x_min, rib_pos, spar_z_max_left),
            ]
            spar_coords_connect = np.array(spar_points)

            spar_coords_connect_transformed = self._transform_coords(
                coords=spar_coords_connect,
                rot_center=shear_center_coords[i, :],
                rotation_y=twists[i],
                translation=translation_coords[i, :],
            )

            # Add spar beam faces (between ribs)
            if last_spar_coords_connect_transformed is not None:
                n = min(
                    len(last_spar_coords_connect_transformed), len(spar_coords_connect_transformed)
                )
                for j in range(n - 1):
                    quad = [
                        last_spar_coords_connect_transformed[j],
                        last_spar_coords_connect_transformed[j + 1],
                        spar_coords_connect_transformed[j + 1],
                        spar_coords_connect_transformed[j],
                    ]
                    all_faces.append(quad)
                    face_colors.append(orangergba)  # Orange for spar
                    edge_colors.append("k")  # Black edges

            if last_le_te_coords_transformed is not None:
                ax.plot3D(
                    xs=[last_le_te_coords_transformed[0, 0], le_the_coords_transformed[0, 0]],
                    ys=[last_le_te_coords_transformed[0, 1], le_the_coords_transformed[0, 1]],
                    zs=[last_le_te_coords_transformed[0, 2], le_the_coords_transformed[0, 2]],
                    color="k",
                    linewidth=0.2,
                )
                ax.plot3D(
                    xs=[last_le_te_coords_transformed[1, 0], le_the_coords_transformed[1, 0]],
                    ys=[last_le_te_coords_transformed[1, 1], le_the_coords_transformed[1, 1]],
                    zs=[last_le_te_coords_transformed[1, 2], le_the_coords_transformed[1, 2]],
                    color="k",
                    linewidth=0.2,
                )

            last_spar_coords_connect_transformed = spar_coords_connect_transformed
            last_le_te_coords_transformed = le_the_coords_transformed

            # Split rib polygon at mean height
            minx, minz, maxx, maxz = rib_polygon.bounds
            mean_z = (minz + maxz) / 2

            # Create horizontal splitting line
            split_line = LineString([(minx - 1, mean_z), (maxx + 1, mean_z)])

            # Split the polygon
            try:
                split_result = split(rib_polygon, split_line)
                rib_halves = list(split_result.geoms)

                # Add filled faces for each half (without edges)
                for half in rib_halves:
                    x_half, z_half = half.exterior.xy
                    half_coords_3d = np.stack(
                        (x_half, np.full_like(x_half, rib_pos), z_half), axis=-1
                    )
                    half_coords_transformed = self._transform_coords(
                        coords=half_coords_3d,
                        rot_center=shear_center_coords[i, :],
                        rotation_y=twists[i],
                        translation=translation_coords[i, :],
                    )
                    all_faces.append(list(half_coords_transformed))
                    face_colors.append(bluergba)  # Blue for rib
                    edge_colors.append("none")  # No edges
            except:  # noqa: E722
                # Fallback: use whole polygon if split fails
                all_faces.append(list(rib_coords_transformed))
                face_colors.append(bluergba)  # Blue for rib
                edge_colors.append("none")

                # Add rib outline as a single face with no fill (only edges)
            all_faces.append(list(rib_coords_transformed))
            face_colors.append((0.0, 0.0, 0.0, 0.0))
            edge_colors.append("k")  # Black edges

            # Add rib holes as single faces with no fill (only edges)
            for interior in rib_polygon.interiors:
                xi, zi = interior.xy
                hole_coords = np.stack((xi, np.full_like(xi, rib_pos), zi), axis=-1)
                hole_coords_transformed = self._transform_coords(
                    coords=hole_coords,
                    rot_center=shear_center_coords[i, :],
                    rotation_y=twists[i],
                    translation=translation_coords[i, :],
                )
                all_faces.append(list(hole_coords_transformed))
                face_colors.append((0.0, 0.0, 0.0, 0.0))
                edge_colors.append("k")  # Black edges

        # Create single collection with all faces
        collection = Poly3DCollection(
            all_faces,
            facecolors=face_colors,
            edgecolors=edge_colors,
            linewidths=0.2,
        )
        collection._zsortfunc = lambda x: 0
        ax.add_collection3d(collection)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        if scale and not pretty_plot:
            ax.text2D(
                0.05,
                0.95,
                f"Twist scaled by {twist_scale}x, Heave scaled by {heave_displacement_scale:.2f}x",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
            ax.set_title("3D Wing Structure Representation")
        ax.view_init(elev=15, azim=35)
        if pretty_plot:
            ax.grid(False)
            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("none")
            ax.yaxis.pane.set_edgecolor("none")
            ax.zaxis.pane.set_edgecolor("none")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")

        if save_path is not None:
            plt.savefig(f"{save_path}.svg", dpi=600, format="svg")
        plt.show()

    def to_dict(self) -> dict:
        """Converts the MechanicalWing object to a dictionary representation.
        Returns:
            A dictionary containing the MechanicalWing attributes.
        """
        dummy = DummyFunction()

        _EI = self._EI.to_dict() if isinstance(self._EI, Function) else dummy.to_dict()
        _GJ = self._GJ.to_dict() if isinstance(self._GJ, Function) else dummy.to_dict()
        _shear_center = (
            self._shear_center.to_dict()
            if isinstance(self._shear_center, Function)
            else dummy.to_dict()
        )
        _centroid = (
            self._centroid.to_dict() if isinstance(self._centroid, Function) else dummy.to_dict()
        )
        _area = self._area.to_dict() if isinstance(self._area, Function) else dummy.to_dict()

        _mass = float(self._mass) if isinstance(self._mass, float) else None

        airfoils_at_ribs: dict[str, dict[str, object]] = {}
        for i, af_chord in enumerate(self._airfoils_at_ribs):
            airfoils_at_ribs[f"airfoil_and_chord_{i}"] = {}
            airfoils_at_ribs[f"airfoil_and_chord_{i}"]["airfoil"] = af_chord[0].to_dict()
            airfoils_at_ribs[f"airfoil_and_chord_{i}"]["chord"] = float(af_chord[1])

        return {
            "airfoils_at_ribs": airfoils_at_ribs,
            "chord_function": self.chord_function.to_dict(),
            "twist_function": self.twist_function.to_dict(),
            "shear_center": _shear_center,
            "centroid": _centroid,
            "EI": _EI,
            "GJ": _GJ,
            "area": _area,
            "mass": _mass,
        }

    @staticmethod
    def from_dict(cfg: AppConfig, data: dict) -> "MechanicalWing":
        """Creates a MechanicalWing object from a dictionary representation.
        Args:
            cfg: Application configuration object.
            data: A dictionary containing the MechanicalWing attributes.
        Returns:
            A MechanicalWing object.
        """
        # Create an uninitialized MechanicalWing instance
        mech_wing = MechanicalWing.__new__(MechanicalWing)
        mech_wing.cfg = cfg
        mech_wing.ys_over_half_span = cfg.fixed.ys_over_half_span

        airfoils_at_ribs = []
        for i in range(len(data["airfoils_at_ribs"])):
            af_data = data["airfoils_at_ribs"][f"airfoil_and_chord_{i}"]
            airfoil = Airfoil.from_dict(af_data["airfoil"])
            chord = np.float64(af_data["chord"])
            airfoils_at_ribs.append((airfoil, chord))
        mech_wing._airfoils_at_ribs = np.array(airfoils_at_ribs)

        mech_wing.chord_function = ChordFunction.from_dict(data["chord_function"])
        mech_wing.twist_function = TwistFunction.from_dict(data["twist_function"])

        mech_wing._shear_center = Function.from_dict(data["shear_center"])
        mech_wing._centroid = Function.from_dict(data["centroid"])
        mech_wing._EI = Function.from_dict(data["EI"])
        mech_wing._GJ = Function.from_dict(data["GJ"])
        mech_wing._area = Function.from_dict(data["area"])
        mech_wing._mass = data["mass"]

        mech_wing.rib_positions = cfg.fixed.rib_positions

        mech_wing.solved = True

        return mech_wing
