from __future__ import annotations

from typing import Sequence

import numpy as np
from sectionproperties.analysis.section import Section
from sectionproperties.pre.geometry import Geometry
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon


def _extract_intersection_points(geometry) -> list[tuple[float, float]]:
    """Return point coordinates from a shapely intersection result."""
    if isinstance(geometry, Point):
        return [(geometry.x, geometry.y)]
    if isinstance(geometry, MultiPoint):
        return [(point.x, point.y) for point in geometry.geoms]
    if isinstance(geometry, GeometryCollection):
        points = []
        for geom in geometry.geoms:
            if isinstance(geom, Point):
                points.append((geom.x, geom.y))
        return points
    return []


def rotate_polygon_about_point(
    polygon: Polygon, angle_rad: float, origin: tuple[float, float]
) -> Polygon:
    """Rotate a shapely polygon about `origin` by `angle_rad`."""
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    ox, oy = origin

    def rot(x: float, y: float) -> tuple[float, float]:
        x0, y0 = x - ox, y - oy
        xr = x0 * cos_a - y0 * sin_a + ox
        yr = x0 * sin_a + y0 * cos_a + oy
        return xr, yr

    return Polygon([rot(x, y) for x, y in polygon.exterior.coords])


def make_open_thin_walled_section() -> Polygon:
    """Create a simple open C-channel polygon used for EI-rotation validation."""
    coords = [
        (0, 0),
        (1.5, 0),
        (1.5, 0.2),
        (0.2, 0.2),
        (0.2, 2.0),
        (0, 2.0),
    ]
    return Polygon(coords)


def rotate_ei_about_shear_center(
    original_ei: np.ndarray,
    twist_angles: np.ndarray,
    centroids: np.ndarray,
    shear_centers: np.ndarray,
    area: np.ndarray,
    young_modulus: float,
) -> np.ndarray:
    """Rotate EI about shear center using shift-rotate-shift (parallel-axis) operations.

    Input EI format is `(EIxx, EIyy, EIxy)` per station.
    """
    transformed_ei = []

    for i in range(len(twist_angles)):
        # EI_centroid = [[EIxx,   -EIxy],
        #                [-EIxy,   EIyy]]
        ei_c = np.array(
            [[original_ei[i][0], -original_ei[i][2]], [-original_ei[i][2], original_ei[i][1]]]
        )

        # Rotation matrix = [[cos(theta), -sin(theta)],
        #                    [sin(theta),  cos(theta)]]
        rot = np.array(
            [
                [np.cos(twist_angles[i]), -np.sin(twist_angles[i])],
                [np.sin(twist_angles[i]), np.cos(twist_angles[i])],
            ]
        )

        # d_f = [[dx],
        #        [dy]]
        d_f = np.array(
            [
                [shear_centers[i][0] - centroids[i][0]],
                [shear_centers[i][1] - centroids[i][1]],
            ]
        )

        # D = [[dy**2,   -dx*dy],
        #      [-dx*dy,   dx**2]]
        d_f_mat = d_f.T @ d_f * np.eye(2) - d_f @ d_f.T
        # Forward shift to shear center: EI_sc = EI_c + E * A * D
        ei_sc = ei_c + young_modulus * area[i] * d_f_mat
        # Rotate about shear center
        ei_sc_rot = rot @ ei_sc @ rot.T

        # Compute d_b for backward shift (centroid rotates as well)
        d_b = rot @ d_f
        d_b_mat = d_b.T @ d_b * np.eye(2) - d_b @ d_b.T
        # Backward shift: I = I' - A*d^2
        ei_final = ei_sc_rot - young_modulus * area[i] * d_b_mat

        transformed_ei.append((ei_final[0, 0], ei_final[1, 1], -ei_final[0, 1]))

    return np.array(transformed_ei)


def offset_airfoil_inward(coords: np.ndarray, offset_distance: float) -> np.ndarray:
    """Create inward offset of airfoil coordinates.

    Raises ValueError if offset produces invalid geometry.
    """
    # Create polygon from coordinates
    polygon = Polygon(coords)
    # Create inward offset (negative buffer)
    inner_polygon = polygon.buffer(-offset_distance, join_style="mitre")

    # Extract coordinates
    if hasattr(inner_polygon, "exterior"):
        inner_coords = list(inner_polygon.exterior.coords)[:-1]
        return np.array(inner_coords)

    raise ValueError("Offset resulted in invalid geometry")


def create_i_beam_polygon(
    inner_polygon: Polygon,
    chord_length: float,
    beam_position: float,
    beam_thickness: float,
    beam_cap_lengths: float,
) -> Polygon:
    """Create an I-beam-like internal spar polygon constrained by inner airfoil cavity."""
    beam_center_x = beam_position * chord_length
    buf = 0.001  # added to guarantee intersection near top

    # Calculate beam x-bounds
    beam_left_x = beam_center_x - beam_cap_lengths / 2 + buf
    beam_right_x = beam_center_x + beam_cap_lengths / 2 - buf
    minx, miny, maxx, maxy = inner_polygon.bounds

    # Create vertical lines at beam boundaries
    left_line = LineString([(beam_left_x, miny - 0.1), (beam_left_x, maxy + 0.1)])
    right_line = LineString([(beam_right_x, miny - 0.1), (beam_right_x, maxy + 0.1)])

    left_intersections = inner_polygon.boundary.intersection(left_line)
    right_intersections = inner_polygon.boundary.intersection(right_line)

    left_points = _extract_intersection_points(left_intersections)
    right_points = _extract_intersection_points(right_intersections)

    if len(left_points) < 2 or len(right_points) < 2:
        raise ValueError(
            f"Beam position {beam_position} doesn't intersect properly with airfoil cavity"
        )

    # Sort points by y-coordinate
    left_points.sort(key=lambda p: p[1])
    right_points.sort(key=lambda p: p[1])

    # Take the top and bottom intersection points
    left_bottom = left_points[0]
    left_top = left_points[-1]
    right_bottom = right_points[0]
    right_top = right_points[-1]

    flange = (beam_cap_lengths - beam_thickness) / 2

    beam_points = [
        (left_bottom[0], left_bottom[1]),
        (right_bottom[0], right_bottom[1]),
        (right_bottom[0], left_bottom[1] + beam_thickness),
        (right_bottom[0] - flange, left_bottom[1] + beam_thickness),
        (right_top[0] - flange, left_top[1] - beam_thickness),
        (right_top[0], left_top[1] - beam_thickness),
        (right_top[0], right_top[1]),
        (left_top[0], left_top[1]),
        (left_top[0], left_top[1] - beam_thickness),
        (left_top[0] + flange, left_top[1] - beam_thickness),
        (left_bottom[0] + flange, left_bottom[1] + beam_thickness),
        (left_bottom[0], left_bottom[1] + beam_thickness),
    ]

    # Create beam polygon
    beam_polygon = Polygon(beam_points)
    return beam_polygon.buffer(0.0005)


def create_rib_polygon(
    outer_polygon: Polygon,
    chord_length: float,
    beam_position: float,
    beam_thickness: float,
    hole_positions: Sequence[float] = (0.2, 0.4, 0.6),
    min_border: float = 0.02,
) -> Polygon:
    """Create a rib polygon by removing circular lightening holes from the outer airfoil."""
    _, miny, _, maxy = outer_polygon.bounds
    i_beam_x = beam_position * chord_length

    rib_polygon = outer_polygon
    for pos in hole_positions:
        x_pos = pos * chord_length
        vertical_line = LineString([(x_pos, miny - 0.1), (x_pos, maxy + 0.1)])
        intersections = outer_polygon.boundary.intersection(vertical_line)

        intersection_points = _extract_intersection_points(intersections)
        if len(intersection_points) < 2:
            continue
        intersection_points.sort(key=lambda p: p[1])
        top_point = intersection_points[-1]
        bottom_point = intersection_points[0]

        airfoil_width = top_point[1] - bottom_point[1]
        radius = min(
            (airfoil_width - 2 * min_border) / 2, abs(x_pos - i_beam_x) - beam_thickness / 2
        )
        if radius <= min_border:
            continue

        rib_circle = Point(x_pos, (top_point[1] + bottom_point[1]) / 2).buffer(
            radius, resolution=64
        )
        rib_polygon = rib_polygon.difference(rib_circle)

    if isinstance(rib_polygon, Polygon):
        return rib_polygon
    geometries = getattr(rib_polygon, "geoms", ())
    polygons = [geom for geom in geometries if isinstance(geom, Polygon)]
    if polygons:
        return max(polygons, key=lambda polygon_item: polygon_item.area)
    raise ValueError("Rib geometry creation failed: resulting geometry is not a Polygon.")


def get_airfoil_properties(airfoil_polygon: Polygon, material):
    """Mesh and solve section properties for a given polygon/material pair."""
    try:
        airfoil_geometry = Geometry(geom=airfoil_polygon, material=material)
        airfoil_geometry.create_mesh(mesh_sizes=[0.001])
        section = Section(geometry=airfoil_geometry)
        section.calculate_geometric_properties()
        section.calculate_warping_properties()

        shear_center = section.get_sc()
        area = section.get_area()
        centroid = section.get_c()
        second_moments = section.get_eic()
        torsion_constant = section.get_ej()
    except Exception as exc:
        print(f"Error calculating properties: {exc}")
        shear_center = None
        area = None
        centroid = None
        second_moments = None
        torsion_constant = None

    return shear_center, area, centroid, second_moments, torsion_constant


def compose_scaled_properties(
    rib_positions: np.ndarray,
    normalized_half_beam_thickness: float,
    scaled_rib_property,
    scaled_beam_property,
):
    """Compose piecewise rib/spar property arrays from scaled station values.

    The output layout mirrors the notebook spanwise assembly logic.
    """
    y = []
    scaled_shear_centers = []
    scaled_areas = []
    scaled_centroids = []
    scaled_second_moments_list = []
    scaled_torsion_constants = []

    for i, pos in enumerate(rib_positions):
        if pos - normalized_half_beam_thickness > 0:
            y.append(pos - normalized_half_beam_thickness - normalized_half_beam_thickness / 10)
            y.append(pos - normalized_half_beam_thickness + normalized_half_beam_thickness / 10)

            scaled_shear_centers.append(
                (scaled_beam_property[0][0][i], scaled_beam_property[0][1][i])
            )
            scaled_areas.append(scaled_beam_property[1][i])
            scaled_centroids.append((scaled_beam_property[2][0][i], scaled_beam_property[2][1][i]))
            scaled_torsion_constants.append(scaled_beam_property[3][i])
            scaled_second_moments_list.append(
                (
                    scaled_beam_property[4][0][i],
                    scaled_beam_property[4][1][i],
                    scaled_beam_property[4][2][i],
                )
            )

            scaled_shear_centers.append(
                (scaled_rib_property[0][0][i], scaled_rib_property[0][1][i])
            )
            scaled_areas.append(scaled_rib_property[1][i])
            scaled_centroids.append((scaled_rib_property[2][0][i], scaled_rib_property[2][1][i]))
            scaled_torsion_constants.append(scaled_rib_property[3][i])
            scaled_second_moments_list.append(
                (
                    scaled_rib_property[4][0][i],
                    scaled_rib_property[4][1][i],
                    scaled_rib_property[4][2][i],
                )
            )
        else:
            y.append(pos)
            scaled_shear_centers.append(
                (scaled_rib_property[0][0][i], scaled_rib_property[0][1][i])
            )
            scaled_areas.append(scaled_rib_property[1][i])
            scaled_centroids.append((scaled_rib_property[2][0][i], scaled_rib_property[2][1][i]))
            scaled_torsion_constants.append(scaled_rib_property[3][i])
            scaled_second_moments_list.append(
                (
                    scaled_rib_property[4][0][i],
                    scaled_rib_property[4][1][i],
                    scaled_rib_property[4][2][i],
                )
            )

        if pos + normalized_half_beam_thickness < 1:
            y.append(pos + normalized_half_beam_thickness - normalized_half_beam_thickness / 10)
            y.append(pos + normalized_half_beam_thickness + normalized_half_beam_thickness / 10)

            scaled_shear_centers.append(
                (scaled_rib_property[0][0][i], scaled_rib_property[0][1][i])
            )
            scaled_areas.append(scaled_rib_property[1][i])
            scaled_centroids.append((scaled_rib_property[2][0][i], scaled_rib_property[2][1][i]))
            scaled_torsion_constants.append(scaled_rib_property[3][i])
            scaled_second_moments_list.append(
                (
                    scaled_rib_property[4][0][i],
                    scaled_rib_property[4][1][i],
                    scaled_rib_property[4][2][i],
                )
            )

            scaled_shear_centers.append(
                (scaled_beam_property[0][0][i], scaled_beam_property[0][1][i])
            )
            scaled_areas.append(scaled_beam_property[1][i])
            scaled_centroids.append((scaled_beam_property[2][0][i], scaled_beam_property[2][1][i]))
            scaled_torsion_constants.append(scaled_beam_property[3][i])
            scaled_second_moments_list.append(
                (
                    scaled_beam_property[4][0][i],
                    scaled_beam_property[4][1][i],
                    scaled_beam_property[4][2][i],
                )
            )
        else:
            y.append(pos)
            scaled_shear_centers.append(
                (scaled_rib_property[0][0][i], scaled_rib_property[0][1][i])
            )
            scaled_areas.append(scaled_rib_property[1][i])
            scaled_centroids.append((scaled_rib_property[2][0][i], scaled_rib_property[2][1][i]))
            scaled_torsion_constants.append(scaled_rib_property[3][i])
            scaled_second_moments_list.append(
                (
                    scaled_rib_property[4][0][i],
                    scaled_rib_property[4][1][i],
                    scaled_rib_property[4][2][i],
                )
            )

    return (
        y,
        scaled_shear_centers,
        scaled_areas,
        scaled_centroids,
        scaled_torsion_constants,
        scaled_second_moments_list,
    )


def area_preserving_boxcar(func, out_y: np.ndarray):
    """Apply area-preserving boxcar averaging at target spanwise points."""
    y = np.asarray(func[0])
    f = np.asarray(func[1])

    out = np.empty_like(out_y, dtype=float)
    y_spacing = np.concatenate([[0], np.diff(out_y), [0]])
    for i, yi in enumerate(out_y):
        a, b = yi - y_spacing[i] / 2, yi + y_spacing[i + 1] / 2
        if b <= a:
            out[i] = np.nan
            continue

        k0 = np.searchsorted(y, a, side="right")
        k1 = np.searchsorted(y, b, side="left")
        xs = np.concatenate(([a], y[k0:k1], [b]))
        vs = np.interp(xs, y, f)
        out[i] = np.trapezoid(vs, xs) / (b - a)

    return [out_y, out]


def interpolate_smoothed(func, query_points: np.ndarray) -> np.ndarray:
    """Interpolate smoothed function values at query points."""
    y = np.asarray(func[0])
    f = np.asarray(func[1])
    return np.interp(query_points, y, f)
