from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import aerosandbox as asb
from aerosandbox.numpy import np
import casadi as cas

from generative_inverse_wing_design.DataGenerator.airfoilrepository import (
    AirfoilRepository,
)

BendingModel = Literal["shear", "moment"]


def late_taper_rational_bezier(
    c_root: float,
    c_tip: float,
    x_ctrl: float = 0.8,  # move toward 1.0 for a later drop
    weight: float = 2.0,  # >1 keeps chords high longer; <1 drops earlier; =1 is standard quadratic
    n_points: int = 5,
    spacing: str = "sine",  # "sine" or "uniform"
):
    """
    Rational quadratic Bézier chord distribution with control-point height fixed to c_root.

    Control points (with weights [1, w, 1]):
      P0 = (0, c_root)
      P1 = (x_ctrl, c_root)   # fixed height at root chord
      P2 = (1, c_tip)

    Parameters
    ----------
    c_root, c_tip : float
        Chords at root (x=0) and tip (x=1).
    x_ctrl : float in [0,1]
        x-position of the control point; higher pushes the drop later.
    weight : float > 0
        Rational weight for the control point.
        w > 1 -> stronger pull toward P1 (later, flatter taper).
        w = 1 -> standard quadratic Bézier.
        w < 1 -> weaker pull (earlier taper).
    n_points : int
        Number of stations to return.
    spacing : {"sine","uniform"}
        Placement of x-stations. "sine" uses sin(linspace(0, pi/2)) normalized.

    Returns
    -------
    x : (n_points,) np.ndarray
        Requested spanwise positions in [0,1].
    chords : (n_points,) np.ndarray
        Chords at those positions, evaluated exactly on the rational Bézier.
    ctrl_pts : np.ndarray shape (3,2)
        [[0, c_root], [x_ctrl, c_root], [1, c_tip]]
    weight : float
        The weight used for P1.
    """
    x_ctrl = float(np.clip(x_ctrl, 0.0, 1.0))
    w = float(max(weight, 1e-9))  # keep positive
    P0 = np.array([0.0, c_root])
    P1 = np.array([x_ctrl, c_root])  # fixed at root height
    P2 = np.array([1.0, c_tip])

    x = np.linspace(0.0, 1.0, n_points)

    # Invert rational x(t) so chords land exactly at the requested x.
    # With P0.x=0, P1.x=x_ctrl, P2.x=1 and weights [1, w, 1]:
    #   D(t)  = (1-t)^2 + 2w(1-t)t + t^2
    #   Nx(t) = 2w(1-t)t*x_ctrl + t^2
    #   x     = Nx / D
    # => quadratic: A t^2 + B t + C = 0  for each target x
    # where:
    #   A = (1 - 2w*x_ctrl) - 2x*(1 - w)
    #   B = (2w*x_ctrl) - 2x*(w - 1)
    #   C = -x
    A = (1.0 - 2.0 * w * x_ctrl) - 2.0 * x * (1.0 - w)
    B = (2.0 * w * x_ctrl) - 2.0 * x * (w - 1.0)
    C = -x

    disc = np.maximum(B * B - 4.0 * A * C, 0.0)

    # Solve per element, choose the root in [0,1]
    t1 = (-B + np.sqrt(disc)) / (2.0 * A + 1e-18)
    t2 = (-B - np.sqrt(disc)) / (2.0 * A + 1e-18)

    # Handle near-linear (A ≈ 0): fall back to B t + C = 0
    linear_mask = np.isclose(A, 0.0)
    t_lin = -C / (B + 1e-18)

    # Pick valid roots in [0,1]
    def pick_valid(t_candidate, t_alt, t_linear, is_linear):
        t = np.where(is_linear, t_linear, t_candidate)
        t_ok = (t >= -1e-9) & (t <= 1 + 1e-9)
        t = np.where(t_ok, t, t_alt)
        t_ok2 = (t >= -1e-9) & (t <= 1 + 1e-9)
        # if still invalid, clamp
        t = np.where(t_ok2, t, np.clip(t_candidate, 0.0, 1.0))
        return np.clip(t, 0.0, 1.0)

    # For monotone x(t) on [0,1], t1 is usually the increasing root; fall back to t2 when needed.
    t = pick_valid(t1, t2, t_lin, linear_mask)

    # Evaluate rational Bézier y(t)
    one_minus_t = 1.0 - t
    D = one_minus_t**2 + 2.0 * w * one_minus_t * t + t**2
    Ny = (one_minus_t**2 + 2.0 * w * one_minus_t * t) * c_root + (t**2) * c_tip
    chords = Ny / (D + 1e-18)

    return x, chords, np.vstack([P0, P1, P2])


def make_wing(
    span: float,
    ys_over_half_span: np.ndarray,
    chords: np.ndarray,
    twists: np.ndarray,
    offsets: np.ndarray | None = None,
    heave_displacements: np.ndarray | cas.MX | None = None,
    twist_displacements: np.ndarray | cas.MX | None = None,
    x_ref_over_chord: float = 0.33,
    airfoil: asb.Airfoil | None = None,
    airfoil_data_path: Path | None = None,
    airfoil_id: int = 730,
    color: str = "black",
) -> asb.Wing:
    """Build a symmetric wing from spanwise chord/twist/offset definitions."""
    if offsets is None:
        offsets = -chords / 4
    if heave_displacements is None:
        heave_displacements = np.zeros_like(ys_over_half_span)
    if twist_displacements is None:
        twist_displacements = np.zeros_like(ys_over_half_span)

    if airfoil is None:
        data_path = airfoil_data_path or Path("../../data/processed/airfoil_with_polars.csv")
        airfoil_repo = AirfoilRepository(data_path)
        airfoil = airfoil_repo.get_airfoil_by_id(airfoil_id).to_asb_airfoil()

    xsecs = []
    for i in range(len(ys_over_half_span)):
        xyz_le = np.array(
            [
                -chords[i] * x_ref_over_chord,
                ys_over_half_span[i] * (span / 2),
                0,
            ]
        )
        rotation_matrix = np.rotation_matrix_3D(
            angle=np.radians(twists[i] + twist_displacements[i]),
            axis="y",
        )
        xyz_le = np.array(
            [
                rotation_matrix[0, 0] * xyz_le[0]
                + rotation_matrix[0, 1] * xyz_le[1]
                + rotation_matrix[0, 2] * xyz_le[2],
                rotation_matrix[1, 0] * xyz_le[0]
                + rotation_matrix[1, 1] * xyz_le[1]
                + rotation_matrix[1, 2] * xyz_le[2],
                rotation_matrix[2, 0] * xyz_le[0]
                + rotation_matrix[2, 1] * xyz_le[1]
                + rotation_matrix[2, 2] * xyz_le[2],
            ]
        )
        xyz_le += np.array(
            [
                offsets[i] + chords[i] * x_ref_over_chord,
                0,
                heave_displacements[i],
            ]
        )

        xsecs.append(
            asb.WingXSec(
                xyz_le=xyz_le,
                chord=chords[i],
                twist=twists[i] + twist_displacements[i],
                airfoil=airfoil,
            )
        )

    return asb.Wing(
        symmetric=True,
        xsecs=xsecs,
        color=color,
    )


def solve_static_aeroelastic_equilibrium(
    make_wing_fn: Callable[..., asb.Wing],
    *,
    span: float,
    ys_over_half_span: np.ndarray,
    chords: np.ndarray,
    twists: np.ndarray,
    offsets: np.ndarray,
    EI: np.ndarray,
    GJ: np.ndarray,
    velocity: float = 50.0,
    alpha: float = 0.0,
    chordwise_resolution: int = 5,
    spanwise_resolution: int = 5,
    bending_model: BendingModel = "moment",
):
    """Solve coupled static aeroelastic equilibrium using shear- or moment-based bending."""
    ys = ys_over_half_span * (span / 2)
    EI_cas = cas.DM(EI)
    GJ_cas = cas.DM(GJ)

    opti = asb.Opti()

    u = opti.variable(init_guess=np.zeros_like(ys_over_half_span))
    theta = opti.variable(init_guess=np.zeros_like(ys_over_half_span))

    wing = make_wing_fn(
        span=span,
        ys_over_half_span=ys_over_half_span,
        chords=chords,
        twists=twists,
        offsets=offsets,
        heave_displacements=u,
        twist_displacements=theta * 180 / np.pi,
    )

    vlm = asb.VortexLatticeMethod(
        airplane=asb.Airplane(
            name="Aerostructures Test",
            xyz_ref=[0, 0, 0],
            wings=[wing],
        ),
        op_point=asb.OperatingPoint(
            velocity=velocity,
            alpha=alpha,
        ),
        chordwise_resolution=chordwise_resolution,
        spanwise_resolution=spanwise_resolution,
    )
    vlm.run()

    du = opti.derivative_of(
        variable=u,
        with_respect_to=ys,
        derivative_init_guess=np.zeros_like(u),
    )
    ddu = opti.derivative_of(
        variable=du,
        with_respect_to=ys,
        derivative_init_guess=np.zeros_like(u),
    )

    opti.subject_to(
        [
            u[0] == 0,
            du[0] == 0,
        ]
    )

    dtheta = opti.derivative_of(
        variable=theta,
        with_respect_to=ys,
        derivative_init_guess=np.zeros_like(theta),
    )
    opti.subject_to([theta[0] == 0])

    num_intervals = len(ys) - 1
    num_panels_per_interval = vlm.spanwise_resolution * vlm.chordwise_resolution
    num_panels = num_intervals * num_panels_per_interval

    loads = vlm.forces_geometry[:num_panels, :]
    vortex_centers = vlm.vortex_centers[:num_panels, :]

    if bending_model == "shear":
        loads_z = loads[:, 2]
        section_panel_loads = cas.sum2(
            cas.reshape(loads_z, (num_intervals, num_panels_per_interval))
        )
        shear_loads = cas.vertcat(cas.cumsum(section_panel_loads[::-1])[::-1], 0)
        opti.constrain_derivative(
            variable=EI_cas * ddu,
            with_respect_to=ys,
            derivative=-shear_loads,
        )
        opti.subject_to(ddu[-1] == 0)

    elif bending_model == "moment":
        section_bending_moments = []
        for sect_idx in range(num_intervals):
            start_idx = sect_idx * num_panels_per_interval
            section_bending_moment = 0
            for j in range(start_idx, num_panels):
                r_y = vortex_centers[j, 1] - ys[sect_idx]
                section_bending_moment += r_y * loads[j, 2]
            section_bending_moments.append(section_bending_moment)
        section_bending_moments = cas.vertcat(*section_bending_moments, 0)
        opti.subject_to(EI_cas * ddu == section_bending_moments)

    else:
        raise ValueError(f"Unknown bending_model '{bending_model}'. Use 'shear' or 'moment'.")

    section_torques = []
    for sect_idx in range(num_intervals):
        shear_center_x = offsets[sect_idx] + chords[sect_idx] * 0.33
        shear_center_pos = cas.horzcat(shear_center_x, ys[sect_idx], u[sect_idx])
        start_idx = sect_idx * num_panels_per_interval

        section_torque = 0
        for j in range(start_idx, num_panels):
            r_vector = vortex_centers[j, :] - shear_center_pos
            section_torque += r_vector[2] * loads[j, 0] - r_vector[0] * loads[j, 2]
        section_torques.append(section_torque)

    section_torques = cas.vertcat(*section_torques, 0)
    opti.subject_to(GJ_cas * dtheta == section_torques)

    sol = opti.solve()

    return {
        "opti": opti,
        "sol": sol,
        "wing": wing,
        "vlm": vlm,
        "u": u,
        "theta": theta,
        "du": du,
        "ddu": ddu,
        "dtheta": dtheta,
        "ys": ys,
        "EI": EI,
        "GJ": GJ,
    }


def summarize_internal_loads(solution: dict) -> dict[str, float]:
    """Compute max signed bending moment and torsion from a solved case."""
    sol = solution["sol"]
    ddu = solution["ddu"]
    dtheta = solution["dtheta"]
    EI = solution["EI"]
    GJ = solution["GJ"]

    bending_moments = sol.value(ddu) * EI
    torques = sol.value(dtheta) * GJ

    max_bending_moment = bending_moments[np.argmax(np.abs(bending_moments))]
    max_torsion_moment = torques[np.argmax(np.abs(torques))]

    return {
        "max_bending_moment": float(max_bending_moment),
        "max_torsion_moment": float(max_torsion_moment),
    }


def run_single_case(
    *,
    span: float,
    ys_over_half_span_case: np.ndarray,
    chord_fun,
    twist_fun,
    EI_func,
    GJ_func,
    chordwise_resolution: int,
    spanwise_resolution: int,
    bending_model: BendingModel = "moment",
) -> dict[str, float | int]:
    """Run one aeroelastic solve for a given structural/aerodynamic discretization.

    The function intentionally mirrors the notebook workflow while centralizing repeated
    logic. Comments are kept verbose so the mapping to the original notebook code is clear.
    """

    # For current N, interpolate geometry to the chosen structural grid.
    chords_case = chord_fun.evaluate(ys_over_half_span_case)
    twists_case = twist_fun.evaluate(ys_over_half_span_case)
    offsets_case = (np.tan(np.radians(1))) * ys_over_half_span_case * (
        span / 2
    ) - chords_case * 0.33

    # Interpolate structural properties to the same structural grid.
    EI_case = EI_func.evaluate(ys_over_half_span_case)
    GJ_case = GJ_func.evaluate(ys_over_half_span_case)

    # Solve the coupled aero-structural equilibrium.
    solution_case = solve_static_aeroelastic_equilibrium(
        make_wing_fn=make_wing,
        span=span,
        ys_over_half_span=ys_over_half_span_case,
        chords=chords_case,
        twists=twists_case,
        offsets=offsets_case,
        EI=EI_case,
        GJ=GJ_case,
        velocity=50,
        alpha=0,
        chordwise_resolution=int(chordwise_resolution),
        spanwise_resolution=int(spanwise_resolution),
        bending_model=bending_model,
    )

    sol_case = solution_case["sol"]
    u_case = solution_case["u"]
    theta_case = solution_case["theta"]

    # Store scalar outputs used throughout the convergence plots.
    tip_deflection_case = float(sol_case(u_case)[-1])
    max_torsion_case = float(np.max(np.abs(sol_case(theta_case) * 180 / np.pi)))

    # --> there are len(ys) - 1 structural intervals.
    # and num_intervals * (spanwise_resolution * chordwise_resolution) panels.
    num_intervals_case = len(ys_over_half_span_case) - 1
    num_panels_case = num_intervals_case * int(chordwise_resolution) * int(spanwise_resolution)

    return {
        "tip_deflection": tip_deflection_case,
        "max_torsion": max_torsion_case,
        "num_panels": int(num_panels_case),
    }


def run_panel_resolution_sweep(
    *,
    span: float,
    ys_over_half_span: np.ndarray,
    chord_fun,
    twist_fun,
    EI_func,
    GJ_func,
    chord_resolutions: np.ndarray,
    span_resolutions: np.ndarray,
    bending_model: BendingModel = "moment",
) -> list[dict]:
    """Run convergence sweep over aerodynamic panel resolution.

    Structural section count is held constant (same ys grid) while chordwise/spanwise
    aerodynamic paneling changes.
    """
    results: list[dict] = []

    # Test all combinations of chordwise and spanwise aerodynamic resolution.
    for chord_res in chord_resolutions:
        for span_res in span_resolutions:
            print(
                f"Running for chordwise resolution: {chord_res}, "
                f"spanwise resolution: {span_res}"
            )
            case_result = run_single_case(
                span=span,
                ys_over_half_span_case=ys_over_half_span,
                chord_fun=chord_fun,
                twist_fun=twist_fun,
                EI_func=EI_func,
                GJ_func=GJ_func,
                chordwise_resolution=int(chord_res),
                spanwise_resolution=int(span_res),
                bending_model=bending_model,
            )
            results.append(
                {
                    "chordwise_resolution": int(chord_res),
                    "spanwise_resolution": int(span_res),
                    **case_result,
                }
            )

    return results


def run_structural_resolution_sweep(
    *,
    span: float,
    chord_fun,
    twist_fun,
    EI_func,
    GJ_func,
    test_cases: list[tuple[int, int]],
    chordwise_panels_per_section: int = 5,
    bending_model: BendingModel = "moment",
) -> list[dict]:
    """Run convergence sweep over structural section count.

    Here the test cases define (n_sections, spanwise_resolution), while chordwise
    resolution is fixed. This keeps panel count in a comparable range.
    """
    results: list[dict] = []

    # Specific test cases: (n_sections, spanwise_resolution)
    for n_sections, spanwise_res in test_cases:
        print(
            f"\nRunning with {n_sections} structural sections "
            f"and {spanwise_res} spanwise resolution"
        )

        # Create structural grid for current case.
        ys_case = np.sin(np.linspace(0, 1, n_sections) * np.pi / 2)

        # --> there are n_sections - 1 intervals.
        # and num_intervals * spanwise_resolution * chordwise_panels_per_section panels.
        total_panels = (n_sections - 1) * int(spanwise_res) * int(chordwise_panels_per_section)

        print(f"  Chordwise resolution: {chordwise_panels_per_section}")
        print(f"  Total panels per half-span: {total_panels}")

        case_result = run_single_case(
            span=span,
            ys_over_half_span_case=ys_case,
            chord_fun=chord_fun,
            twist_fun=twist_fun,
            EI_func=EI_func,
            GJ_func=GJ_func,
            chordwise_resolution=int(chordwise_panels_per_section),
            spanwise_resolution=int(spanwise_res),
            bending_model=bending_model,
        )

        row = {
            "n_sections": int(n_sections),
            "spanwise_resolution": int(spanwise_res),
            "total_panels": int(total_panels),
            **case_result,
        }
        results.append(row)

        print(f"  Tip deflection: {row['tip_deflection']:.6f} m")
        print(f"  Max torsion: {row['max_torsion']:.6f} deg")

    return results
