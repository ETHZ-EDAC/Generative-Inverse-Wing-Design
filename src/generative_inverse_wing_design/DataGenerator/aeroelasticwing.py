"""Contains the AeroelasticWing class for modeling aeroelastic effects on wings."""

from __future__ import annotations

import json
from typing import Optional, Tuple
import uuid

import aerosandbox as asb
from aerosandbox.numpy import np
import aerosandbox.tools.pretty_plots as p
import casadi as cas
from casadi.casadi import MX
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import pandas as pd
import pyvista as pv

from generative_inverse_wing_design.DataGenerator.airfoilrepository import (
    Airfoil,
    AirfoilRepository,
)
from generative_inverse_wing_design.DataGenerator.appconfig import AppConfig
from generative_inverse_wing_design.DataGenerator.mechanicalwing import MechanicalWing
from generative_inverse_wing_design.DataGenerator.utils import (
    ChordFunction,
    TwistFunction,
)


class AeroelasticWing:
    """Aeroelastic wing model.
    Args:
        cfg: Application configuration object.
        sampled_params: Dictionary of sampled parameters for the wing.
        airfoils: Array of Airfoil objects at spanwise stations.
    Returns:
        An AeroelasticWing object.
    """

    def __init__(self, cfg: AppConfig, sampled_params: dict, airfoils: np.ndarray[Airfoil]):
        self.cfg: AppConfig = cfg
        self.id: str = uuid.uuid4().hex
        self.ys_over_half_span: np.ndarray = cfg.fixed.ys_over_half_span
        self.span: float = cfg.fixed.span
        self.airfoils: np.ndarray[Airfoil] = airfoils

        config_keys = set(k[:-5] for k in vars(self.cfg.variables).keys()) # Exclude _dist in key names
        sampled_keys = set(sampled_params.keys())
        assert sampled_keys == config_keys, \
            f"Sampled params keys {sampled_keys} do not match config variable keys {config_keys}"
        assert len(self.ys_over_half_span) == len(airfoils), \
            "Number of airfoils must match number of spanwise stations."

        self.chord_function = ChordFunction(
            chord_ctrl_pos=float(sampled_params["chord_ctrl_pos"]),
            chord_weight_ctrl=float(sampled_params["chord_weight_ctrl"]),
            chord_root=float(self.cfg.fixed.root_chord),
            chord_tip=float(sampled_params["chord_tip"])
        )
        self.twist_function = TwistFunction(
            twist_root=float(sampled_params["twist_root"]),
            twist_tip=float(sampled_params["twist_tip"]),
            twist_mean_pos=float(sampled_params["twist_mean_pos"])
        )

        self.spar_sweep = float(sampled_params["spar_sweep"])

        self.mechanical_wing = MechanicalWing(cfg, self.chord_function, self.twist_function, self.airfoils)

        self.asb_wing: Optional[asb.Wing] = None
        self.heave_displacements = np.zeros_like(self.ys_over_half_span)
        self.twist_displacements = np.zeros_like(self.ys_over_half_span)
        self.performance: dict = {}
        self.solved: bool = False
        self._max_stress_point: Optional[np.ndarray] = None

    def _to_asb_wing(
        self, heave_displacements: MX | np.ndarray,
        twist_displacements: MX | np.ndarray
        ) -> asb.Wing:
        """
        Generates a AeroSandbox wing based on a given set of per-cross-section characteristics.
        This function is adapted from Aerosandbox's Github tutorial.
        Args:
            heave_displacements: Array of heave displacements at spanwise stations [m].
            twist_displacements: Array of twist displacements at spanwise stations [degrees].
        """
        assert self.mechanical_wing.solved,\
                "Mechanical wing properties must be solved before creating the aeroelastic wing."

        chords = self.chord_function.evaluate(self.ys_over_half_span)
        twists = self.twist_function.evaluate(self.ys_over_half_span)
        shear_centers = self.mechanical_wing.shear_center.evaluate(self.ys_over_half_span)
        # Calculate leading edge offsets caused by spar sweep
        slope = np.tan(np.radians(self.spar_sweep))
        # calculate offsets, scale to true lengths and subtract spar position offset
        offsets = slope * self.ys_over_half_span * (self.span / 2) - chords * self.cfg.fixed.spar_position

        xsecs = []

        for i, pos in enumerate(self.ys_over_half_span):
            #first shift to shear center
            xyz_le = np.array([
                -shear_centers[i][0],
                pos * (self.span / 2),
                -shear_centers[i][1]
            ])
            #rotate around y-axis for twist
            xyz_le = np.rotation_matrix_3D(
                angle=np.radians(twists[i] + twist_displacements[i]),
                axis="y"
            ) @ xyz_le
            #shift back from shear center and apply heave and offset
            #the offset shift is neglecting the rotation due to small angles
            xyz_le += np.array([
                offsets[i] + shear_centers[i][0],
                0,
                heave_displacements[i] + shear_centers[i][1]
            ])

            xsecs.append(
                asb.WingXSec(
                    xyz_le=xyz_le,
                    chord=chords[i],
                    twist=twists[i] + twist_displacements[i],
                    airfoil=self.airfoils[i].to_asb_airfoil(),
                )
            )

        asb_wing = asb.Wing(
            symmetric=True,
            xsecs=xsecs,
            color="C0", # light gray
        )

        return asb_wing

    def set_asb_wing(self) -> asb.Wing:
        """Sets the AeroSandbox wing based on current heave and twist displacements."""

        self.asb_wing = self._to_asb_wing(
            heave_displacements=self.heave_displacements,
            twist_displacements=self.twist_displacements
        )
        return self.asb_wing

    def _compute_max_stress(self,
                            bending_moments: np.ndarray,
                            EIxx: np.ndarray,
                            torques: np.ndarray,
                            GJ: np.ndarray) -> Tuple[float, np.ndarray]:
        """Computes the maximum van Mises stress in the wing structure.
        Assuemes the maximum stress is governed by the bending. Therfore the stress
        is computed at the point the farthest from the neutral xx-axis in bending.
        also assumes that the arrays are given at ys_over_half_span positions.
        Also assumes that the maximum stress formulas:
            sigma_bending = M*y/I
            tau_torsion = T*r/J
        Note that this is a simplified approach that is not generall not accurate
        for thin-walled hollow cross-sections.
        Args:
            bending_moments: Array of bending moments at ys_over_half_span [Nm].
            EIxx: Array of bending stiffness at ys_over_half_span [Nm^2].
            torques: Array of torques at ys_over_half_span [Nm].
            GJ: Array of torsional stiffness at ys_over_half_span [Nm^2].
        Returns:
            Maximum Van Mises stress value [Pa].
            max_stress_point: Point (x, y_rel_half_span, z) where maximum stress occurs.
        """

        assert self.mechanical_wing.solved, \
            "Mechanical wing properties must be solved before computing maximum stress."

        chords = self.chord_function.evaluate(self.ys_over_half_span)
        shear_centers = self.mechanical_wing.shear_center.evaluate(self.ys_over_half_span)
        centroids = self.mechanical_wing.centroid.evaluate(self.ys_over_half_span)
        angles_of_incidence = self.twist_function.evaluate(self.ys_over_half_span) + self.twist_displacements

        #calculate the points with maximum distance in y-direction from centroid
        max_dist_points_array = np.empty((len(self.ys_over_half_span), 2))
        for i in range(len(self.ys_over_half_span)):
            airfoil: Airfoil = self.airfoils[i]
            chord: float = chords[i]
            centroid: np.ndarray = centroids[i]
            angle_of_incidence: float = angles_of_incidence[i]
            shear_center: np.ndarray = shear_centers[i]

            min_point, max_point = airfoil.min_max_y_coords(chord_length=chord,
                                                            incidence_angle=angle_of_incidence,
                                                            center_of_rotation=shear_center)
            y_dist_to_min = min_point[1] - centroid[1]
            y_dist_to_max = max_point[1] - centroid[1]
            if abs(y_dist_to_min) > abs(y_dist_to_max):
                max_dist_points_array[i, :] = min_point
            else:
                max_dist_points_array[i, :] = max_point

        youngs_modulus = self.cfg.fixed.material.elastic_modulus
        shear_modulus = youngs_modulus / (2 * (1 + self.cfg.fixed.material.poissons_ratio))

        sigma_bending_array = (bending_moments*youngs_modulus / EIxx) *  max_dist_points_array[:, 1]

        shear_to_max_point_dist = np.linalg.norm(max_dist_points_array - shear_centers, axis=1) #is norming along axis 1 coorect?
        tau_torsion_array = (torques*shear_modulus / GJ) * shear_to_max_point_dist

        #van mises stress sqrt(sigma^2 + 3*tau^2)
        von_mises_stress_array = np.sqrt(sigma_bending_array**2 + 3 * tau_torsion_array**2)
        max_stress_index = np.argmax(von_mises_stress_array)
        max_stress = von_mises_stress_array[max_stress_index]
        max_stress_point = max_dist_points_array[max_stress_index, :]

        max_stress_point = np.array([
            max_stress_point[0],
            self.ys_over_half_span[max_stress_index],
            max_stress_point[1]
        ])

        return max_stress, max_stress_point



    def solve_all_my_problems(self, 
                              heave_init: Optional[np.ndarray] = None, 
                              twist_init: Optional[np.ndarray] = None) -> None:
        """Placeholder method to solve aeroelastic problems."""

        assert self.mechanical_wing.solved, \
            "Mechanical wing properties must be solved before aeroelastic analysis."

        ys = self.ys_over_half_span * (self.span / 2)
        chords = self.chord_function.evaluate(self.ys_over_half_span)
        offsets = np.tan(np.radians(self.spar_sweep)) * ys - chords * self.cfg.fixed.spar_position
        sc = self.mechanical_wing.shear_center.evaluate(self.ys_over_half_span)
        sc_x = sc[:, 0]
        sc_z = sc[:, 1]

        optimizer = asb.Opti()

        u = optimizer.variable(
            init_guess=heave_init if heave_init is not None else np.zeros_like(ys),
        )

        theta = optimizer.variable(
            init_guess=twist_init if twist_init is not None else np.zeros_like(ys),
        )

        wing = self._to_asb_wing(heave_displacements=u, twist_displacements=(theta)*180/np.pi)

        simulation = asb.VortexLatticeMethod(
            airplane=asb.Airplane(
                name="Aerostructures Test",
                xyz_ref=[0, 0, 0],
                wings=[wing],
            ),
            op_point=asb.OperatingPoint(
                velocity=self.cfg.operating_point.velocity,
                alpha=0,
            ),
            chordwise_resolution=self.cfg.operating_point.chordwise_resolution,
            spanwise_resolution=self.cfg.operating_point.spanwise_resolution,
        )
        simulation.run()

        #Access EIxx and GJ values at spanwise stations
        # The mechanical Wing treats its properties as unrotated.
        # The public Values are rotated according to the wing twist values.
        # Deformation caused twist is neglected in this model.
        EI = self.mechanical_wing.EI.evaluate(self.ys_over_half_span)[:, 0]
        GJ = self.mechanical_wing.GJ.evaluate(self.ys_over_half_span)

        ### Do the heave structures analysis
        du = optimizer.derivative_of(
            variable=u, with_respect_to=ys,
            derivative_init_guess=np.zeros_like(u),
        )
        ddu = optimizer.derivative_of(
            variable=du, with_respect_to=ys,
            derivative_init_guess=np.zeros_like(u),
        )

        optimizer.subject_to([  # Add the boundary conditions
            u[0] == 0,
            du[0] == 0,
            #ddu[-1] == 0, --> is enforced ba M[-1]=0 zero moment at tip
        ])

        # --> there are len(ys) - 1 intervals
        num_intervals = (len(ys) - 1)
        num_panels_per_interval = simulation.spanwise_resolution * simulation.chordwise_resolution
        # and num_intervals * num_panels_per_interval pannels
        num_panels = num_intervals * num_panels_per_interval
        # access z-loads from simulation, each force corresponds to a segment between two ys
        loads = simulation.forces_geometry[:num_panels, :]

        # previous approach
        # get only the z-loads for heave displacement calculation
        # loads_z = simulation.forces_geometry[:num_panels, 2]
        # summ up panel loads inbetween the ys points
        # section_panel_loads = cas.sum2(cas.reshape(loads_z, (num_intervals, num_panels_per_interval)))
        # shear is given by V(y) = integral from y to b of q(y') dy'
        # --> summ up panel loads from tip to ys
        # shear_loads = cas.vertcat(cas.cumsum(section_panel_loads[::-1])[::-1], 0)

        # ### Link the aerodynamic and heave analyses together
        # optimizer.constrain_derivative(
        #     variable=EI * ddu, with_respect_to=ys,
        #     derivative= -shear_loads
        # )

        ### Do the torsion structures analysis
        dtheta = optimizer.derivative_of(
            variable=theta, with_respect_to=ys,
            derivative_init_guess=np.zeros_like(theta),
        )

        optimizer.subject_to([  # Add the boundary conditions
            theta[0] == 0,
            #dtheta[-1] == 0, --> is enforced by T[-1]=0 zero torque at tip
        ])


        #get the vortex centers to compute the torque at the shear center for each section
        vortex_centers = simulation.vortex_centers[:num_panels, :]
        section_torques = []
        section_bending_moments = []

        for sect_idx in range(num_intervals):
            # Get shear center position for this section (interpolate from reference)
            # the offset shift is neglecting the rotation due to small angles
            shear_center_pos = cas.horzcat(sc_x[sect_idx] + offsets[sect_idx],
                                           ys[sect_idx],
                                           sc_z[sect_idx] + u[sect_idx])

            # Get panels belonging to this section
            start_idx = sect_idx * num_panels_per_interval

            section_torque = 0
            section_bending_moment = 0
            for j in range(start_idx, num_panels):
                # Position vector from shear center to vortex center
                r_vector = vortex_centers[j, :] - shear_center_pos

                # y_distance from vortex center to section(bending) --> take only lift as simplification
                r_y = vortex_centers[j, 1] - ys[sect_idx]

                # Cross product: r × F (only take y-component for torsion about span)
                # (r × F)_y = r_z * F_x - r_x * F_z
                torque = r_vector[2] * loads[j, 0] - r_vector[0] * loads[j, 2]
                section_torque += torque

                # Bending Moment M(y) = integral from y to b of r_y * q(y') dy'
                # --> summ up panel forces multiplied by their distance in y-direction to the section
                moment = r_y * loads[j, 2] # only take lift as simplification
                section_bending_moment += moment

            section_torques.append(section_torque)
            section_bending_moments.append(section_bending_moment)

        # Convert to CasADi vector
        section_torques = cas.vertcat(*section_torques, 0)
        section_bending_moments = cas.vertcat(*section_bending_moments, 0)


        # Direct equality constraint: GJ·dθ/dy = T(y)
        optimizer.subject_to(GJ * dtheta == section_torques)

        # Direct equality constraint: EI·d²u/dy² = M(y)
        optimizer.subject_to(EI * ddu == section_bending_moments)

        ### Solve the system
        sol = optimizer.solve(verbose=False)

        # Update displacement variables with solved values
        self.heave_displacements = sol.value(u)
        self.twist_displacements = sol.value(theta)*180/np.pi

        bending_moments = sol.value(ddu) * EI
        torques = sol.value(dtheta) * GJ
        """
        To evaluate static stability, we start with the principle of virtual work:
        an equilibrium state requires:

        δWint=δWext for virtual displacements δu, δtheta

        δWint=∫(EI⋅u′′⋅δu′′+GJ⋅θ′⋅δθ′)dy where u′′=d²u/dy², θ′=dθ/dy

        we can formulate this in code as:
        dWint = [δu    ]T    [D_u.T @ W * EI @ D_u            0               ]     [u    ]
                [δtheta]  @  [        0           D_theta.T @ W * GJ @ D_theta]  @  [theta]
        where D_u = ∂(d²u/dy²)/∂u, D_theta = ∂(dθ/dy)/∂θ, W = integration weights

        Note that the internal work stiffness matrix has no coupling terms, 
        as bending work only depends on u, and torsion work only on theta,
        this is a simplification as EI(theta), but in our framework we calculate EI only based
        on the undeformed twist.

        δWext=∫Maero(u,θ)⋅δu′′dy+∫Taero(u,θ)⋅δθ′dy, where Maero, Taero are the aerodynamic moments/torques

        we can formulate this in code as:
        dWext = [δu    ]T    [    D_u.T @ W @ Maero]
                [δtheta]  @  [D_theta.T @ W @ Taero]

        δWint - δWext = 0
        δx.T @ (K_struct @ x - Qaero(x)) = 0
        where x = [u; theta], δx.T @ K_struct @ x = δWint, δx.T @ Qaero(x) = δWext
        This leads to:
        R(x) = K_struct @ x - Qaero(x) = 0
        The tangent stiffness is then given by

        K_T = ∂R/∂x = K_struct - ∂Qaero/∂x
        --> This tells us how the residual changes for a small change in Δx
        if it is positive definite, any small perturbation will lead and increase in residual,
        and in the second order to an increase in potential energy, meaning the system is stable.

        In matrix form: 
        K_T = [D_u.T @ W * EI @ D_u            0               ] - [∂Q1/∂u    ∂Q1/∂theta]
              [               0    D_theta.T @ W * GJ @ D_theta]   [∂Q2/∂u    ∂Q2/∂theta]
        
        We cannot use ddu and dtheta directly from optimizer.derivative_of() because:

        1. These are created as independent design variables, not symbolic derivatives
        2. The optimizer links them to u/theta via IMPLICIT trapezoidal integration constraints
        3. In the symbolic graph: ∂(ddu)/∂u = 0 (no direct dependency)
        
        The optimizer enforces: u[i] = u[i-1] + trapz_integral(du), but this constraint
        is satisfied during the solve, not expressed as an explicit derivative relationship.

        To compute the Hessian post-solve, we must reconstruct the derivatives by inverting
        the same trapezoidal integration scheme the optimizer used, giving us explicit
        symbolic expressions where ∂(ddu_reconstructed)/∂u ≠ 0.
        """
        
        # retrieve solved values
        du_sol = sol.value(du)
        ddu_sol = sol.value(ddu)
        dtheta_sol = sol.value(dtheta)

        n = len(du_sol)
        dy = np.diff(ys)
        
        """
        The trapezoidal integration scheme used by the optimizer is:
        
        u[i] = u[i-1] + (du[i-1] + du[i])/2 * dy[i-1]
        
        we setup a linear system A·du = b to invert this relationship and solve for du given u.
        
        A is an n x n matrix, b is an n-vector, A follows the trapezoidal coefficients.
        
        [ 1     0     0    ...  0  ]   [du[0]  ]   [    0        ] # boundary condition u[0] = 0
        [dy₀/2 dy₀/2  0    ...  0  ]   [du[1]  ]   [u[1] - u[0]  ]
        [ 0    dy₁/2 dy₁/2 ...  0  ] * [du[2]  ] = [u[2] - u[1]  ]
        [ ...                      ]   [ ...   ]   [ ...         ]
        [ 0     0    ...   ... ... ]   [du[n-1]]   [u[n-1]-u[n-2]]
        
        """
        # du reconstruction
        A_du = cas.DM.zeros((n, n))
        A_du[0, 0] = 1
        for i in range(1, n):
            A_du[i, i-1] = dy[i-1] / 2
            A_du[i, i] = dy[i-1] / 2
        b_du = cas.vertcat(0, *[u[i] - u[i-1] for i in range(1, n)])    
            
        du_recon = cas.solve(A_du, b_du)
        # this is now explicitly dependent on u and therefore differentiable
        
        """
        Following the same approach but with different boundary conditions for ddu:
        
        [dy/2  dy/2  0    ...  0   ]   [ddu[0]  ]   [du[1] - du[0]  ]
        [ 0    dy/2  dy/2 ...  0   ]   [ddu[1]  ]   [du[2] - du[1]  ]
        [ ...                      ] * [ddu[2]  ] = [ ...           ]
        [ 0     0    ... dy/2  dy/2]   [ddu[n-2]]   [du[n-1]-du[n-2]]
        [ 0     0    ...  0     1  ]   [ddu[n-1]]   [      0        ] # boundary condition ddu[n-1] = 0
        
        """
        # ddu reconstruction (already working)
        A_ddu = cas.DM.zeros((n, n))
        A_ddu[-1, -1] = 1
        for i in range(n-1):
            A_ddu[i, i] = dy[i] / 2
            A_ddu[i, i+1] = dy[i] / 2
        b_ddu = cas.vertcat(*[du_recon[i+1] - du_recon[i] for i in range(n-1)], 0)
        
        ddu_recon = cas.solve(A_ddu, b_ddu)
        # this is now explicitly dependent on u and therefore differentiable

        heave_moment_equation = EI *  ddu_recon - section_bending_moments

        
        # Verify reconstruction accuracy
        # print("=== Heave Derivatives (Verified) ===")
        # print(f"du error: {np.linalg.norm(sol.value(du_recon) - du_sol):.2e}")
        # print(f"ddu error: {np.linalg.norm(sol.value(ddu_recon) - ddu_sol):.2e}")
        # print(f"Residual heave error: {np.linalg.norm(sol.value(heave_moment_equation)):.2e}")

        """
        We can follow the same approach for dtheta reconstruction. Here the boundary conditions
        are not explicitly stated in the problem setup, but we defined:
        
        GJ * dtheta = section_torques and we know for free ended torsion problems that:
        dtheta/dy = 0 at free end (no twist rate at free end) --> dtheta[-1] = 0
        note that this information could aslo be used from the equality as section_torques[-1] = 0
        
        So we can setup the system as:
        [dy/2  dy/2  0    ...  0   ]   [dtheta[0]  ]   [theta[1] - theta[0]  ]
        [ 0    dy/2  dy/2 ...  0   ]   [dtheta[1]  ]   [theta[2] - theta[1]  ]
        [ ...                      ] * [dtheta[2]  ] = [ ...                 ]
        [ 0     0    ... dy/2  dy/2]   [dtheta[n-2]]   [theta[n-1]-theta[n-2]]
        [ 0     0    ...  0     1  ]   [dtheta[n-1]]   [      0              ] # boundary condition dtheta[n-1] = 0
        """
        
        A_dtheta = cas.DM.zeros((n, n))
        A_dtheta[-1, -1] = 1
        for i in range(n-1):
            A_dtheta[i, i] = dy[i] / 2
            A_dtheta[i, i+1] = dy[i] / 2
        b_dtheta = cas.vertcat(*[theta[i+1] - theta[i] for i in range(n-1)], 0)
            
        dtheta_recon = cas.solve(A_dtheta, b_dtheta)
        
        torsion_moment_equation = GJ * dtheta_recon - section_torques
        
        # print("=== Torsion Derivatives (Verified) ===")
        # print(f"dtheta error: {np.linalg.norm(sol.value(dtheta_recon) - dtheta_sol):.2e}")
        # print(f"Residual torsion error: {np.linalg.norm(sol.value(torsion_moment_equation)):.2e}")
        """
        Finally we can compute the Potential Energy Hessian:
        Π = Uint - Wext
        Uint = 1/2∫EI*ddu^2 dy + 1/2∫GJ*dtheta^2 dy
        Wext = ∫external_moments*ddu dy + ∫section_torques*dtheta dy

        For the integration we use the trapezoidal rule again.
        
        """

        # Structural part (positive definite, block diagonal)
        # Calculate integration weights for trapezoidal rule
        dy_weights = np.zeros(n)
        dy_weights[0] = dy[0] / 2
        dy_weights[1:-1] = (dy[:-1] + dy[1:]) / 2
        dy_weights[-1] = dy[-1] / 2

        W = cas.diag(dy_weights)

        D_u = cas.jacobian(ddu_recon, u)
        D_theta = cas.jacobian(dtheta_recon, theta)

        K_struct_uu = D_u.T @ cas.diag(EI * dy_weights) @ D_u
        K_struct_tt = D_theta.T @ cas.diag(GJ * dy_weights) @ D_theta

        # Aero part
        K_aero_uu = cas.jacobian(D_u.T @ W @ section_bending_moments, u)
        K_aero_ut = cas.jacobian(D_u.T @ W @ section_bending_moments, theta)
        K_aero_tu = cas.jacobian(D_theta.T @ W @ section_torques, u)
        K_aero_tt = cas.jacobian(D_theta.T @ W @ section_torques, theta)

        # Assemble full tangent stiffness
        K_struct = cas.vertcat(
            cas.horzcat(K_struct_uu, cas.DM.zeros((n,n))),
            cas.horzcat(cas.DM.zeros((n,n)), K_struct_tt)
        )

        K_aero = cas.vertcat(
            cas.horzcat(K_aero_uu, K_aero_ut),
            cas.horzcat(K_aero_tu, K_aero_tt)
        )

        # Total tangent stiffness
        K_T = K_struct - K_aero

        K_T_numeric = np.array(sol.value(K_T))

        # Reduce stiffness matrix by removing fixed constrained DOFs
        # Fixed DOFs: u[0] (root heave = 0) and theta[0] (root twist = 0)
        # These correspond to indices 0 and n in the 2*n state vector
        
        # Create mask for free indices (exclude 0 and n)
        keep_indices = [i for i in range(2 * n) if i != 0 and i != n]
        
        # Extract the submatrix for free degrees of freedom
        K_T_reduced = K_T_numeric[np.ix_(keep_indices, keep_indices)]

        # Stability: all eigenvalues of the reduced matrix must be positive
        eigenvalues = np.linalg.eigvals(K_T_reduced)

        max_eigenvalue = np.max(np.real(eigenvalues))
        min_eigenvalue = np.min(np.real(eigenvalues))
        
        # print("=== Eigenvalue Analysis ===")
        # print(f"Eigenvalue with BC: {eigenvalues}")
        # print(f"Positive: {np.sum(np.real(eigenvalues) > 1e-6)}")
        # print(f"Negative: {np.sum(np.real(eigenvalues) < -1e-6)}")
        # print(f"Near-zero: {np.sum(np.abs(np.real(eigenvalues)) <= 1e-6)}")

        # Check stability
        is_stable = min_eigenvalue > 1e-6

        self.performance["max_jacobian_eigenvalue"] = float(max_eigenvalue)
        self.performance["min_jacobian_eigenvalue"] = float(min_eigenvalue)
        self.performance["is_statically_stable"] = bool(is_stable)

        max_von_mises_stress, max_stress_point = self._compute_max_stress(bending_moments, EI, torques, GJ)

        # get the max abs values with sign
        max_bending_moment = bending_moments[np.argmax(np.abs(bending_moments))]
        max_torsion = torques[np.argmax(np.abs(torques))]

        self._max_stress_point = max_stress_point

        self.performance["max_bending_moment"] = float(max_bending_moment)
        self.performance["max_torsion"] = float(max_torsion)
        self.performance["max_von_mises_stress"] = float(max_von_mises_stress)

        # set the wing as the AeroSandbox wing
        self.asb_wing = self.set_asb_wing()

        self._calculate_performance()
        self.solved = True

    def _calculate_performance(self) -> None:
        """Calculates and stores performance metrics of the aeroelastic wing.
        """
        assert isinstance(self.asb_wing, asb.Wing), \
            "AeroSandbox wing must be set before calculating performance."

        assert self.mechanical_wing.solved, \
            "Mechanical wing properties must be solved before calculating performance."

        simulation = asb.VortexLatticeMethod(
            airplane=asb.Airplane(
                name="Aerostructures Test",
                xyz_ref=[0, 0, 0],
                wings=[self.asb_wing],
            ),
            op_point=asb.OperatingPoint(
                velocity=self.cfg.operating_point.velocity,
                alpha=0,
            ),
            chordwise_resolution=self.cfg.operating_point.chordwise_resolution,
            spanwise_resolution=self.cfg.operating_point.spanwise_resolution,
        )
        aero = simulation.run_with_stability_derivatives(alpha=True, beta=False, p=False, q=False, r=False)

        L = aero["L"]
        D = aero["D"]
        CL = aero["CL"]
        CD = aero["CD"]
        CM = aero["Cm"] #lowercase m for moment coefficient
        CMa = aero["Cma"] #pitching moment stability derivative
        CLa = aero["Cla"] #lift curve slope

        max_heave_displacement_idx = np.argmax(np.abs(self.heave_displacements))
        max_twist_displacement_idx = np.argmax(np.abs(self.twist_displacements))

        self.performance.update({
            "L": float(L),
            "D": float(D),
            "CL": float(CL),
            "CD": float(CD),
            "CM": float(CM),
            "CMa": float(CMa),
            "CLa": float(CLa),
            "max_heave_displacement": float(self.heave_displacements[max_heave_displacement_idx]),
            "max_twist_displacement": float(self.twist_displacements[max_twist_displacement_idx]),
            "mass": float(self.mechanical_wing.mass),
        })

    def reset(self) -> None:
        """Resets the aeroelastic wing to its undeformed state."""
        self.heave_displacements = np.zeros_like(self.ys_over_half_span)
        self.twist_displacements = np.zeros_like(self.ys_over_half_span)
        self.performance = {}
        self._max_stress_point = None
        self.asb_wing = None
        self.solved = False

        self.mechanical_wing.reset()

    def draw_vlm(self,
                 show_wing: bool = True,
                 show_vortex_centers: bool = False,
                 show_shear_centers: bool = False,
                 show_forces: bool = True,
                 show_lever_arms: Optional[int] = None,
                 reduce_res: bool = True,
                 show_centroids: bool = False,
                 show_max_stress_point: bool = False) -> None:
        """Draws the Vortex Lattice Method mesh of the AeroSandbox wing.
        Args:
            show_wing: Whether to show the wing mesh.
            show_vortex_centers: Whether to show the vortex centers.
            show_schear_centers: Whether to show the shear centers.
            show_forces: Whether to show the aerodynamic forces.
            show_lever_arms: Index of the wing cross-section to draw lever arms to.
                        Recomended to set show_wing to False if this is set.
            reduce_res: Reduces the span and chordwise resolution to 2
                        for more comprehensive plots if set to True.
            show_centroids: Whether to show the centroids.
            show_max_stress_point: Whether to show the maximum stress point.
        """
        assert isinstance(self.asb_wing, asb.Wing),\
            "AeroSandbox wing has not been set yet."

        if show_max_stress_point:
            assert isinstance(self._max_stress_point,np.ndarray),\
                "Maximum stress point has not been computed yet."

        if show_lever_arms is not None:
            assert isinstance(show_lever_arms, int)\
                and show_lever_arms in range(len(self.ys_over_half_span)),\
                "show_lever_arms must be an integer index of the wing cross-section\
                the lever arms are drawn to."
            section_idx = show_lever_arms
        else:
            section_idx = -1  #default to last section if not set

        #set reduced resolution for more comprehensible plots
        ch_res = 2 if reduce_res else self.cfg.operating_point.chordwise_resolution
        sp_res = 2 if reduce_res else self.cfg.operating_point.spanwise_resolution

        simulation = asb.VortexLatticeMethod(
            airplane=asb.Airplane(
                name="Aerostructures Test",
                xyz_ref=[0, 0, 0],
                wings=[self.asb_wing],
            ),
            op_point=asb.OperatingPoint(
                velocity=self.cfg.operating_point.velocity,
                alpha=0,
            ),
            chordwise_resolution=ch_res,
            spanwise_resolution=sp_res,
        )
        simulation.run()

        # Get necessary data for plotting
        ys = self.ys_over_half_span * (self.span / 2)
        chords = self.chord_function.evaluate(self.ys_over_half_span)
        num_intervals = (len(ys) - 1)
        num_panels_per_interval = simulation.spanwise_resolution * simulation.chordwise_resolution
        num_panels = num_intervals * num_panels_per_interval
        # Calculate leading edge offsets caused by spar sweep
        slope = np.tan(np.radians(self.spar_sweep))
        # calculate offsets, scale to true lengths and subtract spar position offset
        offsets = slope * self.ys_over_half_span * (self.span / 2) - chords * self.cfg.fixed.spar_position

        vortex_centers = simulation.vortex_centers[:num_panels, :]
        forces = simulation.forces_geometry[:num_panels, :]
        # Shear centers without considering heave displacements and sweep offsets
        sc = self.mechanical_wing.shear_center.evaluate(self.ys_over_half_span)
        sc_x = sc[:, 0]
        sc_z = sc[:, 1]

        # the offset shift is neglecting the rotation due to small angles
        shear_centers = np.vstack((sc_x + offsets,
                                   ys,
                                   sc_z + self.heave_displacements)).T
        cent = self.mechanical_wing.centroid.evaluate(self.ys_over_half_span)

        cent_x = cent[:, 0]
        cent_z = cent[:, 1]
        centroids = np.vstack((cent_x + offsets,
                               ys,
                               cent_z + self.heave_displacements)).T

        r_vectors = []
        start_idx = section_idx * num_panels_per_interval
        # draw lever arms from shear center to vortex centers
        for j in range(start_idx, num_panels):
            # Position vector from shear center to vortex center
            r_vectors.append((shear_centers[section_idx, :], vortex_centers[j, :]))


        if show_wing:
            plotter = simulation.draw(show=False, colorbar_label="lift strength [N]", cmap="viridis")
            plotter.remove_scalar_bar()
        else:
            plotter = pv.Plotter()  # type: ignore
            plotter.title = "Forces and Moment Arms"
            #this throws a mypy error for some reason???
            plotter.add_axes() # type: ignore[call-arg]
            plotter.show_grid(color="gray") # type: ignore[call-arg]

        # Add vortex center points colored by force magnitude
        if show_vortex_centers:
            vortex_centers_cloud = pv.PolyData(vortex_centers)
            plotter.add_mesh(
                vortex_centers_cloud,
                point_size=5,
                render_points_as_spheres=True,
                label='Vortex Centers',
                color="red"
            )
        if show_forces:
            # Add force vectors with consistent coloring
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes)
            min_force = np.min(force_magnitudes)

            force_scale = 0.001
            arrows = pv.PolyData(vortex_centers)
            arrows["force_vectors"] = forces
            arrows["lift strength [N]"] = force_magnitudes

            arrow_glyphs = arrows.glyph(
                orient="force_vectors",
                scale="force_vectors",
                factor=force_scale,
                geom=pv.Arrow()
            )

            plotter.add_mesh(
                arrow_glyphs,
                scalars="lift strength [N]",
                cmap='viridis',
                label=f"Force Vectors (scale {force_scale})",
                clim=[min_force, max_force]
            )
        if show_shear_centers:
            # Add shear centers
            shear_centers_cloud = pv.PolyData(shear_centers)
            plotter.add_mesh(
                shear_centers_cloud,
                color='blue',
                point_size=8,
                render_points_as_spheres=True,
                label='Shear Centers'
            )
        if show_centroids:
            # Add centroids
            centroids_cloud = pv.PolyData(centroids)
            plotter.add_mesh(
                centroids_cloud,
                color='green',
                point_size=8,
                render_points_as_spheres=True,
                label='Centroids'
            )
        if show_max_stress_point:
            assert isinstance(self._max_stress_point, np.ndarray),\
                "Maximum stress point has not been computed yet."

            max_stress_point = [self._max_stress_point[0],
                                self._max_stress_point[1] * (self.span / 2),
                                self._max_stress_point[2]]
             # Add max stress point
            max_stress_point_cloud = pv.PolyData(max_stress_point)
            plotter.add_mesh(
                max_stress_point_cloud,
                color='yellow',
                point_size=8,
                render_points_as_spheres=True,
                label='Max Stress Point'
            )
        if show_lever_arms is not None:
            # Add moment arm lines for first section
            for i in range(len(r_vectors)):
                line = pv.Line(r_vectors[i][0], r_vectors[i][1])
                plotter.add_mesh(
                    line,
                    color='orange',
                    opacity=1,
                    line_width=2.5
                )

        # Add legend and show
        plotter.add_legend(
            size=(0.2, 0.2),
            loc='upper right',
            face='rectangle'
        )
        plotter.show()
               
    def plot_asb_wing(self) -> None:
        """Plots the AeroSandbox wing if it has been setted.
        """
        assert isinstance(self.asb_wing, asb.Wing),\
            "AeroSandbox wing has not been set yet."

        self.asb_wing.draw_three_view(style="wireframe", show=True)

    def plot_comparison(self) -> None:
        """Plots a comparison between the set asb_wing and the undeformed wing.
        """
        assert self.solved,\
            "Aeroelastic problem must be solved before plotting comparison."
        assert isinstance(self.asb_wing, asb.Wing),\
            "AeroSandbox wing has not been set yet."

        ys = self.ys_over_half_span * (self.span / 2)
        u = self.heave_displacements
        twist_displacements = self.twist_displacements

        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(ys, u, label="Heave")
        ax[0].set_ylabel("Heave Displacement [m]")
        ax[1].plot(ys, twist_displacements, label="Twist")
        ax[1].set_ylabel("Twist Displacement [deg]")
        p.show_plot(
            title="Wing Displacement (Heave + Twist) at Static Aeroelastic Equilibrium",
            xlabel="Spanwise Position [m]",
            legend=False,
        )

        zero_heave_displacements = np.zeros_like(self.ys_over_half_span)
        zero_twist_displacements = np.zeros_like(self.ys_over_half_span)
        undeformed_wing: asb.Wing = self._to_asb_wing(zero_heave_displacements,
                                                  zero_twist_displacements)

        axs = undeformed_wing.draw_three_view(show=False)
        self.asb_wing.draw_three_view(axs=axs, style="wireframe", show=True)
        
    def plot_half_span(self, 
                       plot_tip_lines: bool = True, 
                       **draw_kwargs) -> None:
        """Plots the scaled half-span of the AeroSandbox wing if it has been setted.
        Args:
            plot_tip_lines: Whether to plot guide lines from root to tip.
        """

        heave_displacements = self.heave_displacements    
        twist_displacements = self.twist_displacements

        chords = self.chord_function.evaluate(self.ys_over_half_span)
        twists = self.twist_function.evaluate(self.ys_over_half_span)
        shear_centers = self.mechanical_wing.shear_center.evaluate(self.ys_over_half_span)
        # Calculate leading edge offsets caused by spar sweep
        slope = np.tan(np.radians(self.spar_sweep))
        # calculate offsets, scale to true lengths and subtract spar position offset
        offsets = slope * self.ys_over_half_span * (self.span / 2) - chords * self.cfg.fixed.spar_position

        xsecs = []

        for i, pos in enumerate(self.ys_over_half_span):
            #first shift to shear center
            xyz_le = np.array([
                -shear_centers[i][0],
                pos * (self.span / 2),
                -shear_centers[i][1]
            ])
            #rotate around y-axis for twist
            # note the offsets are calculated for a straight spar but applied after rotateing
            # the offsets iself are not rotated, this only holds for small twist angles  
            xyz_le = np.rotation_matrix_3D(
                angle=np.radians(twists[i] + twist_displacements[i]),
                axis="y"
            ) @ xyz_le
            #shift back from shear center and apply heave and offset
            xyz_le += np.array([
                offsets[i] + shear_centers[i][0],
                0,
                heave_displacements[i] + shear_centers[i][1]
            ])

            xsecs.append(
                asb.WingXSec(
                    xyz_le=xyz_le,
                    chord=chords[i],
                    twist=twists[i] + twist_displacements[i],
                    airfoil=self.airfoils[i].to_asb_airfoil(),
                )
            )

        asb_wing = asb.Wing(
            symmetric=False,
            xsecs=xsecs,
            color="C0", # light gray
        )
        
        ax = asb_wing.draw_wireframe(show=False, **draw_kwargs)

        xz_width = 3*max(chords)
        ax.set_xlim([- xz_width/2, xz_width/2])
        ax.set_ylim([0, self.span/2])
        ax.set_zlim([- xz_width/2, xz_width/2])

        if plot_tip_lines:
            x_min = ax.get_xlim()[0]
            length = xsecs[0].xyz_le[0] - x_min
            x_ref = xsecs[0].xyz_le[0] - length
            z_const = xsecs[0].xyz_le[2]
            y_tip = xsecs[-1].xyz_le[1]

            ax.plot([x_ref, x_ref], [0, y_tip], [z_const, z_const],
                    color='black', linestyle=':', linewidth=1, alpha=0.8)

            prev_y = xsecs[0].xyz_le[1]
            prev_z = z_const
            prev_z_twist = z_const

            for i, xsec in enumerate(xsecs):
                le = xsec.xyz_le
                x_start, y, z = le
                # Dotted line in negative x
                x_end = x_start - length
                ax.plot([x_start, x_ref], [y, y], [z, z], color="blue", linestyle=":", alpha=0.7, linewidth=1)

                # Line in twist direction
                total_twist_deg = twists[i] + twist_displacements[i]
                # negative because the twist is defined as positive nose-up
                total_twist_rad = -np.radians(total_twist_deg)
                slope = np.tan(total_twist_rad)
                
                dz = (x_ref - x_start) * slope
                x_twist = x_ref
                z_twist = z + dz
                ax.plot(
                    [x_start, x_twist],
                    [y, y],
                    [z, z_twist],
                    color='red', linestyle=':', linewidth=1, alpha=0.8
                )
                if i > 0:
                    ax.plot(
                        [x_ref, x_ref],
                        [prev_y, y],
                        [prev_z, z],
                        color='blue', linestyle='-', linewidth=1, alpha=0.7
                    )
                    ax.plot(
                        [x_ref, x_ref],
                        [prev_y, y],
                        [prev_z_twist, z_twist],
                        color='red', linestyle='-', linewidth=1, alpha=0.8
                    )

                prev_y = y
                prev_z = z
                prev_z_twist = z_twist

        ax.view_init(elev=20, azim=35) 
        plt.savefig("half_span_wing.svg", dpi=600)
        plt.show()
        
    def plot_mechanics_3D(self,
                          include_sweep: bool = True,
                          include_twist: bool = False,
                          include_deformations: bool = False,
                          scale: bool = False,
                          pretty_plot: bool = False,
                          save_path: str | None = None) -> None:
        """Plots the 3D mechanical properties of the wing.
        Args:
            include_sweep: Whether to include the spar sweep in the plot.
            include_twist: Whether to include the base twist in the plot.
            include_deformations: Whether to include the heave and twist deformations in the plot.
            scale: Whether to scale the mechanical properties for better visibility.
                the total twist is scaled with a factor of 10
                the heave displacements are scaled to reference.
            pretty_plot: Whether to use a predefined set of parameters for a more visually appealing plot.
            save_path: If provided, the path to save the plot as an image file. (as svg)
        """
        
        self.mechanical_wing.plot_3D(
            spar_sweep=self.spar_sweep if include_sweep else 0.0,
            include_twist=include_twist,
            twist_displacements=self.twist_displacements if include_deformations else None,
            heave_displacements=self.heave_displacements if include_deformations else None,
            scale=scale,
            pretty_plot=pretty_plot,
            save_path=save_path
        )

    def to_dict(self) -> dict:
        """Converts the AeroelasticWing object to a dictionary representation.
        Returns:
            A dictionary containing the wing parameters and displacements.
        """
        sampled_params = {
                **self.chord_function.to_dict(),
                **self.twist_function.to_dict(),
                "spar_sweep": self.spar_sweep
            }

        airfoils = {}
        for i, af in enumerate(self.airfoils):
            airfoils[f"airfoil_{i}"] = af.to_dict()

        _max_stress_point = self._max_stress_point.tolist()\
            if self._max_stress_point is not None else None

        aeroelastic_wing ={
            "id": self.id,
            "chord_function": self.chord_function.to_dict(),
            "twist_function": self.twist_function.to_dict(),
            "spar_sweep": self.spar_sweep,
            "heave_displacements": self.heave_displacements.tolist(),
            "twist_displacements": self.twist_displacements.tolist(),
            "performance": self.performance,
            "_max_stress_point": _max_stress_point,
            "airfoils": airfoils,
            "mechanical_wing": self.mechanical_wing.to_dict()
        }

        return {
            **sampled_params,
            **self.performance,
            "aeroelastic_wing": json.dumps(aeroelastic_wing)
        }

    @staticmethod
    def from_dict(cfg: DictConfig | AppConfig, data: dict) -> "AeroelasticWing":
        """Creates an AeroelasticWing object from a dictionary representation.
        Args:
            cfg: Application configuration object.
            data: A dictionary containing the wing parameters and displacements.
        Returns:
            An AeroelasticWing object.
        """
        # Create an uninitialized AeroelasticWing instance
        aero_wing = AeroelasticWing.__new__(AeroelasticWing)
        aero_wing.cfg = AppConfig(cfg) if isinstance(cfg, DictConfig) else cfg
        aero_wing.id = data["id"]

        aero_wing.chord_function = ChordFunction.from_dict(data["chord_function"])
        aero_wing.twist_function = TwistFunction.from_dict(data["twist_function"])
        aero_wing.spar_sweep = data["spar_sweep"]

        airfoils = []
        for i in range(len(data["airfoils"])):
            af_data = data["airfoils"][f"airfoil_{i}"]
            airfoil = Airfoil.from_dict(af_data)
            airfoils.append(airfoil)
        aero_wing.airfoils = np.array(airfoils)

        aero_wing.ys_over_half_span = aero_wing.cfg.fixed.ys_over_half_span
        aero_wing.span = aero_wing.cfg.fixed.span

        aero_wing.mechanical_wing = MechanicalWing.from_dict(aero_wing.cfg, data["mechanical_wing"])

        aero_wing.heave_displacements = np.array(data["heave_displacements"])
        aero_wing.twist_displacements = np.array(data["twist_displacements"])
        aero_wing.performance = data["performance"]

        aero_wing.asb_wing = aero_wing._to_asb_wing(
            heave_displacements=aero_wing.heave_displacements,
            twist_displacements=aero_wing.twist_displacements
        )

        aero_wing._max_stress_point = np.array(data["_max_stress_point"])\
            if data["_max_stress_point"] is not None else None

        aero_wing.solved = True

        return aero_wing

    @staticmethod
    def from_pd(cfg: DictConfig | AppConfig, row: pd.Series) -> "AeroelasticWing":
        """Creates an AeroelasticWing object from a CSV row (pandas Series).
        Args:
            cfg: Application configuration object.
            row: A pandas Series representing a row from the CSV file.
        Returns:
            An AeroelasticWing object.
        """
        aero_wing_str = row["aeroelastic_wing"]
        aero_wing_dict = json.loads(aero_wing_str)
        return AeroelasticWing.from_dict(cfg, aero_wing_dict)




@hydra.main(version_base=None, config_path="../../../configs/", config_name="dataset_config")
def main(cfg: DictConfig):
    from time import time
    config = AppConfig(cfg)
    airfoilrepo = AirfoilRepository(config.run.airfoils_csv_path)
    #airfoils = airfoilrepo.sample_airfoils(np.array([5,5,1,1,1], dtype=int)) # 730
    airfoils = airfoilrepo.get_airfoils_by_ids(np.ones(5, dtype=int) * 0)
    #assert isinstance(wing.EI, Function)
    wing = AeroelasticWing(config, {
        "chord_ctrl_pos": 0.75,
        "chord_weight_ctrl": 1.0,
        "chord_tip": 0.3,
        "twist_root": 5.0,
        "twist_tip": 3.0,
        "twist_mean_pos": 0.5,
        "spar_sweep": 0.0
    }, airfoils)
    start_time = time()
    wing.mechanical_wing.solve_for_properties()
    end_time = time()
    wing.mechanical_wing.plot_at_rib_positions()
    wing.mechanical_wing.plot_mesh_at_rib_position(rib_index=2)
    print(f"Solved mechanical properties in {end_time - start_time:.2f} seconds")
    #wing.set_asb_wing()
    #wing.plot_asb_wing()
    start_time = time()
    solutions = []
    wing.solve_all_my_problems()
    """
    solutions.append((wing.heave_displacements.copy(), wing.twist_displacements.copy()))
    print("Heave:", solutions[-1][0])
    print("Twist:", solutions[-1][1])
    for i in range(10):
        try:
            pertubated_heave = wing.heave_displacements + np.random.normal(0, 0.05, size=wing.heave_displacements.shape)
            pertubated_twist = wing.twist_displacements + np.random.normal(0, 0.05, size=wing.twist_displacements.shape)
            wing.solve_all_my_problems(heave_init=pertubated_heave, twist_init=pertubated_twist)
            solutions.append((wing.heave_displacements.copy(), wing.twist_displacements.copy()))
            print(f"Run {i+1} completed")
            print("Heave:", solutions[-1][0])
            print("Twist:", solutions[-1][1])
        except Exception as e:
            pass
    """
    end_time = time()
    print(f"Solved aeroelastic problem in {end_time - start_time:.2f} seconds")
    #wing.set_asb_wing()
    #wing.plot_asb_wing()
    wing.plot_comparison()
    #wing.plot_at_rib_pos()
    #wing.EI.plot("log")
    #wing.EI.area_preserving_boxcar(box_centers=wing.rib_positions).plot("log")
    print("Performance:", wing.performance)
    #wing.draw_vlm(show_vortex_centers=True, show_shear_centers=True, reduce_res=False, show_centroids=True, show_max_stress_point=False)
    #wing.draw_vlm(show_wing=False, show_vortex_centers=True, show_shear_centers=True, reduce_res=False, show_centroids=True, show_max_stress_point=False, show_lever_arms=0)
    #print(wing.mechanical_wing.EI.evaluate(wing.ys_over_half_span))

    #test to see if it preserves the integral --> we can smooth and still use it for mass calculation
    #print(wing.mechanical_wing.area.integrate().evaluate(1.0))
    #print(wing.mechanical_wing.area.area_preserving_boxcar(box_centers=wing.mechanical_wing.rib_positions).integrate().evaluate(1.0))
    #print(wing.mechanical_wing.area.integrate_between(0.0, 1.0))
    #print(wing.mechanical_wing.area.area_preserving_boxcar(box_centers=wing.mechanical_wing.rib_positions).integrate_between(0.0, 1.0))
    # wing.mechanical_wing.plot_mesh_at_rib_position(rib_index=2)
    # wing.plot_half_span()
    wing.plot_mechanics_3D(include_sweep=True, include_twist=True, include_deformations=False, scale=False, pretty_plot=True, save_path="mechanics_3D_plot.svg")

if __name__ == "__main__":
    main()
