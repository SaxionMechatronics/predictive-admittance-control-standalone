"""
Standalone Example of using Acados to implement Predictive Admittance Control
to control Fully Actuated Hexarotor (Fiberthex) simulated in MuJoCo

The main reference are:
    [1] A. Alharbat, et al., "Predictive Admittance Control for Aerial Physical
        Interaction," in IEEE RA-L 2025
    [2] A. Alharbat, et al., "Three Fundamental Paradigms for Aerial Physical
        Interaction Using Nonlinear Model Predictive Control," in ICUAS 2022

================
LOG FILE FORMAT:
================
This script generates .pkl files containing comprehensive simulation data with the following structure:

To load and inspect a log file:
    >>> import pickle
    >>> with open('logs/your_file.pkl', 'rb') as f:
    >>>     data = pickle.load(f)
    >>> print(data.keys())  # See all available fields
    >>> print(data['metadata'])  # View metadata and data format description

Main data fields:
    - metadata: File description, timestamp, version, and data format documentation
    - times: Time stamps (n,) [seconds]
    - references: Dict with position(n,3), quaternion(n,4), velocity(n,3),
                  angular_velocity(n,3)
    - actual_states: Dict with position(n,3), quaternion(n,4), velocity(n,3),
                     angular_velocity(n,3), rotor_forces(n,6),
                     impedance_position(n,3), impedance_velocity(n,3),
                     contact_forces(n,3)
    - actual_controls: Control inputs (n,6) [N/s]
    - predicted_states: List of MPC predicted trajectories, each (N+1, nx)
    - predicted_controls: List of MPC predicted controls, each (N, nu)
    - solve_times: MPC solver times (n,) [seconds]
    - robot_config: Complete robot parameters (mass, inertia, geometry, etc.)
    - controller_config: Complete controller parameters (weights, horizon,
                         impedance, etc.)
    - simulation_config: Simulation parameters (dt, duration, etc.)

For a helper function to inspect log files, see inspect_log_file() at the end
of this script.
"""

import os
import pickle
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

import casadi as ca
import mujoco
import mujoco.viewer
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from scipy import signal
from scipy.spatial.transform import Rotation as R

from models.mujoco.fiberthex_wall import FiberthexWallScene

# Import logging and visualization
from utils.logger import ExperimentLogger
from utils.utils import alloc, dist_quat, hamilton_prod, quat2rot_casadi, quat2rot_numpy
from utils.visualization import ResultsVisualizer

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================


@dataclass
class HexarotorConfig:
    """Configuration parameters for hexarotor drone"""

    name: str = "hexarotor"
    mass: float = 2.0  # kg
    arm_length: float = 0.39  # m
    alpha_tilt: float = 20.0  # deg
    beta_tilt: float = 0.0  # deg

    Jxx: float = 0.06185585085
    Jyy: float = 0.46467853407
    Jzz: float = 0.47162119244

    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    com_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    # End-effector offset in body frame [x, y, z] (meters)
    ee_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.60188765563, -0.3475, 0.0])
    )

    num_rotors: int = 6
    c_f: float = 9.9e-4
    c_t: float = 1.9e-5
    w_min: float = 16.0
    w_max: float = 110.0

    nx: int = (
        28  # States: [pos(3), quat(4), vel(3), angvel(3), rotor forces(6), imp_pos(3), imp_vel(3), contact_forces(3)]
    )
    nu: int = 6  # Controls: [dF1...dF6]
    ny: int = 19  # Outputs

    def get_inertia_matrix(self) -> np.ndarray:
        return np.diag([self.Jxx, self.Jyy, self.Jzz])

    def get_force_limits(self) -> tuple:
        f_min = self.c_f * self.w_min**2
        f_max = self.c_f * self.w_max**2
        return f_min, f_max

    def get_hover_force(self) -> float:
        total_hover_force = self.mass * np.linalg.norm(self.gravity)
        return total_hover_force / self.num_rotors / np.cos(np.deg2rad(self.alpha_tilt))


@dataclass
class ControllerConfig:
    """Configuration for NMPC controller"""

    dt: float = 0.1
    horizon: int = 10

    # Impedance parameters
    M: np.ndarray = field(default_factory=lambda: np.array([1.5, 1.5, 1.5]))
    D: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 10.0]))
    K: np.ndarray = field(default_factory=lambda: np.array([6.0, 6.0, 6.0]))

    integrator_type: str = "ERK"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    nlp_solver: str = "SQP_RTI"
    hpipm_mode: str = "BALANCE"
    tolerance: float = 1e-6
    hessian_approx: str = "GAUSS_NEWTON"  # "EXACT" or "GAUSS_NEWTON"
    regularize_method: str = "CONVEXIFY"
    reg_epsilon: float = 1e-8
    warm_start: int = 1
    sim_method_num_stages: int = 4
    sim_method_num_steps: int = 1

    position_weight: np.ndarray = field(
        default_factory=lambda: np.array([1e1, 1e1, 1e1])
    )
    quaternion_weight: float = 1e4
    velocity_weight: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 10.0, 10.0])
    )
    angular_velocity_weight: np.ndarray = field(
        default_factory=lambda: np.array([10.0, 10.0, 10.0])
    )
    force_weight: np.ndarray = field(default_factory=lambda: np.zeros(6))
    control_weight: float = 0.1

    df_min: float = -20.0
    df_max: float = 25.0
    num_rotors: int = 6
    gen_build_code: bool = True
    look_ahead_reference: bool = (
        False  # If True, use different references for each horizon point
    )

    def get_weight_vector(self) -> np.ndarray:
        return np.concatenate([
            self.position_weight,
            np.array([self.quaternion_weight]),
            np.zeros(3),
            self.velocity_weight,
            self.angular_velocity_weight,
            self.force_weight,
        ])

    def get_control_weight_vector(self) -> np.ndarray:
        return np.ones(self.num_rotors) * self.control_weight

    def get_control_limits(self) -> tuple:
        lbu = np.ones(self.num_rotors) * self.df_min
        ubu = np.ones(self.num_rotors) * self.df_max
        return lbu, ubu


@dataclass
class SimulationConfig:
    """Configuration for simulation"""

    dt: float = 0.001
    dt_control: float = 0.01
    decimation: int = dt_control / dt
    num_steps: int = 10000
    log_directory: str = "logs"
    log_reference: bool = True
    render_video: bool = False
    video_filename: str = "simulation.mp4"
    video_fps: int = 30
    video_width: int = 1920
    video_height: int = 1080
    video_show_progress: bool = True  # Show progress bar when saving video
    save_video_frames_as_images: bool = False  # Save individual frames as PNG images


# ============================================================================
# HEXAROTOR DYNAMICS
# ============================================================================


class HexarotorDynamics:
    """Fully-actuated hexarotor dynamics model with Impedance Dynamics and Contact Forces Dynamics

    This class implements the full nonlinear dynamics of a tilted hexarotor with:
    - 6DOF rigid body dynamics (position, orientation, velocities)
    - Individual rotor thrust dynamics
    - Impedance dynamics for compliant interaction
    - Contact force dynamics for physical interaction

    State Vector (28 dimensions):
        [0:3]   - Position (x, y, z) [m]
        [3:7]   - Quaternion (w, x, y, z) (unit quaternion)
        [7:10]  - Linear velocity (vx, vy, vz) [m/s]
        [10:13] - Angular velocity (wx, wy, wz) [rad/s]
        [13:19] - Rotor thrust forces (F1, ..., F6) [N]
        [19:22] - Impedance position (px, py, pz) [m]
        [22:25] - Impedance velocity (vx, vy, vz) [m/s]
        [25:28] - Contact forces (fx, fy, fz) [N]

    Control inputs Vector (6 dimensions):
        [0:6] - Rotor thrust derivatives (dF1/dt, ..., dF6/dt) [N/s]
    """

    def __init__(self, config: HexarotorConfig):
        self.config = config
        self.states = None
        self.controls = None
        self.x_dot = None
        self.GF = None
        self.GT = None

        # Symbolic parameters for impedance
        self.M_sym = None
        self.D_sym = None
        self.K_sym = None

    def setup_symbolic_model(self) -> tuple[ca.SX, ca.SX, ca.SX]:
        """Setup hexarotor dynamics"""
        if self.states is not None:
            return self.states, self.controls, self.x_dot

        cfg = self.config

        states = ca.SX.sym("states", cfg.nx, 1)
        controls = ca.SX.sym("controls", cfg.nu, 1)

        q = states[3:7]
        q = q / ca.norm_2(q)
        vel = states[7:10]
        omega = states[10:13]
        rotor_thrusts = states[13:19]
        imp_pos = states[19:22]
        imp_vel = states[22:25]
        contact_forces = states[25:28]

        self.GF, self.GT = alloc(
            cfg.num_rotors,
            cfg.arm_length,
            cfg.alpha_tilt,
            cfg.beta_tilt,
            cfg.c_f,
            cfg.c_t,
            com=cfg.com_offset.tolist(),
        )

        # End-effector position in body frame
        r_ee = ca.SX(cfg.ee_offset)

        R = quat2rot_casadi(q)
        J = cfg.get_inertia_matrix()
        J_ca = ca.SX(J)
        Jinv = ca.inv(J_ca)

        B_p_com = ca.SX(cfg.com_offset)
        # Reference: Eq 2b from [1]
        omega_dot = Jinv @ (
            -ca.cross(omega, J_ca @ omega)
            + self.GT @ rotor_thrusts
            + ca.cross(r_ee, ca.transpose(R) @ contact_forces)
            + cfg.mass * ca.cross(B_p_com, ca.transpose(R) @ cfg.gravity)
        )

        omega_term = R @ (
            ca.cross(omega_dot, r_ee) + ca.cross(omega, ca.cross(omega, r_ee))
        )

        pos_dot = vel
        # Reference: Eq 2c from [1]
        quat_dot = hamilton_prod(q, ca.vertcat(0, omega)) / 2
        # Reference: Eq 2a from [1]
        vel_dot = (
            ((R @ self.GF @ rotor_thrusts) + contact_forces) / cfg.mass
            + cfg.gravity
            + omega_term
        )
        # Reference: Eq 3 from [1]
        rotor_thrusts_dot = controls

        # Impedance dynamics
        # Create symbolic parameters
        self.M_sym = ca.SX.sym("M", 3)
        self.D_sym = ca.SX.sym("D", 3)
        self.K_sym = ca.SX.sym("K", 3)
        M_mat = ca.diag(self.M_sym)
        D_mat = ca.diag(self.D_sym)
        K_mat = ca.diag(self.K_sym)

        imp_pos_dot = imp_vel
        # p_z_ddot = M^(-1) * (-D * p_z_dot - K * p_z + f_c_w)
        # Reference: Eq 5 from [1]
        imp_vel_dot = ca.inv(M_mat) @ (
            -D_mat @ imp_vel - K_mat @ imp_pos + contact_forces
        )
        # imp_vel_dot = ca.SX.zeros(3)

        contact_forces_dot = ca.SX.zeros(3)

        x_dot = ca.vertcat(
            pos_dot,
            quat_dot,
            vel_dot,
            omega_dot,
            rotor_thrusts_dot,
            imp_pos_dot,
            imp_vel_dot,
            contact_forces_dot,
        )

        self.states = states
        self.controls = controls
        self.x_dot = x_dot

        return states, controls, x_dot

    def get_initial_state(self, hover: bool = True) -> np.ndarray:
        """Get initial state vector"""
        x0 = np.zeros(self.config.nx)
        x0[3] = 1.0

        if hover:
            hover_force = self.config.get_hover_force()
            x0[13:19] = hover_force

        return x0


# ============================================================================
# NMPC CONTROLLER
# ============================================================================


class NMPCController:
    """Nonlinear Model Predictive Controller using Acados"""

    def __init__(
        self,
        dynamics: HexarotorDynamics,
        controller_config: ControllerConfig,
        robot_config: HexarotorConfig,
    ):
        self.dynamics = dynamics
        self.ctrl_cfg = controller_config
        self.robot_cfg = robot_config
        self.solver = None
        self.integrator = None
        self.counter = 1
        self.impedance_position = np.zeros(3)
        self.impedance_velocity = np.zeros(3)

        self._setup_ocp()

    def _setup_pac_objective_function(self, states, controls, xdot, refs, W, W_u):
        """Setup Predictive Admittance Control objective function"""

        # Reference: Eq (8) from [1]
        p = states[:3]
        q = states[3:7]
        pdot = states[7:10]
        omega = states[10:13]
        imp_p = states[19:22]
        imp_pdot = states[22:25]
        pddot = xdot[7:10]
        omega_dot = xdot[10:13]
        imp_pddot = xdot[22:25]

        stage_cost = (
            0.5
            * ca.transpose(p - imp_p - refs[:3])
            @ ca.diag(W[:3])
            @ (p - imp_p - refs[:3])
            + 0.5 * dist_quat(q, refs[3:7]) * W[3]
            + 0.5
            * ca.transpose(pdot - imp_pdot - refs[7:10])
            @ ca.diag(W[7:10])
            @ (pdot - imp_pdot - refs[7:10])
            + 0.5
            * ca.transpose(omega - refs[10:13])
            @ ca.diag(W[10:13])
            @ (omega - refs[10:13])
            + 0.5
            * ca.transpose(pddot - imp_pddot - refs[13:16])
            @ ca.diag(W[13:16])
            @ (pddot - imp_pddot - refs[13:16])
            + 0.5
            * ca.transpose(omega_dot - refs[16:19])
            @ ca.diag(W[16:19])
            @ (omega_dot - refs[16:19])
            + 0.5 * ca.transpose(controls) @ ca.diag(W_u) @ controls
        )

        terminal_cost = (
            0.5
            * ca.transpose(p - imp_p - refs[:3])
            @ ca.diag(W[:3])
            @ (p - imp_p - refs[:3])
            + 0.5 * dist_quat(q, refs[3:7]) * W[3]
            + 0.5
            * ca.transpose(pdot - imp_pdot - refs[7:10])
            @ ca.diag(W[7:10])
            @ (pdot - imp_pdot - refs[7:10])
            + 0.5
            * ca.transpose(omega - refs[10:13])
            @ ca.diag(W[10:13])
            @ (omega - refs[10:13])
            + 0.5
            * ca.transpose(pddot - imp_pddot - refs[13:16])
            @ ca.diag(W[13:16])
            @ (pddot - imp_pddot - refs[13:16])
            + 0.5
            * ca.transpose(omega_dot - refs[16:19])
            @ ca.diag(W[16:19])
            @ (omega_dot - refs[16:19])
        )

        return stage_cost, terminal_cost

    def _setup_traj_track_objective_function(
        self, states, controls, xdot, refs, W, W_u
    ):
        """Setup Trajectory Tracking Control objective function"""

        # Reference: Eq (8) from [1]
        p = states[:3]
        q = states[3:7]
        pdot = states[7:10]
        omega = states[10:13]
        pddot = xdot[7:10]
        omega_dot = xdot[10:13]

        stage_cost = (
            0.5 * ca.transpose(p - refs[:3]) @ ca.diag(W[:3]) @ (p - refs[:3])
            + 0.5 * dist_quat(q, refs[3:7]) * W[3]
            + 0.5
            * ca.transpose(pdot - refs[7:10])
            @ ca.diag(W[7:10])
            @ (pdot - refs[7:10])
            + 0.5
            * ca.transpose(omega - refs[10:13])
            @ ca.diag(W[10:13])
            @ (omega - refs[10:13])
            + 0.5
            * ca.transpose(pddot - refs[13:16])
            @ ca.diag(W[13:16])
            @ (pddot - refs[13:16])
            + 0.5
            * ca.transpose(omega_dot - refs[16:19])
            @ ca.diag(W[16:19])
            @ (omega_dot - refs[16:19])
            + 0.5 * ca.transpose(controls) @ ca.diag(W_u) @ controls
        )

        terminal_cost = (
            0.5 * ca.transpose(p - refs[:3]) @ ca.diag(W[:3]) @ (p - refs[:3])
            + 0.5 * dist_quat(q, refs[3:7]) * W[3]
            + 0.5
            * ca.transpose(pdot - refs[7:10])
            @ ca.diag(W[7:10])
            @ (pdot - refs[7:10])
            + 0.5
            * ca.transpose(omega - refs[10:13])
            @ ca.diag(W[10:13])
            @ (omega - refs[10:13])
            + 0.5
            * ca.transpose(pddot - refs[13:16])
            @ ca.diag(W[13:16])
            @ (pddot - refs[13:16])
            + 0.5
            * ca.transpose(omega_dot - refs[16:19])
            @ ca.diag(W[16:19])
            @ (omega_dot - refs[16:19])
        )

        return stage_cost, terminal_cost

    def _setup_ocp(self):
        """Setup Acados optimal control problem"""
        states, controls, x_dot = self.dynamics.setup_symbolic_model()

        refs = ca.SX.sym("refs", self.robot_cfg.ny, 1)
        W = ca.SX.sym("W", self.robot_cfg.ny, 1)
        W_u = ca.SX.sym("W_u", self.robot_cfg.nu, 1)

        stage_cost, terminal_cost = self._setup_pac_objective_function(
            states, controls, x_dot, refs, W, W_u
        )

        model = AcadosModel()
        model.name = f"{self.robot_cfg.name}_nmpc"
        model.x = states
        model.u = controls
        model.f_expl_expr = x_dot
        model.p = ca.vertcat(
            refs, W, W_u, self.dynamics.M_sym, self.dynamics.D_sym, self.dynamics.K_sym
        )

        ocp = AcadosOcp()
        ocp.model = model

        ocp.solver_options.N_horizon = self.ctrl_cfg.horizon
        ocp.dims.np = model.p.shape[0]
        ocp.parameter_values = np.zeros((ocp.dims.np,))

        ocp.cost.cost_type = "EXTERNAL"
        ocp.model.cost_expr_ext_cost = stage_cost
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.model.cost_expr_ext_cost_e = terminal_cost

        x0 = self.dynamics.get_initial_state(hover=True)
        ocp.constraints.x0 = x0

        f_min, f_max = self.robot_cfg.get_force_limits()
        force_indices = list(range(13, 13 + self.robot_cfg.nu))
        ocp.constraints.idxbx = np.array(force_indices)
        ocp.constraints.lbx = np.ones(self.robot_cfg.nu) * f_min
        ocp.constraints.ubx = np.ones(self.robot_cfg.nu) * f_max

        lbu, ubu = self.ctrl_cfg.get_control_limits()
        ocp.constraints.idxbu = np.arange(self.robot_cfg.nu)
        ocp.constraints.lbu = lbu
        ocp.constraints.ubu = ubu

        ocp.solver_options.integrator_type = self.ctrl_cfg.integrator_type
        ocp.solver_options.tf = self.ctrl_cfg.horizon * self.ctrl_cfg.dt
        ocp.solver_options.sim_method_num_stages = self.ctrl_cfg.sim_method_num_stages
        ocp.solver_options.sim_method_num_steps = self.ctrl_cfg.sim_method_num_steps

        ocp.solver_options.qp_solver = self.ctrl_cfg.qp_solver
        ocp.solver_options.hpipm_mode = self.ctrl_cfg.hpipm_mode
        ocp.solver_options.qp_solver_warm_start = self.ctrl_cfg.warm_start
        ocp.solver_options.nlp_solver_type = self.ctrl_cfg.nlp_solver
        ocp.solver_options.tol = self.ctrl_cfg.tolerance
        ocp.solver_options.hessian_approx = self.ctrl_cfg.hessian_approx
        ocp.solver_options.regularize_method = self.ctrl_cfg.regularize_method
        ocp.solver_options.reg_epsilon = self.ctrl_cfg.reg_epsilon

        self.solver = AcadosOcpSolver(
            ocp,
            json_file="pac_ocp.json",
            generate=self.ctrl_cfg.gen_build_code,
            build=self.ctrl_cfg.gen_build_code,
        )
        self.integrator = AcadosSimSolver(
            ocp,
            generate=self.ctrl_cfg.gen_build_code,
            build=self.ctrl_cfg.gen_build_code,
        )

    def set_ocp_parameters(
        self,
        reference: dict,
        current_time: float = 0.0,
        reference_fn: Callable[[float], dict] = None,
        contact_forces: np.array = np.zeros(3),
    ):
        """Set OCP parameters

        Args:
            reference: Reference dict at current time
            current_time: Current simulation time (used for look-ahead)
            reference_fn: Reference function that takes time and returns reference
                         dict (required if look_ahead_reference=True)
            contact_forces: Contact forces acting on the end effector (optional,
                           defaults to [0, 0, 0])
        """
        weights = self.ctrl_cfg.get_weight_vector()
        w_u = self.ctrl_cfg.get_control_weight_vector()

        # Impedance parameters
        M = self.ctrl_cfg.M
        D = self.ctrl_cfg.D
        K = self.ctrl_cfg.K

        if self.ctrl_cfg.look_ahead_reference:
            # Set different references for each horizon point
            if reference_fn is None:
                raise ValueError(
                    "reference_fn must be provided when look_ahead_reference=True"
                )

            for i in range(self.ctrl_cfg.horizon + 1):
                # Calculate time for this horizon point
                future_time = current_time + i * self.ctrl_cfg.dt
                future_ref = reference_fn(future_time)

                # Build reference vector for this horizon point
                refs = np.zeros(self.robot_cfg.ny)
                refs[:3] = future_ref.get("position", np.zeros(3))
                refs[3:7] = future_ref.get("quaternion", np.array([1, 0, 0, 0]))
                refs[7:10] = future_ref.get("velocity", np.zeros(3))
                refs[10:13] = future_ref.get("angular_velocity", np.zeros(3))

                param_vector = np.concatenate([refs, weights, w_u, M, D, K])
                self.solver.set(i, "p", param_vector)
        else:
            # Original behavior: same reference for all horizon points
            refs = np.zeros(self.robot_cfg.ny)
            refs[:3] = reference.get("position", np.zeros(3))
            refs[3:7] = reference.get("quaternion", np.array([1, 0, 0, 0]))
            refs[7:10] = reference.get("velocity", np.zeros(3))
            refs[10:13] = reference.get("angular_velocity", np.zeros(3))

            param_vector = np.concatenate([refs, weights, w_u, M, D, K])

            for i in range(self.ctrl_cfg.horizon):
                self.solver.set(i, "p", param_vector)
            self.solver.set(self.ctrl_cfg.horizon, "p", param_vector)

    def compute_control(
        self,
        mj_state: np.ndarray,
        reference: dict,
        current_time: float = 0.0,
        reference_fn: Callable[[float], dict] = None,
        contact_forces: np.ndarray = None,
    ) -> np.ndarray:
        """Compute control action using NMPC

        Args:
            mj_state: Current state from MuJoCo
            reference: Reference dict at current time
            current_time: Current simulation time (used for look-ahead)
            reference_fn: Reference function (required if look_ahead_reference=True)
            contact_forces: Contact forces acting on the end effector (optional, defaults to zero)
        """

        self.set_ocp_parameters(reference, current_time, reference_fn, contact_forces)

        # Add the impedance states to the state vector
        self.impedance_position, self.impedance_velocity = self.get_impedance_state()

        if contact_forces is None:
            contact_forces = np.zeros(3)

        state = np.concatenate([
            mj_state,
            self.impedance_position,
            self.impedance_velocity,
            contact_forces,
        ])

        control = self.solver.solve_for_x0(x0_bar=state, fail_on_nonzero_status=False)

        status = self.solver.get_status()
        if status not in [0, 2, 5]:
            print(f"Warning: Acados solver returned status {status}")

        return control

    def get_impedance_state(self):
        """Get the impedance state"""

        # The internal NMPC dt is higher that the control dt_control
        # (e.g. 0.1s > 0.01s)
        # The simulated impedance state is only updated every
        # (dt/dt_control) control steps
        # e.g. (0.1s/0.01s) = 10 control steps

        if self.counter == 10:
            predicted_state = self.solver.get(1, "x")
            impedance_position = predicted_state[19:22]
            impedance_velocity = predicted_state[22:25]
            self.counter = 1
            return impedance_position, impedance_velocity
        else:
            self.counter += 1
            return self.impedance_position, self.impedance_velocity

    def reset(self):
        """Reset controller"""
        if self.solver is not None:
            x0 = self.dynamics.get_initial_state(hover=True)
            for i in range(self.ctrl_cfg.horizon):
                self.solver.set(i, "x", x0)
            self.solver.set(self.ctrl_cfg.horizon, "x", x0)

    def get_solver_stats(self) -> dict:
        """Get solver statistics"""
        if self.solver is None:
            return {"status": -1, "time": 0.0}

        status = self.solver.get_status()
        solve_time = self.solver.get_stats("time_tot")

        return {"status": status, "time": solve_time}

    def get_prediction(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get predicted state and control trajectories
        Returns:
            states: Array of shape (N+1, nx)
            controls: Array of shape (N, nu)
        """
        N = self.ctrl_cfg.horizon
        nx = self.robot_cfg.nx
        nu = self.robot_cfg.nu

        states = np.zeros((N + 1, nx))
        controls = np.zeros((N, nu))

        for i in range(N):
            states[i, :] = self.solver.get(i, "x")
            controls[i, :] = self.solver.get(i, "u")
        states[N, :] = self.solver.get(N, "x")

        return states, controls


# ============================================================================
# MUJOCO INTERFACE
# ============================================================================


class MuJoCoInterface:
    """Interface between NMPC controller and MuJoCo Fiberthex simulation"""

    def __init__(
        self,
        model,
        controller: NMPCController,
        robot_config: HexarotorConfig,
        sim_config: SimulationConfig,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.controller = controller
        self.robot_config = robot_config
        self.sim_config = sim_config
        self.dt = sim_config.dt

        self.model.opt.timestep = self.dt

        self._find_body()
        self._find_actuators()
        self._find_sensors()
        self._setup_allocation_matrix()

        # Initialize Butterworth filter for contact forces
        self._setup_contact_force_filter()

        self.current_forces = np.zeros(6)
        self.reset_logs()

    def _find_body(self):
        """Find robot body ID"""
        try:
            self.body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "robot"
            )
            print(f"Found robot body (ID: {self.body_id})")
        except Exception:
            self.body_id = 1
            print(f"Warning: Using default body_id: {self.body_id}")

    def _find_actuators(self):
        """Find wrench actuator IDs"""
        actuator_names = [
            "Force_X",
            "Force_Y",
            "Force_Z",
            "Torque_X",
            "Torque_Y",
            "Torque_Z",
        ]

        self.actuator_ids = {}
        for name in actuator_names:
            try:
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
                self.actuator_ids[name] = act_id
            except Exception:
                print(f"Warning: Actuator '{name}' not found")
                self.actuator_ids[name] = None

        print(f"Found actuators: {list(self.actuator_ids.keys())}")

    def _setup_allocation_matrix(self):
        """Setup allocation matrix"""
        self.GF, self.GT = alloc(
            self.robot_config.num_rotors,
            self.robot_config.arm_length,
            self.robot_config.alpha_tilt,
            self.robot_config.beta_tilt,
            self.robot_config.c_f,
            self.robot_config.c_t,
            com=self.robot_config.com_offset.tolist(),
        )

        self.G = np.vstack([self.GF, self.GT])
        print(f"Allocation matrix shape: {self.G.shape}")

    def _find_sensors(self):
        """Find force sensor ID"""
        try:
            self.force_sensor_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_sensor"
            )
            print(f"Found force sensor 'ft_sensor' (ID: {self.force_sensor_id})")
        except Exception:
            self.force_sensor_id = None
            print("Warning: Force sensor 'ft_sensor' not found")

    def _setup_contact_force_filter(self):
        """Setup second-order Butterworth filter for contact force measurements"""
        # Filter parameters
        cutoff_freq = 10.0  # Hz
        filter_order = 2

        # Nyquist frequency (half of sampling frequency)
        nyquist_freq = 0.5 / self.dt

        # Normalized cutoff frequency
        normalized_cutoff = cutoff_freq / nyquist_freq

        # Design Butterworth filter
        self.filter_b, self.filter_a = signal.butter(
            filter_order, normalized_cutoff, btype="low", analog=False
        )

        # Initialize filter state for each force component (x, y, z)
        # zi is the initial condition for the filter delay elements
        self.filter_zi = np.zeros((3, filter_order))

        print(
            f"Contact force filter initialized with order: {filter_order}, and"
            f" cutoff frequency: {cutoff_freq} Hz"
        )

    def _add_reference_sphere_to_scene(
        self, scene, position: np.ndarray, geom_idx: int | None = None
    ):
        """Add reference position sphere to a MuJoCo scene

        Args:
            scene: MuJoCo scene object (either viewer.user_scn or video_renderer.scene)
            position: 3D position of the reference sphere [x, y, z]
            geom_idx: Optional geometry index to use (if None, uses scene.ngeom)
        """
        if geom_idx is None:
            if scene.ngeom >= scene.maxgeom:
                return
            geom_idx = scene.ngeom

        mujoco.mjv_initGeom(
            scene.geoms[geom_idx],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.05, 0, 0]),  # size (radius)
            position,  # position
            np.eye(3).flatten(),  # rotation matrix (identity)
            np.array([0.2, 0.8, 0.2, 0.2]),  # rgba (green, semi-transparent)
        )

        if geom_idx == scene.ngeom:
            scene.ngeom += 1

    def get_contact_forces_from_sensor(self) -> np.ndarray:
        """Read contact forces from the end effector force sensor

        Note: MuJoCo contact sensors return forces in the local (body) frame.
        This method transforms them to the world frame and applies a 2nd order
        Butterworth filter.
        """
        if self.force_sensor_id is None:
            return np.zeros(3)

        # Force sensors in MuJoCo return 3D force vectors in body frame
        # The sensor data is stored in data.sensordata
        # We need to find the correct index in sensordata array
        sensor_adr = self.model.sensor_adr[self.force_sensor_id]
        contact_force_body = self.data.sensordata[sensor_adr : sensor_adr + 3].copy()

        # Transform from body frame to world frame
        # Get current orientation quaternion
        quaternion = self.data.qpos[3:7].copy()

        # Compute rotation matrix from body to world
        R_body2world = quat2rot_numpy(quaternion)

        # compute rotation matrix from end-effector to body
        # TODO: get this from the "End_Effector_Sensor" site's euler angles
        R_ee2body = R.from_euler("xyz", [0, 0, np.deg2rad(-30)]).as_matrix()

        # Transform contact force to world frame
        # (and flip the sign, such that it represents the external force acting on the robot)
        contact_force_world = -R_body2world @ R_ee2body @ contact_force_body

        # Apply Butterworth filter to each force component
        contact_force_filtered = np.zeros(3)
        for i in range(3):
            # Apply filter with state maintenance for continuity
            filtered_value, self.filter_zi[i] = signal.lfilter(
                self.filter_b,
                self.filter_a,
                [contact_force_world[i]],
                zi=self.filter_zi[i],
            )
            contact_force_filtered[i] = filtered_value[0]
        return contact_force_filtered

    def reset_logs(self):
        """Reset data logging arrays"""
        self.log_time = []
        self.log_position = []
        self.log_quaternion = []
        self.log_velocity = []
        self.log_angular_velocity = []
        self.log_impedance_position = []
        self.log_impedance_velocity = []
        self.log_forces = []
        self.log_controls = []
        self.log_wrench = []
        self.log_references = []
        self.log_solve_times = []
        self.log_predicted_states = []
        self.log_predicted_controls = []
        self.log_contact_forces = []

    def get_state_from_mujoco(self) -> np.ndarray:
        """Extract current state from MuJoCo simulation"""
        position = self.data.qpos[:3].copy()
        quaternion = self.data.qpos[3:7].copy()
        velocity = self.data.qvel[:3].copy()
        angular_velocity = self.data.qvel[3:6].copy()
        forces = self.current_forces.copy()

        # Convert from body frame to end-effector frame
        R_b = quat2rot_numpy(quaternion)
        p_ee_b = self.robot_config.ee_offset
        position_ee = position + (R_b @ p_ee_b)
        velocity_ee = velocity + (R_b @ np.cross(angular_velocity, p_ee_b))

        return np.concatenate(
            [position_ee, quaternion, velocity_ee, angular_velocity, forces]
        )

    def rotor_forces_to_wrench(self, forces: np.ndarray) -> np.ndarray:
        """Convert rotor forces to body wrench"""
        return self.G @ forces

    def apply_control_to_mujoco(self, control: np.ndarray, state: np.ndarray):
        """Apply control to MuJoCo simulation"""
        current_forces = state[13:19]

        new_forces = current_forces + control * self.dt

        f_min, f_max = self.robot_config.get_force_limits()
        new_forces = np.clip(new_forces, f_min, f_max)

        self.current_forces = new_forces

        wrench = self.rotor_forces_to_wrench(new_forces)

        if self.actuator_ids["Force_X"] is not None:
            self.data.ctrl[self.actuator_ids["Force_X"]] = wrench[0]
        if self.actuator_ids["Force_Y"] is not None:
            self.data.ctrl[self.actuator_ids["Force_Y"]] = wrench[1]
        if self.actuator_ids["Force_Z"] is not None:
            self.data.ctrl[self.actuator_ids["Force_Z"]] = wrench[2]
        if self.actuator_ids["Torque_X"] is not None:
            self.data.ctrl[self.actuator_ids["Torque_X"]] = wrench[3]
        if self.actuator_ids["Torque_Y"] is not None:
            self.data.ctrl[self.actuator_ids["Torque_Y"]] = wrench[4]
        if self.actuator_ids["Torque_Z"] is not None:
            self.data.ctrl[self.actuator_ids["Torque_Z"]] = wrench[5]

        return wrench

    def initialize_hover(self):
        """Initialize simulation at hover condition"""
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[2] = 1.0

        hover_force = self.robot_config.get_hover_force()
        self.current_forces = np.ones(6) * hover_force

        hover_wrench = self.rotor_forces_to_wrench(self.current_forces)

        for i, key in enumerate([
            "Force_X",
            "Force_Y",
            "Force_Z",
            "Torque_X",
            "Torque_Y",
            "Torque_Z",
        ]):
            actuator_id = self.actuator_ids[key]
            if actuator_id is not None:
                self.data.ctrl[actuator_id] = hover_wrench[i]

        print("Initializing at hover condition...")
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)

        print(f"Initial position: {self.data.qpos[:3]}")
        print(f"Initial quaternion: {self.data.qpos[3:7]}")

    def run_control_loop(
        self,
        reference_fn: Callable[[float], dict],
        duration: float | None = None,
        render: bool = True,
        realtime: bool = True,
        verbose: bool = True,
        video_filename: str | None = None,
    ):
        """Run closed-loop NMPC control with MuJoCo

        Args:
            video_filename: Optional custom filename for video output (overrides sim_config.video_filename)
        """
        self.initialize_hover()
        self.reset_logs()
        self.controller.reset()

        if duration is None:
            duration = self.sim_config.num_steps * self.sim_config.dt

        num_steps = int(duration / self.dt)
        decimation = self.sim_config.decimation

        viewer = None
        video_renderer = None
        video_frames = []
        camera_id = -1

        # Get camera ID (used by both viewer and video renderer)
        try:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "parallel_cam"
            )
        except Exception:
            print("Warning: Camera 'parallel_cam' not found, using default view")

        if render:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # Enable contact points (mjVIS_CONTACTPOINT)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            # Enable contact forces (mjVIS_CONTACTFORCE)
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            # Set camera to parallel view
            if camera_id != -1:
                viewer.cam.fixedcamid = camera_id
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        # Setup video recording if enabled (independent of viewer)
        video_vopt = None
        if self.sim_config.render_video:
            video_renderer = mujoco.Renderer(
                self.model,
                height=self.sim_config.video_height,
                width=self.sim_config.video_width,
            )

            # Create visualization options for video renderer
            video_vopt = mujoco.MjvOption()
            video_vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            video_vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            # Determine final video filename
            final_video_filename = (
                video_filename if video_filename else self.sim_config.video_filename
            )
            print(
                "\nVideo recording enabled:"
                f" {self.sim_config.video_width}x{self.sim_config.video_height} @"
                f" {self.sim_config.video_fps} FPS"
            )
            print(f"Video will be saved as: {final_video_filename}")

        # Store reference position for visualization callback
        self._ref_position = np.zeros(3)

        print(f"\nRunning NMPC control for {duration}s ({num_steps} steps)")
        print(
            f"Control decimation: {decimation} (update every"
            f" {decimation * self.dt:.3f}s)"
        )
        print("=" * 70)

        start_time = time.time()
        last_print_time = 0
        control = np.zeros(self.robot_config.num_rotors)
        frame_interval = (
            int(1.0 / (self.sim_config.video_fps * self.dt))
            if self.sim_config.render_video
            else 0
        )

        try:
            for step in range(num_steps):
                step_start = time.time()
                current_time = step * self.dt

                state = self.get_state_from_mujoco()
                contact_forces = self.get_contact_forces_from_sensor()
                reference = reference_fn(current_time)

                solve_time = 0.0
                predicted_states = None
                predicted_controls = None
                if step % decimation == 0:
                    # Compute control (only every 'decimation' steps)
                    control = self.controller.compute_control(
                        state,
                        reference,
                        current_time,
                        reference_fn,
                        contact_forces,
                    )

                    # Get solver stats if available
                    if hasattr(self.controller, "get_solver_stats"):
                        stats = self.controller.get_solver_stats()
                        solve_time = stats.get("time", 0.0)
                        if stats["status"] not in [0, 2, 5]:
                            print(
                                f"\nWarning at t={current_time:.2f}s: Solver"
                                f" status {stats['status']}"
                            )

                    # Get predictions if available
                    if hasattr(self.controller, "get_prediction"):
                        predicted_states, predicted_controls = (
                            self.controller.get_prediction()
                        )

                wrench = self.apply_control_to_mujoco(control, state)
                mujoco.mj_step(self.model, self.data)

                # Update reference position for visualization
                self._ref_position = reference["position"].copy()

                if render and viewer is not None:
                    viewer.sync()

                    # Add reference position sphere visualization
                    if viewer.user_scn is not None:
                        # Reset user scene geometry count
                        viewer.user_scn.ngeom = 0
                        # Add sphere at reference position
                        self._add_reference_sphere_to_scene(
                            viewer.user_scn, self._ref_position, geom_idx=0
                        )

                # Capture video frame at specified FPS
                if (
                    self.sim_config.render_video
                    and video_renderer is not None
                    and step % frame_interval == 0
                ):
                    # Create camera for video rendering
                    cam = mujoco.MjvCamera()
                    if camera_id != -1:
                        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        cam.fixedcamid = camera_id

                    # Update scene with current data and visualization options
                    scene = video_renderer.scene
                    mujoco.mjv_updateScene(
                        self.model,
                        self.data,
                        video_vopt,
                        None,  # no perturbation
                        cam,
                        mujoco.mjtCatBit.mjCAT_ALL,
                        scene,
                    )

                    # Add reference position sphere to video renderer scene
                    self._add_reference_sphere_to_scene(scene, self._ref_position)

                    # Render the frame
                    frame = video_renderer.render()
                    video_frames.append(frame)

                self.log_time.append(current_time)
                self.log_position.append(state[0:3].copy())
                self.log_quaternion.append(state[3:7].copy())
                self.log_velocity.append(state[7:10].copy())
                self.log_angular_velocity.append(state[10:13].copy())
                self.log_impedance_position.append(
                    self.controller.impedance_position.copy()
                )
                self.log_impedance_velocity.append(
                    self.controller.impedance_velocity.copy()
                )
                self.log_forces.append(self.current_forces.copy())
                self.log_controls.append(control.copy())
                self.log_wrench.append(wrench.copy())
                self.log_references.append(reference)
                self.log_solve_times.append(solve_time)
                self.log_predicted_states.append(predicted_states)
                self.log_predicted_controls.append(predicted_controls)
                self.log_contact_forces.append(
                    contact_forces if "contact_forces" in locals() else np.zeros(3)
                )

                if verbose and (current_time - last_print_time) >= 1.0:
                    pos = state[0:3]
                    ref_pos = reference["position"]
                    error = np.linalg.norm(pos - ref_pos)
                    print(
                        f"t={current_time:5.2f}s | pos=[{pos[0]:6.3f},"
                        f" {pos[1]:6.3f}, {pos[2]:6.3f}] |"
                        f" ref=[{ref_pos[0]:6.3f}, {ref_pos[1]:6.3f},"
                        f" {ref_pos[2]:6.3f}] | err={error:6.4f}m"
                    )
                    last_print_time = current_time

        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")

        finally:
            if viewer is not None:
                print("\nClose the viewer window to continue...")
                while viewer.is_running():
                    time.sleep(0.1)

            # Save video if recording was enabled
            if self.sim_config.render_video and len(video_frames) > 0:
                print(f"\nSaving video with {len(video_frames)} frames...")
                self._save_video_frames(video_frames, final_video_filename)

        total_time = time.time() - start_time
        print("=" * 70)
        print("Simulation complete!")
        print(
            f"Total time: {total_time:.2f}s (realtime factor:"
            f" {duration / total_time:.2f}x)"
        )

        # Construct states array combining all state components
        # States format: [pos(3), quat(4), vel(3), omega(3), forces(6)]
        num_steps = len(self.log_time)
        states = np.zeros((num_steps, 28))  # 3+4+3+3+6+3+3+3 = 28

        for i in range(num_steps):
            states[i, 0:3] = self.log_position[i]
            states[i, 3:7] = self.log_quaternion[i]
            states[i, 7:10] = self.log_velocity[i]
            states[i, 10:13] = self.log_angular_velocity[i]
            states[i, 13:19] = self.log_forces[i]
            states[i, 19:22] = self.log_impedance_position[i]
            states[i, 22:25] = self.log_impedance_velocity[i]
            states[i, 25:28] = self.log_contact_forces[i]

        # Extract reference positions and quaternions from logged references
        ref_positions = np.array(
            [ref.get("position", np.zeros(3)) for ref in self.log_references]
        )
        ref_quaternions = np.array([
            ref.get("quaternion", np.array([1.0, 0.0, 0.0, 0.0]))
            for ref in self.log_references
        ])

        # Return results as dictionary in format expected by ExperimentLogger and ResultsVisualizer
        results = {
            "times": np.array(self.log_time),
            "states": states,
            "controls": np.array(self.log_controls),
            "solve_times": np.array(self.log_solve_times),
            "ref_positions": ref_positions,
            "ref_quaternions": ref_quaternions,
            "predicted_states": self.log_predicted_states,
            "predicted_controls": self.log_predicted_controls,
            "contact_forces": np.array(self.log_contact_forces),
            "impedance_position": np.array(self.log_impedance_position),
            "impedance_velocity": np.array(self.log_impedance_velocity),
        }

        return results

    def _save_video_frames(self, frames: list, filename: str):
        """Save video frames to MP4 file using imageio

        Args:
            frames: List of RGB frame arrays
            filename: Output video filename
        """
        try:
            import imageio
        except ImportError:
            print(
                "Warning: imageio not installed. Install with: pip install"
                " imageio[ffmpeg]"
            )
            return

        # Create logs directory if it doesn't exist
        log_dir = self.sim_config.log_directory
        os.makedirs(log_dir, exist_ok=True)

        # Save individual frames as images if requested
        if self.sim_config.save_video_frames_as_images:
            frames_dir = os.path.join(log_dir, filename.replace(".mp4", "_frames"))
            os.makedirs(frames_dir, exist_ok=True)
            print(f"\nSaving {len(frames)} frames as images to {frames_dir}...")

            for i, frame in enumerate(frames):
                frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
                imageio.imwrite(frame_path, frame)
                if self.sim_config.video_show_progress and (
                    i % 10 == 0 or i == len(frames) - 1
                ):
                    progress = (i + 1) / len(frames) * 100
                    print(
                        f"\r  Progress: {progress:.1f}% ({i + 1}/{len(frames)})",
                        end="",
                        flush=True,
                    )
            print(f"\n✓ Frames saved to: {frames_dir}")

        # Full path for video file
        video_path = os.path.join(log_dir, filename)

        # Write video using imageio
        try:
            writer = imageio.get_writer(
                video_path,
                fps=self.sim_config.video_fps,
                codec="libx264",
                quality=8,  # High quality (scale 0-10)
                pixelformat="yuv420p",  # Ensure compatibility with most players
                macro_block_size=1,
            )

            if self.sim_config.video_show_progress:
                print("\nEncoding video...")

            for i, frame in enumerate(frames):
                writer.append_data(frame)
                if self.sim_config.video_show_progress and (
                    i % 10 == 0 or i == len(frames) - 1
                ):
                    progress = (i + 1) / len(frames) * 100
                    print(
                        f"\r  Progress: {progress:.1f}% ({i + 1}/{len(frames)})",
                        end="",
                        flush=True,
                    )

            writer.close()

            if self.sim_config.video_show_progress:
                print()  # New line after progress bar

            print(f"✓ Video saved to: {video_path}")
            print(
                "  Resolution:"
                f" {self.sim_config.video_width}x{self.sim_config.video_height}"
            )
            print(f"  Frame rate: {self.sim_config.video_fps} FPS")
            print(f"  Total frames: {len(frames)}")
            print(f"  Duration: {len(frames) / self.sim_config.video_fps:.2f}s")

        except Exception as e:
            print(f"Error saving video: {e}")
            print("Make sure ffmpeg is installed on your system")

    def save_predictions_to_file(self, filename: str = "predictions_log.pkl"):
        """
        Save prediction logs to a file with comprehensive metadata

        Args:
            filename: Name of the output file (supports .pkl, .npz)

        Saved data structure (for .pkl files):
            'metadata': Dictionary containing:
                - 'description': Human-readable description of the data format
                - 'timestamp': When the file was created
                - 'version': Data format version
            'times': np.ndarray of shape (n,) - Time stamps in seconds
            'predicted_states': List of n arrays, each of shape (N+1, nx) - Predicted state trajectories from MPC
            'predicted_controls': List of n arrays, each of shape (N, nu) - Predicted control trajectories from MPC
            'references': Dictionary containing reference trajectories:
                - 'position': np.ndarray (n, 3) - Reference positions [x, y, z] in meters
                - 'quaternion': np.ndarray (n, 4) - Reference quaternions [w, x, y, z]
                - 'velocity': np.ndarray (n, 3) - Reference velocities [vx, vy, vz] in m/s
                - 'angular_velocity': np.ndarray (n, 3) - Reference angular velocities [wx, wy, wz] in rad/s
            'actual_states': Dictionary containing actual measured states:
                - 'position': np.ndarray (n, 3) - End-effector position [x, y, z] in meters
                - 'quaternion': np.ndarray (n, 4) - Body quaternion [w, x, y, z]
                - 'velocity': np.ndarray (n, 3) - End-effector velocity [vx, vy, vz] in m/s
                - 'angular_velocity': np.ndarray (n, 3) - Body angular velocity [wx, wy, wz] in rad/s
                - 'rotor_forces': np.ndarray (n, 6) - Individual rotor thrust forces in N
                - 'impedance_position': np.ndarray (n, 3) - Virtual impedance position in meters
                - 'impedance_velocity': np.ndarray (n, 3) - Virtual impedance velocity in m/s
                - 'contact_forces': np.ndarray (n, 3) - Contact forces at end-effector [fx, fy, fz] in N
            'actual_controls': np.ndarray (n, 6) - Applied control inputs (rotor force derivatives dF/dt)
            'solve_times': np.ndarray (n,) - MPC solver computation times in seconds
            'robot_config': Dictionary containing all robot parameters (mass, inertia, geometry, etc.)
            'controller_config': Dictionary containing all controller parameters (weights, horizon, etc.)
            'simulation_config': Dictionary containing simulation parameters (dt, duration, etc.)
        """

        # Create logs directory if it doesn't exist
        log_dir = self.sim_config.log_directory
        os.makedirs(log_dir, exist_ok=True)

        filepath = os.path.join(log_dir, filename)

        # Prepare data for saving
        # Convert references from list of dicts to dict of arrays
        ref_positions = np.array(
            [ref.get("position", np.zeros(3)) for ref in self.log_references]
        )
        ref_quaternions = np.array([
            ref.get("quaternion", np.array([1.0, 0.0, 0.0, 0.0]))
            for ref in self.log_references
        ])
        ref_velocities = np.array(
            [ref.get("velocity", np.zeros(3)) for ref in self.log_references]
        )
        ref_angular_velocities = np.array(
            [ref.get("angular_velocity", np.zeros(3)) for ref in self.log_references]
        )

        # Convert config dataclasses to dictionaries for serialization
        robot_config_dict = {
            "name": self.robot_config.name,
            "mass": self.robot_config.mass,
            "arm_length": self.robot_config.arm_length,
            "alpha_tilt": self.robot_config.alpha_tilt,
            "beta_tilt": self.robot_config.beta_tilt,
            "Jxx": self.robot_config.Jxx,
            "Jyy": self.robot_config.Jyy,
            "Jzz": self.robot_config.Jzz,
            "gravity": self.robot_config.gravity,
            "com_offset": self.robot_config.com_offset,
            "ee_offset": self.robot_config.ee_offset,
            "num_rotors": self.robot_config.num_rotors,
            "c_f": self.robot_config.c_f,
            "c_t": self.robot_config.c_t,
            "w_min": self.robot_config.w_min,
            "w_max": self.robot_config.w_max,
            "nx": self.robot_config.nx,
            "nu": self.robot_config.nu,
            "ny": self.robot_config.ny,
        }

        controller_config_dict = {
            "dt": self.controller.ctrl_cfg.dt,
            "horizon": self.controller.ctrl_cfg.horizon,
            "M": self.controller.ctrl_cfg.M,
            "D": self.controller.ctrl_cfg.D,
            "K": self.controller.ctrl_cfg.K,
            "integrator_type": self.controller.ctrl_cfg.integrator_type,
            "qp_solver": self.controller.ctrl_cfg.qp_solver,
            "nlp_solver": self.controller.ctrl_cfg.nlp_solver,
            "hpipm_mode": self.controller.ctrl_cfg.hpipm_mode,
            "tolerance": self.controller.ctrl_cfg.tolerance,
            "hessian_approx": self.controller.ctrl_cfg.hessian_approx,
            "regularize_method": self.controller.ctrl_cfg.regularize_method,
            "reg_epsilon": self.controller.ctrl_cfg.reg_epsilon,
            "warm_start": self.controller.ctrl_cfg.warm_start,
            "sim_method_num_stages": self.controller.ctrl_cfg.sim_method_num_stages,
            "sim_method_num_steps": self.controller.ctrl_cfg.sim_method_num_steps,
            "position_weight": self.controller.ctrl_cfg.position_weight,
            "quaternion_weight": self.controller.ctrl_cfg.quaternion_weight,
            "velocity_weight": self.controller.ctrl_cfg.velocity_weight,
            "angular_velocity_weight": self.controller.ctrl_cfg.angular_velocity_weight,
            "force_weight": self.controller.ctrl_cfg.force_weight,
            "control_weight": self.controller.ctrl_cfg.control_weight,
            "df_min": self.controller.ctrl_cfg.df_min,
            "df_max": self.controller.ctrl_cfg.df_max,
            "look_ahead_reference": self.controller.ctrl_cfg.look_ahead_reference,
        }

        simulation_config_dict = {
            "dt": self.sim_config.dt,
            "dt_control": self.sim_config.dt_control,
            "decimation": self.sim_config.decimation,
            "num_steps": self.sim_config.num_steps,
            "duration": len(self.log_time) * self.sim_config.dt,
            "log_directory": self.sim_config.log_directory,
        }

        predictions_data = {
            "metadata": {
                "description": (
                    "Predictive Admittance Control (PAC) simulation log file"
                ),
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
                "data_format": {
                    "times": "Time stamps (n,) [seconds]",
                    "predicted_states": (
                        "List of predicted state trajectories from MPC, each (N+1, nx)"
                    ),
                    "predicted_controls": (
                        "List of predicted control trajectories from MPC, each (N, nu)"
                    ),
                    "references": (
                        "Reference trajectories as dict with position(n,3),"
                        " quaternion(n,4), velocity(n,3),"
                        " angular_velocity(n,3)"
                    ),
                    "actual_states": (
                        "Measured states as dict with position(n,3),"
                        " quaternion(n,4), velocity(n,3),"
                        " angular_velocity(n,3), rotor_forces(n,6),"
                        " impedance_position(n,3), impedance_velocity(n,3),"
                        " contact_forces(n,3)"
                    ),
                    "actual_controls": "Applied control inputs (n, nu) [N/s]",
                    "solve_times": "MPC solver times (n,) [seconds]",
                },
            },
            "times": np.array(self.log_time),
            "predicted_states": self.log_predicted_states,
            "predicted_controls": self.log_predicted_controls,
            "references": {
                "position": ref_positions,
                "quaternion": ref_quaternions,
                "velocity": ref_velocities,
                "angular_velocity": ref_angular_velocities,
            },
            "actual_states": {
                "position": np.array(self.log_position),
                "quaternion": np.array(self.log_quaternion),
                "velocity": np.array(self.log_velocity),
                "angular_velocity": np.array(self.log_angular_velocity),
                "rotor_forces": np.array(self.log_forces),
                "impedance_position": np.array(self.log_impedance_position),
                "impedance_velocity": np.array(self.log_impedance_velocity),
                "contact_forces": np.array(self.log_contact_forces),
            },
            "actual_controls": np.array(self.log_controls),
            "solve_times": np.array(self.log_solve_times),
            "robot_config": robot_config_dict,
            "controller_config": controller_config_dict,
            "simulation_config": simulation_config_dict,
        }

        if filename.endswith(".pkl"):
            with open(filepath, "wb") as f:
                pickle.dump(predictions_data, f)
            print(f"✓ Predictions saved to: {filepath}")
            print(
                "  Includes metadata, all configurations, and"
                f" {len(self.log_time)} time steps"
            )
            print(
                f"  To inspect: use inspect_log_file('{filepath}') or load"
                " with pickle.load()"
            )
        elif filename.endswith(".npz"):
            # For npz, we need to flatten the nested structure
            np.savez(
                filepath,
                **{
                    "times": predictions_data["times"],
                    "actual_positions": predictions_data["actual_states"]["position"],
                    "actual_quaternions": predictions_data["actual_states"][
                        "quaternion"
                    ],
                    "actual_velocities": predictions_data["actual_states"]["velocity"],
                    "actual_angular_velocities": predictions_data["actual_states"][
                        "angular_velocity"
                    ],
                    "actual_forces": predictions_data["actual_states"]["rotor_forces"],
                    "actual_controls": predictions_data["actual_controls"],
                    "solve_times": predictions_data["solve_times"],
                },
            )
            print(f"✓ Predictions saved to: {filepath}")
            print(
                "  Note: .npz format doesn't include metadata, configs, or"
                " predicted trajectories. Use .pkl for full data."
            )
        else:
            raise ValueError(f"Unsupported file format: {filename}. Use .pkl or .npz")

        return filepath


# ============================================================================
# REFERENCE TRAJECTORY FUNCTIONS
# ============================================================================


def constant_reference(
    position=None, quaternion=None, velocity=None, angular_velocity=None
):
    """Create a constant reference trajectory function"""
    ref_dict = {}
    if position is not None:
        ref_dict["position"] = position
    if quaternion is not None:
        ref_dict["quaternion"] = quaternion
    if velocity is not None:
        ref_dict["velocity"] = velocity
    if angular_velocity is not None:
        ref_dict["angular_velocity"] = angular_velocity

    return lambda t: ref_dict


def step_reference(time_switch: float, ref_before: dict, ref_after: dict):
    """Create a step reference that switches at a given time"""
    return lambda t: ref_before if t < time_switch else ref_after


def trajectory_reference(
    positions: np.ndarray, quaternions: np.ndarray, times: np.ndarray
) -> Callable[[float], dict]:
    """
    Create a time-parameterized reference trajectory
    Args:
        positions: Array of positions (N, 3)
        quaternions: Array of quaternions (N, 4)
        times: Time stamps for each point (N,)
    Returns:
        Function that interpolates reference for any time
    """

    def reference_fn(t):
        # Find nearest time index
        idx = np.searchsorted(times, t)
        idx = min(idx, len(times) - 1)

        return {"position": positions[idx], "quaternion": quaternions[idx]}

    return reference_fn


# ============================================================================
# EXAMPLE FUNCTIONS
# ============================================================================


def example_fiberthex_hover():
    """Example 1: Hover at fixed position"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Hover Control with Fiberthex in MuJoCo")
    print("=" * 70)

    robot_config = HexarotorConfig(
        mass=2.0,
        arm_length=0.39,
        alpha_tilt=20.0,
        Jxx=0.06185585085,
        Jyy=0.46467853407,
        Jzz=0.47162119244,
    )

    controller_config = ControllerConfig(
        dt=0.1,
        horizon=10,
        position_weight=np.array([2e1, 2e1, 2e1]),
        quaternion_weight=1e4,
        velocity_weight=np.array([10.0, 10.0, 10.0]),
        angular_velocity_weight=np.array([10.0, 10.0, 10.0]),
        control_weight=0.1,
        gen_build_code=True,
    )

    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sim_config = SimulationConfig(
        num_steps=10000,
        render_video=True,
        video_fps=30,
        video_width=1920,
        video_height=1080,
    )

    print("\nInitializing NMPC controller...")
    dynamics = HexarotorDynamics(robot_config)
    controller = NMPCController(dynamics, controller_config, robot_config)
    print("✓ Controller initialized successfully!")

    print("\nCreating MuJoCo interface...")
    mujoco_interface = MuJoCoInterface(
        FiberthexWallScene, controller, robot_config, sim_config
    )
    print("✓ Interface created successfully!")

    reference = constant_reference(
        position=np.array([0.0, 0.0, 1.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    results = mujoco_interface.run_control_loop(
        reference_fn=reference,
        duration=10.0,
        render=True,
        realtime=True,
        verbose=True,
        video_filename=f"hover_{timestamp}.mp4",
    )

    # Log results to CSV
    print("\nLogging results...")
    logger = ExperimentLogger(robot_config, controller_config, sim_config)
    logger.log_results(results, experiment_name="hover", reference_trajectory=reference)

    # Save predictions to file
    print("\nSaving predictions...")
    mujoco_interface.save_predictions_to_file(f"hover_predictions_{timestamp}.pkl")

    # Visualize results
    print("\nGenerating visualization...")
    visualizer = ResultsVisualizer(results)
    visualizer.plot_all(save_path=f"logs/hover_{timestamp}.png")


def example_fiberthex_tracking():
    """Example 2: Track position step change"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Position Tracking with Fiberthex")
    print("=" * 70)

    robot_config = HexarotorConfig(
        mass=2.0,
        arm_length=0.39,
        alpha_tilt=20.0,
        Jxx=0.06185585085,
        Jyy=0.46467853407,
        Jzz=0.47162119244,
    )

    controller_config = ControllerConfig(
        dt=0.1,
        horizon=10,
        position_weight=np.array([3e1, 3e1, 3e1]),
        quaternion_weight=1e4,
        velocity_weight=np.array([15.0, 15.0, 15.0]),
        angular_velocity_weight=np.array([15.0, 15.0, 15.0]),
        control_weight=0.1,
        gen_build_code=True,
    )

    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sim_config = SimulationConfig(
        num_steps=10000,
        render_video=True,
        video_fps=30,
        video_width=1920,
        video_height=1080,
    )

    print("\nInitializing controller...")
    dynamics = HexarotorDynamics(robot_config)
    controller = NMPCController(dynamics, controller_config, robot_config)

    mujoco_interface = MuJoCoInterface(
        FiberthexWallScene, controller, robot_config, sim_config
    )

    ref_before = {
        "position": np.array([0.0, 0.0, 0.5]),
        "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    ref_after = {
        "position": np.array([1.0, 0.5, 0.8]),
        "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    reference = step_reference(3.0, ref_before, ref_after)

    results = mujoco_interface.run_control_loop(
        reference_fn=reference,
        duration=10.0,
        render=True,
        realtime=True,
        verbose=True,
        video_filename=f"tracking_{timestamp}.mp4",
    )

    # Log results to CSV
    print("\nLogging results...")
    logger = ExperimentLogger(robot_config, controller_config, sim_config)
    logger.log_results(
        results, experiment_name="tracking", reference_trajectory=reference
    )

    # Save predictions to file
    print("\nSaving predictions...")
    mujoco_interface.save_predictions_to_file(f"tracking_predictions_{timestamp}.pkl")

    # Visualize results
    print("\nGenerating visualization...")
    visualizer = ResultsVisualizer(results)
    visualizer.plot_all(save_path=f"logs/tracking_{timestamp}.png")


def example_fiberthex_circle():
    """Example 3: Follow circular trajectory"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Circular Trajectory with Fiberthex")
    print("=" * 70)

    robot_config = HexarotorConfig(
        mass=2.0,
        arm_length=0.39,
        alpha_tilt=20.0,
        Jxx=0.06185585085,
        Jyy=0.46467853407,
        Jzz=0.47162119244,
    )

    controller_config = ControllerConfig(
        dt=0.1,
        horizon=10,
        position_weight=np.array([4e1, 4e1, 4e1]),
        quaternion_weight=1e4,
        velocity_weight=np.array([10.0, 10.0, 10.0]),
        angular_velocity_weight=np.array([10.0, 10.0, 10.0]),
        control_weight=0.1,
        gen_build_code=True,
    )

    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sim_config = SimulationConfig(
        num_steps=10000,
        render_video=True,
        video_fps=30,
        video_width=1920,
        video_height=1080,
    )

    print("\nInitializing controller...")
    dynamics = HexarotorDynamics(robot_config)
    controller = NMPCController(dynamics, controller_config, robot_config)

    mujoco_interface = MuJoCoInterface(
        FiberthexWallScene, controller, robot_config, sim_config
    )

    def circular_trajectory(t):
        """Circle in XY plane"""
        radius = 0.8
        omega = 0.4
        center_z = 0.8

        if t < 2.0:
            return {
                "position": np.array([0.0, 0.0, center_z]),
                "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
            }

        t_motion = t - 2.0
        return {
            "position": np.array([
                radius * np.cos(omega * t_motion),
                radius * np.sin(omega * t_motion),
                center_z,
            ]),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        }

    results = mujoco_interface.run_control_loop(
        reference_fn=circular_trajectory,
        duration=10.0,
        render=True,
        realtime=True,
        verbose=True,
        video_filename=f"circle_{timestamp}.mp4",
    )

    # Log results to CSV
    print("\nLogging results...")
    logger = ExperimentLogger(robot_config, controller_config, sim_config)
    logger.log_results(
        results,
        experiment_name="circle",
        reference_trajectory=circular_trajectory,
    )

    # Save predictions to file
    print("\nSaving predictions...")
    mujoco_interface.save_predictions_to_file(f"circle_predictions_{timestamp}.pkl")

    # Visualize results
    print("\nGenerating visualization...")
    visualizer = ResultsVisualizer(results)
    visualizer.plot_all(save_path=f"logs/circle_{timestamp}.png")


def example_fiberthex_interaction():
    """Example 4: Physical Interaction example"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Physical Interaction with the wall.")
    print("=" * 70)

    robot_config = HexarotorConfig(
        mass=2.0,
        arm_length=0.39,
        alpha_tilt=20.0,
        Jxx=0.06185585085,
        Jyy=0.46467853407,
        Jzz=0.47162119244,
    )

    controller_config = ControllerConfig(
        dt=0.1,
        horizon=10,
        position_weight=np.array([3e1, 3e1, 3e1]),
        quaternion_weight=1e8,
        velocity_weight=np.array([15.0, 15.0, 15.0]),
        angular_velocity_weight=np.array([15.0, 15.0, 15.0]),
        control_weight=0.1,
        gen_build_code=True,
    )

    # Generate timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sim_config = SimulationConfig(
        num_steps=2000,
        render_video=True,
        video_fps=30,
        video_width=1920,
        video_height=1080,
    )

    print("\nInitializing controller...")
    dynamics = HexarotorDynamics(robot_config)
    controller = NMPCController(dynamics, controller_config, robot_config)

    mujoco_interface = MuJoCoInterface(
        FiberthexWallScene, controller, robot_config, sim_config
    )

    positions = np.array([
        [1.0, 0.0, 1.0],
        [2.9, 0.0, 1.0],
        [3.4, 0.0, 1.0],
        [3.8, 0.0, 1.0],
        [2.5, 0.0, 1.0],
    ])
    quaternions = np.array([
        [np.cos(np.deg2rad(15)), 0.0, 0.0, np.sin(np.deg2rad(15))],
        [np.cos(np.deg2rad(15)), 0.0, 0.0, np.sin(np.deg2rad(15))],
        [np.cos(np.deg2rad(15)), 0.0, 0.0, np.sin(np.deg2rad(15))],
        [np.cos(np.deg2rad(15)), 0.0, 0.0, np.sin(np.deg2rad(15))],
        [np.cos(np.deg2rad(15)), 0.0, 0.0, np.sin(np.deg2rad(15))],
    ])
    times = np.array([2.0, 5.0, 7.0, 10.0, 12.0])
    reference = trajectory_reference(positions, quaternions, times)

    results = mujoco_interface.run_control_loop(
        reference_fn=reference,
        duration=15.0,
        render=True,
        realtime=True,
        verbose=True,
        video_filename=f"APhI_{timestamp}.mp4",
    )

    # Log results to CSV
    print("\nLogging results...")
    logger = ExperimentLogger(robot_config, controller_config, sim_config)
    logger.log_results(results, experiment_name="APhI", reference_trajectory=reference)

    # Save predictions to file
    print("\nSaving predictions...")
    mujoco_interface.save_predictions_to_file(f"APhI_{timestamp}.pkl")

    # Visualize results
    print("\nGenerating visualization...")
    visualizer = ResultsVisualizer(results)
    visualizer.plot_all(save_path=f"logs/APhI_{timestamp}.png")


def inspect_log_file(filepath: str):
    """
    Helper function to inspect and print summary of a PAC log file

    Args:
        filepath: Path to the .pkl log file

    Example:
        >>> inspect_log_file('logs/APhI_20250101_120000.pkl')
    """
    print(f"\n{'=' * 70}")
    print(f"INSPECTING LOG FILE: {filepath}")
    print(f"{'=' * 70}\n")

    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # Print metadata
        if "metadata" in data:
            print("METADATA:")
            print(f"  Description: {data['metadata'].get('description', 'N/A')}")
            print(f"  Timestamp: {data['metadata'].get('timestamp', 'N/A')}")
            print(f"  Version: {data['metadata'].get('version', 'N/A')}")
            print()

        # Print data summary
        print("DATA SUMMARY:")
        print(
            "  Number of time steps:"
            f" {len(data['times']) if 'times' in data else 'N/A'}"
        )
        print(
            f"  Duration: {data['times'][-1]:.2f}s"
            if "times" in data and len(data["times"]) > 0
            else "  Duration: N/A"
        )
        print(
            f"  Average solve time: {np.mean(data['solve_times']):.4f}s"
            if "solve_times" in data
            else "  Average solve time: N/A"
        )
        print()

        # Print configuration summaries
        if "robot_config" in data:
            print("ROBOT CONFIG:")
            print(f"  Mass: {data['robot_config'].get('mass', 'N/A')} kg")
            print(f"  Arm length: {data['robot_config'].get('arm_length', 'N/A')} m")
            print(f"  Alpha tilt: {data['robot_config'].get('alpha_tilt', 'N/A')} deg")
            print(f"  State dimension (nx): {data['robot_config'].get('nx', 'N/A')}")
            print(f"  Control dimension (nu): {data['robot_config'].get('nu', 'N/A')}")
            print()

        if "controller_config" in data:
            print("CONTROLLER CONFIG:")
            print(f"  Horizon: {data['controller_config'].get('horizon', 'N/A')}")
            print(f"  Control dt: {data['controller_config'].get('dt', 'N/A')}s")
            print(f"  Impedance M: {data['controller_config'].get('M', 'N/A')}")
            print(f"  Impedance D: {data['controller_config'].get('D', 'N/A')}")
            print(f"  Impedance K: {data['controller_config'].get('K', 'N/A')}")
            print(
                "  Position weight:"
                f" {data['controller_config'].get('position_weight', 'N/A')}"
            )
            print(
                "  Quaternion weight:"
                f" {data['controller_config'].get('quaternion_weight', 'N/A')}"
            )
            print()

        if "simulation_config" in data:
            print("SIMULATION CONFIG:")
            print(f"  Simulation dt: {data['simulation_config'].get('dt', 'N/A')}s")
            print(
                f"  Control dt: {data['simulation_config'].get('dt_control', 'N/A')}s"
            )
            print(f"  Decimation: {data['simulation_config'].get('decimation', 'N/A')}")
            print(f"  Duration: {data['simulation_config'].get('duration', 'N/A')}s")
            print()

        # Print available data fields
        print("AVAILABLE DATA FIELDS:")
        for key in data.keys():
            if key == "metadata":
                continue
            if isinstance(data[key], dict):
                print(f"  {key}:")
                for subkey, value in data[key].items():
                    if isinstance(value, np.ndarray):
                        print(f"    {subkey}: shape {value.shape}, dtype {value.dtype}")
                    else:
                        print(f"    {subkey}: {type(value).__name__}")
            elif isinstance(data[key], np.ndarray):
                print(f"  {key}: shape {data[key].shape}, dtype {data[key].dtype}")
            elif isinstance(data[key], list):
                print(f"  {key}: list with {len(data[key])} elements")
                if len(data[key]) > 0 and isinstance(data[key][0], np.ndarray):
                    print(f"    First element shape: {data[key][0].shape}")
            else:
                print(f"  {key}: {type(data[key]).__name__}")

        print(f"\n{'=' * 70}\n")

        return data

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Main function to run examples"""
    print("\n" + "=" * 70)
    print("Fiberthex MuJoCo NMPC Control Examples (Standalone Version)")
    print("=" * 70)
    print("\nSelect example to run:")
    print("1. Hover control (maintain position)")
    print("2. Position tracking")
    print("3. Circular trajectory")
    print("4. Physical interaction")
    print("5. Inspect a log file")

    choice = input("\nEnter choice (1-5, or Enter for default=4): ").strip()

    if not choice:
        choice = "4"

    try:
        if choice == "1":
            example_fiberthex_hover()
        elif choice == "2":
            example_fiberthex_tracking()
        elif choice == "3":
            example_fiberthex_circle()
        elif choice == "4":
            example_fiberthex_interaction()
        elif choice == "5":
            filepath = input("Enter path to log file: ").strip()
            inspect_log_file(filepath)
        else:
            print("Invalid choice. Running default (hover control)...")
            example_fiberthex_hover()

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
