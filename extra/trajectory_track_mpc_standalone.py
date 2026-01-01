"""
Standalone MuJoCo Fiberthex NMPC Control Example
This version imports utility functions from the pac module.
"""

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import casadi as ca
import mujoco
import mujoco.viewer
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import utility functions from pac module
from utils.utils import alloc, dist_quat, hamilton_prod, quat2rot_casadi

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

    num_rotors: int = 6
    c_f: float = 9.9e-4
    c_t: float = 1.9e-5
    w_min: float = 16.0
    w_max: float = 110.0

    nx: int = 19  # States: [pos(3), quat(4), vel(3), angvel(3), forces(6)]
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

    integrator_type: str = "ERK"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    nlp_solver: str = "SQP_RTI"
    hpipm_mode: str = "BALANCE"
    tolerance: float = 1e-6
    hessian_approx: str = "GAUSS_NEWTON"
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
    decimation: int = 10
    num_steps: int = 10000


# ============================================================================
# HEXAROTOR DYNAMICS
# ============================================================================


class HexarotorDynamics:
    """Hexarotor dynamics model"""

    def __init__(self, config: HexarotorConfig):
        self.config = config
        self.states = None
        self.controls = None
        self.x_dot = None
        self.GF = None
        self.GT = None

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
        forces = states[13:19]

        self.GF, self.GT = alloc(
            cfg.num_rotors,
            cfg.arm_length,
            cfg.alpha_tilt,
            cfg.beta_tilt,
            cfg.c_f,
            cfg.c_t,
            com=cfg.com_offset.tolist(),
        )

        R = quat2rot_casadi(q)
        J = cfg.get_inertia_matrix()
        J_ca = ca.SX(J)
        Jinv = ca.inv(J_ca)

        pos_dot = vel
        quat_dot = hamilton_prod(q, ca.vertcat(0, omega)) / 2
        vel_dot = (R @ self.GF @ forces) / cfg.mass + cfg.gravity

        B_p_com = ca.SX(cfg.com_offset)
        omega_dot = Jinv @ (
            self.GT @ forces
            - ca.cross(omega, J_ca @ omega)
            + cfg.mass * ca.cross(B_p_com, ca.transpose(R) @ cfg.gravity)
        )

        forces_dot = controls

        x_dot = ca.vertcat(pos_dot, quat_dot, vel_dot, omega_dot, forces_dot)

        self.states = states
        self.controls = controls
        self.x_dot = x_dot

        return states, controls, x_dot

    def get_output_expressions(self) -> tuple[ca.SX, ca.SX]:
        """Get output expressions for cost function"""
        if self.states is None:
            self.setup_symbolic_model()

        h = ca.vertcat(self.states[0:13], self.x_dot[7:13])

        cfg = self.config
        q = self.states[3:7] / ca.norm_2(self.states[3:7])
        omega = self.states[10:13]
        forces = self.states[13:19]

        R = quat2rot_casadi(q)
        J_ca = ca.SX(cfg.get_inertia_matrix())
        Jinv = ca.inv(J_ca)
        B_p_com = ca.SX(cfg.com_offset)

        vel_dot_terminal = (R @ self.GF @ forces) / cfg.mass + cfg.gravity
        omega_dot_terminal = Jinv @ (
            self.GT @ forces
            - ca.cross(omega, J_ca @ omega)
            + cfg.mass * ca.cross(B_p_com, ca.transpose(R) @ cfg.gravity)
        )

        hN = ca.vertcat(self.states[0:13], vel_dot_terminal, omega_dot_terminal)

        return h, hN

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

        self._setup_ocp()

    def _setup_ocp(self):
        """Setup Acados optimal control problem"""
        states, controls, x_dot = self.dynamics.setup_symbolic_model()
        h, hN = self.dynamics.get_output_expressions()

        refs = ca.SX.sym("refs", self.robot_cfg.ny, 1)
        W = ca.SX.sym("W", self.robot_cfg.ny, 1)
        W_u = ca.SX.sym("W_u", self.robot_cfg.nu, 1)

        model = AcadosModel()
        model.name = f"{self.robot_cfg.name}_nmpc"
        model.x = states
        model.u = controls
        model.f_expl_expr = x_dot
        model.p = ca.vertcat(refs, W, W_u)

        stage_cost = (
            0.5 * ca.transpose(h[:3] - refs[:3]) @ ca.diag(W[:3]) @ (h[:3] - refs[:3])
            + 0.5 * dist_quat(h[3:7], refs[3:7]) * W[3]
            + 0.5 * ca.transpose(h[7:] - refs[7:]) @ ca.diag(W[7:]) @ (h[7:] - refs[7:])
            + 0.5 * ca.transpose(controls) @ ca.diag(W_u) @ controls
        )

        terminal_cost = (
            0.5 * ca.transpose(hN[:3] - refs[:3]) @ ca.diag(W[:3]) @ (hN[:3] - refs[:3])
            + 0.5 * dist_quat(hN[3:7], refs[3:7]) * W[3]
            + 0.5
            * ca.transpose(hN[7:] - refs[7:])
            @ ca.diag(W[7:])
            @ (hN[7:] - refs[7:])
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
            json_file="trajectory_track_ocp.json",
            generate=self.ctrl_cfg.gen_build_code,
            build=self.ctrl_cfg.gen_build_code,
        )
        self.integrator = AcadosSimSolver(
            ocp,
            generate=self.ctrl_cfg.gen_build_code,
            build=self.ctrl_cfg.gen_build_code,
        )

    def set_reference(self, reference: dict):
        """Set reference trajectory"""
        refs = np.zeros(self.robot_cfg.ny)
        refs[:3] = reference.get("position", np.zeros(3))
        refs[3:7] = reference.get("quaternion", np.array([1, 0, 0, 0]))
        refs[7:10] = reference.get("velocity", np.zeros(3))
        refs[10:13] = reference.get("angular_velocity", np.zeros(3))

        weights = self.ctrl_cfg.get_weight_vector()
        w_u = self.ctrl_cfg.get_control_weight_vector()

        param_vector = np.concatenate([refs, weights, w_u])

        for i in range(self.ctrl_cfg.horizon):
            self.solver.set(i, "p", param_vector)
        self.solver.set(self.ctrl_cfg.horizon, "p", param_vector)

    def compute_control(self, state: np.ndarray, reference: dict) -> np.ndarray:
        """Compute control action using NMPC"""
        self.set_reference(reference)
        control = self.solver.solve_for_x0(x0_bar=state)

        status = self.solver.get_status()
        if status not in [0, 2, 5]:
            print(f"Warning: Acados solver returned status {status}")

        return control

    def reset(self):
        """Reset controller"""
        if self.solver is not None:
            x0 = self.dynamics.get_initial_state(hover=True)
            for i in range(self.ctrl_cfg.horizon):
                self.solver.set(i, "x", x0)
            self.solver.set(self.ctrl_cfg.horizon, "x", x0)


# ============================================================================
# MUJOCO INTERFACE
# ============================================================================


class FiberthexMuJoCoInterface:
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
        self._setup_allocation_matrix()

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

    def reset_logs(self):
        """Reset data logging arrays"""
        self.log_time = []
        self.log_position = []
        self.log_quaternion = []
        self.log_velocity = []
        self.log_angular_velocity = []
        self.log_forces = []
        self.log_controls = []
        self.log_wrench = []
        self.log_references = []

    def get_state_from_mujoco(self) -> np.ndarray:
        """Extract current state from MuJoCo simulation"""
        position = self.data.qpos[:3].copy()
        quaternion = self.data.qpos[3:7].copy()
        velocity = self.data.qvel[:3].copy()
        angular_velocity = self.data.qvel[3:6].copy()
        forces = self.current_forces.copy()

        return np.concatenate(
            [position, quaternion, velocity, angular_velocity, forces]
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

        self.data.qpos[2] = 0.5

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
    ):
        """Run closed-loop NMPC control with MuJoCo"""
        self.initialize_hover()
        self.reset_logs()
        self.controller.reset()

        if duration is None:
            duration = self.sim_config.num_steps * self.sim_config.dt

        num_steps = int(duration / self.dt)
        decimation = self.sim_config.decimation

        viewer = None
        if render:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)

            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, "scene"
            )
            if camera_id != -1:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = self.model.cam_pos[camera_id]
                viewer.cam.distance = 2.0
                viewer.cam.elevation = 0
                viewer.cam.azimuth = -20

        print(f"\nRunning NMPC control for {duration}s ({num_steps} steps)")
        print(
            f"Control decimation: {decimation} (update every"
            f" {decimation * self.dt:.3f}s)"
        )
        print("=" * 70)

        start_time = time.time()
        last_print_time = 0
        control = np.zeros(self.robot_config.num_rotors)

        try:
            for step in range(num_steps):
                step_start = time.time()
                current_time = step * self.dt

                state = self.get_state_from_mujoco()
                reference = reference_fn(current_time)

                if step % decimation == 0:
                    control = self.controller.compute_control(state, reference)

                wrench = self.apply_control_to_mujoco(control, state)
                mujoco.mj_step(self.model, self.data)

                if render and viewer is not None:
                    viewer.sync()

                self.log_time.append(current_time)
                self.log_position.append(state[0:3].copy())
                self.log_quaternion.append(state[3:7].copy())
                self.log_velocity.append(state[7:10].copy())
                self.log_angular_velocity.append(state[10:13].copy())
                self.log_forces.append(self.current_forces.copy())
                self.log_controls.append(control.copy())
                self.log_wrench.append(wrench.copy())
                self.log_references.append(reference)

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

                if realtime:
                    elapsed = time.time() - step_start
                    sleep_time = self.dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")

        finally:
            if viewer is not None:
                print("\nClose the viewer window to continue...")
                while viewer.is_running():
                    time.sleep(0.1)

        total_time = time.time() - start_time
        print("=" * 70)
        print("Simulation complete!")
        print(
            f"Total time: {total_time:.2f}s (realtime factor:"
            f" {duration / total_time:.2f}x)"
        )


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


# ============================================================================
# EXAMPLE FUNCTIONS
# ============================================================================


def example_fiberthex_hover():
    """Example 1: Hover at fixed position"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Hover Control with Fiberthex in MuJoCo")
    print("=" * 70)

    # Import the MuJoCo model
    from models.mujoco.fiberthex_wall import FiberthexWallScene

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

    sim_config = SimulationConfig(
        dt=0.001,
        decimation=10,
        num_steps=10000,
    )

    print("\nInitializing NMPC controller...")
    dynamics = HexarotorDynamics(robot_config)
    controller = NMPCController(dynamics, controller_config, robot_config)
    print("✓ Controller initialized successfully!")

    print("\nCreating MuJoCo interface...")
    mujoco_interface = FiberthexMuJoCoInterface(
        FiberthexWallScene, controller, robot_config, sim_config
    )
    print("✓ Interface created successfully!")

    reference = constant_reference(
        position=np.array([0.0, 0.0, 1.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    mujoco_interface.run_control_loop(
        reference_fn=reference,
        duration=10.0,
        render=True,
        realtime=True,
        verbose=True,
    )


def example_fiberthex_tracking():
    """Example 2: Track position step change"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Position Tracking with Fiberthex")
    print("=" * 70)

    from models.mujoco.fiberthex_wall import FiberthexWallScene

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

    sim_config = SimulationConfig(
        dt=0.001,
        decimation=10,
        num_steps=10000,
    )

    print("\nInitializing controller...")
    dynamics = HexarotorDynamics(robot_config)
    controller = NMPCController(dynamics, controller_config, robot_config)

    mujoco_interface = FiberthexMuJoCoInterface(
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

    mujoco_interface.run_control_loop(
        reference_fn=reference,
        duration=15.0,
        render=True,
        realtime=True,
        verbose=True,
    )


def example_fiberthex_circle():
    """Example 3: Follow circular trajectory"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Circular Trajectory with Fiberthex")
    print("=" * 70)

    from models.mujoco.fiberthex_wall import FiberthexWallScene

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

    sim_config = SimulationConfig(
        dt=0.001,
        decimation=10,
        num_steps=10000,
    )

    print("\nInitializing controller...")
    dynamics = HexarotorDynamics(robot_config)
    controller = NMPCController(dynamics, controller_config, robot_config)

    mujoco_interface = FiberthexMuJoCoInterface(
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

    mujoco_interface.run_control_loop(
        reference_fn=circular_trajectory,
        duration=20.0,
        render=True,
        realtime=True,
        verbose=True,
    )


def main():
    """Main function to run examples"""
    print("\n" + "=" * 70)
    print("Fiberthex MuJoCo NMPC Control Examples (Standalone Version)")
    print("=" * 70)
    print("\nSelect example to run:")
    print("1. Hover control (maintain position)")
    print("2. Position tracking (step change)")
    print("3. Circular trajectory")
    print("4. Run all examples")

    choice = input("\nEnter choice (1-4, or Enter for default=1): ").strip()

    if not choice:
        choice = "1"

    try:
        if choice == "1":
            example_fiberthex_hover()
        elif choice == "2":
            example_fiberthex_tracking()
        elif choice == "3":
            example_fiberthex_circle()
        elif choice == "4":
            print("\nRunning all examples sequentially...\n")
            example_fiberthex_hover()
            example_fiberthex_tracking()
            example_fiberthex_circle()
        else:
            print("Invalid choice. Running default (hover control)...")
            example_fiberthex_hover()

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
