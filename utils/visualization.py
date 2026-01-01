"""
Visualization utilities for simulation results
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.spatial.transform import Rotation as R


def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to 3x3 rotation matrix"""
    # Normalize quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if norm > 0:
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    else:
        qw, qx, qy, qz = 1, 0, 0, 0

    # Rotation matrix
    return np.array([
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qx * qw),
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx**2 + qy**2),
        ],
    ])


class ResultsVisualizer:
    """Visualizer for simulation results"""

    def __init__(self, results: dict):
        """
        Initialize visualizer
        Args:
            results: Dictionary from simulator with keys:
                - states: (N, nx) array
                - controls: (N-1, nu) array
                - times: (N,) array
        """
        self.results = results
        self.states = results["states"]
        self.controls = results["controls"]
        self.times = results["times"]
        self.solve_times = results.get("solve_times", np.zeros(len(self.controls)))

        # Extract state components
        self.positions = self.states[:, 0:3]
        self.quaternions = self.states[:, 3:7]
        self.velocities = self.states[:, 7:10]
        self.angular_velocities = self.states[:, 10:13]
        self.forces = self.states[:, 13:19]

        # extract impedance position and velocity if available
        self.impedance_position = results.get(
            "impedance_position", np.zeros((len(self.times), 3))
        )
        self.impedance_velocity = results.get(
            "impedance_velocity", np.zeros((len(self.times), 3))
        )

        # Extract contact forces if available
        self.contact_forces = results.get(
            "contact_forces", np.zeros((len(self.times), 3))
        )

        # Compute Euler angles from quaternions
        r = R.from_quat(self.quaternions[:, [1, 2, 3, 0]])  # scipy uses [qx,qy,qz,qw]
        self.euler_angles = r.as_euler("xyz", degrees=True)

        # Extract references if available
        self.ref_positions = None
        self.ref_quaternions = None
        self.ref_euler_angles = None

        # Try to get references from results (simulator output)
        if "ref_positions" in results:
            self.ref_positions = results["ref_positions"]
        # Try to get references from CSV columns
        elif (
            "ref_pos_x" in results and "ref_pos_y" in results and "ref_pos_z" in results
        ):
            self.ref_positions = np.column_stack([
                results["ref_pos_x"],
                results["ref_pos_y"],
                results["ref_pos_z"],
            ])

        # Try to get reference quaternions
        if "ref_quaternions" in results:
            self.ref_quaternions = results["ref_quaternions"]
        elif "ref_quat_w" in results:
            self.ref_quaternions = np.column_stack([
                results["ref_quat_w"],
                results["ref_quat_x"],
                results["ref_quat_y"],
                results["ref_quat_z"],
            ])

        # Compute reference Euler angles if quaternions are available
        if self.ref_quaternions is not None:
            r_ref = R.from_quat(self.ref_quaternions[:, [1, 2, 3, 0]])
            self.ref_euler_angles = r_ref.as_euler("xyz", degrees=True)

    def plot_all(self, save_path: str | None = None):
        """Create comprehensive plot of all states and controls"""

        if save_path:
            matplotlib.use("agg")

        fig = plt.figure(figsize=(20, 15))

        # 3D trajectory plot
        ax1 = fig.add_subplot(3, 4, 1, projection="3d")
        ax1.plot(
            self.positions[:, 0],
            self.positions[:, 1],
            self.positions[:, 2],
            "b-",
            linewidth=2,
        )
        ax1.scatter(
            self.positions[0, 0],
            self.positions[0, 1],
            self.positions[0, 2],
            c="g",
            marker="o",
            s=100,
            label="Start",
        )
        ax1.scatter(
            self.positions[-1, 0],
            self.positions[-1, 1],
            self.positions[-1, 2],
            c="r",
            marker="x",
            s=100,
            label="End",
        )
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_zlabel("Z [m]")
        ax1.set_title("3D Trajectory")
        ax1.legend()
        ax1.grid(True)

        # Position vs time
        ax2 = fig.add_subplot(3, 4, 2)
        ax2.plot(self.times, self.positions[:, 0], "r-", label="x")
        if self.ref_positions is not None:
            ax2.plot(
                self.times,
                self.ref_positions[:, 0],
                "r--",
                alpha=0.7,
                label="x ref",
            )

        ax2.plot(self.times, self.positions[:, 1], "g-", label="y")
        if self.ref_positions is not None:
            ax2.plot(
                self.times,
                self.ref_positions[:, 1],
                "g--",
                alpha=0.7,
                label="y ref",
            )

        ax2.plot(self.times, self.positions[:, 2], "b-", label="z")
        if self.ref_positions is not None:
            ax2.plot(
                self.times,
                self.ref_positions[:, 2],
                "b--",
                alpha=0.7,
                label="z ref",
            )
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Position [m]")
        ax2.set_title("Position")
        ax2.grid(True)
        ax2.legend()

        # Quaternion vs time
        ax3 = fig.add_subplot(3, 4, 3)
        ax3.plot(self.times, self.quaternions[:, 0], "r-", label="qw")
        ax3.plot(self.times, self.quaternions[:, 1], "g-", label="qx")
        ax3.plot(self.times, self.quaternions[:, 2], "b-", label="qy")
        ax3.plot(self.times, self.quaternions[:, 3], "m-", label="qz")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Quaternion")
        ax3.set_title("Quaternion")
        ax3.grid(True)
        ax3.legend()

        # Euler angles vs time
        ax4 = fig.add_subplot(3, 4, 4)
        ax4.plot(self.times, self.euler_angles[:, 0], "r-", label="Roll")
        if self.ref_euler_angles is not None:
            ax4.plot(
                self.times,
                self.ref_euler_angles[:, 0],
                "r--",
                alpha=0.7,
                label="Roll ref",
            )

        ax4.plot(self.times, self.euler_angles[:, 1], "g-", label="Pitch")
        if self.ref_euler_angles is not None:
            ax4.plot(
                self.times,
                self.ref_euler_angles[:, 1],
                "g--",
                alpha=0.7,
                label="Pitch ref",
            )

        ax4.plot(self.times, self.euler_angles[:, 2], "b-", label="Yaw")
        if self.ref_euler_angles is not None:
            ax4.plot(
                self.times,
                self.ref_euler_angles[:, 2],
                "b--",
                alpha=0.7,
                label="Yaw ref",
            )
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Angle [deg]")
        ax4.set_title("Euler Angles")
        ax4.grid(True)
        ax4.legend()

        # Velocity vs time
        ax5 = fig.add_subplot(3, 4, 5)
        ax5.plot(self.times, self.velocities[:, 0], "r-", label="vx")
        ax5.plot(self.times, self.velocities[:, 1], "g-", label="vy")
        ax5.plot(self.times, self.velocities[:, 2], "b-", label="vz")
        ax5.set_xlabel("Time [s]")
        ax5.set_ylabel("Velocity [m/s]")
        ax5.set_title("Linear Velocity")
        ax5.grid(True)
        ax5.legend()

        # Angular velocity vs time
        ax6 = fig.add_subplot(3, 4, 6)
        ax6.plot(self.times, self.angular_velocities[:, 0], "r-", label="ωx")
        ax6.plot(self.times, self.angular_velocities[:, 1], "g-", label="ωy")
        ax6.plot(self.times, self.angular_velocities[:, 2], "b-", label="ωz")
        ax6.set_xlabel("Time [s]")
        ax6.set_ylabel("Angular Velocity [rad/s]")
        ax6.set_title("Angular Velocity")
        ax6.grid(True)
        ax6.legend()

        # Rotor Forces vs time
        ax7 = fig.add_subplot(3, 4, 7)
        for i in range(self.forces.shape[1]):
            ax7.plot(self.times, self.forces[:, i], label=f"F{i + 1}")
        ax7.set_xlabel("Time [s]")
        ax7.set_ylabel("Force [N]")
        ax7.set_title("Rotor Forces")
        ax7.grid(True)
        ax7.legend()

        # Control inputs vs time
        ax8 = fig.add_subplot(3, 4, 8)
        for i in range(self.controls.shape[1]):
            min_len = min(self.controls.shape[0], self.times.shape[0])
            self.controls = self.controls[:min_len]
            self.times = self.times[:min_len]
            ax8.plot(self.times, self.controls[:, i], label=f"dF{i + 1}")
        ax8.set_xlabel("Time [s]")
        ax8.set_ylabel("Control [N/s]")
        ax8.set_title("Control Inputs")
        ax8.grid(True)
        ax8.legend()

        # Contact Forces vs time
        ax9 = fig.add_subplot(3, 4, 9)
        ax9.plot(self.times, self.contact_forces[:, 0], "r-", label="Fx")
        ax9.plot(self.times, self.contact_forces[:, 1], "g-", label="Fy")
        ax9.plot(self.times, self.contact_forces[:, 2], "b-", label="Fz")
        ax9.set_xlabel("Time [s]")
        ax9.set_ylabel("Force [N]")
        ax9.set_title("Contact Forces")
        ax9.grid(True)
        ax9.legend()

        # impedance position vs time
        ax10 = fig.add_subplot(3, 4, 10)
        ax10.plot(self.times, self.impedance_position[:, 0], "r-", label="px")
        ax10.plot(self.times, self.impedance_position[:, 1], "g-", label="py")
        ax10.plot(self.times, self.impedance_position[:, 2], "b-", label="pz")
        ax10.set_xlabel("Time [s]")
        ax10.set_ylabel("Position [m]")
        ax10.set_title("Impedance Position")
        ax10.grid(True)
        ax10.legend()

        # impedance velocity vs time
        ax11 = fig.add_subplot(3, 4, 11)
        ax11.plot(self.times, self.impedance_velocity[:, 0], "r-", label="vx")
        ax11.plot(self.times, self.impedance_velocity[:, 1], "g-", label="vy")
        ax11.plot(self.times, self.impedance_velocity[:, 2], "b-", label="vz")
        ax11.set_xlabel("Time [s]")
        ax11.set_ylabel("Velocity [m/s]")
        ax11.set_title("Impedance Velocity")
        ax11.grid(True)
        ax11.legend()

        # Computation time plot
        ax12 = fig.add_subplot(3, 4, 12)
        solve_times_ms = self.solve_times * 1000  # Convert to milliseconds

        ax12.plot(
            self.times,
            solve_times_ms,
            "b-",
            alpha=0.6,
            linewidth=1,
            label="Solve time",
        )
        ax12.axhline(
            y=np.mean(solve_times_ms),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(solve_times_ms):.2f} ms",
        )
        ax12.axhline(
            y=np.max(solve_times_ms),
            color="orange",
            linestyle=":",
            linewidth=1.5,
            label=f"Max: {np.max(solve_times_ms):.2f} ms",
        )

        # Add statistics text box
        # stats_text = f"Min: {np.min(solve_times_ms):.2f} ms\n" \
        #             f"Mean: {np.mean(solve_times_ms):.2f} ms\n" \
        #             f"Max: {np.max(solve_times_ms):.2f} ms\n" \
        #             f"Std: {np.std(solve_times_ms):.2f} ms"
        # ax9.text(0.02, 0.98, stats_text, transform=ax9.transAxes,
        #         fontsize=10, verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax12.set_xlabel("Time [s]")
        ax12.set_ylabel("Computation Time [ms]")
        ax12.set_title("NMPC Computation Time")
        ax12.grid(True, alpha=0.3)
        ax12.legend(loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
            plt.close(fig)  # Clean up when saving to file
        else:
            plt.show()

    def animate_3d(self, frame_skip: int = 10, save_path: str | None = None):
        """
        Create 3D animation of drone with orientation frame
        Args:
            frame_skip: Show every nth frame
            save_path: Path to save animation (e.g., 'animation.mp4')
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Set axis limits
        pos_range = np.max(np.abs(self.positions))
        ax.set_xlim([-pos_range, pos_range])
        ax.set_ylim([-pos_range, pos_range])
        ax.set_zlim([0, 2 * pos_range])
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Hexarotor Animation")
        ax.grid(True)

        # Plot full trajectory
        ax.plot(
            self.positions[:, 0],
            self.positions[:, 1],
            self.positions[:, 2],
            "b--",
            alpha=0.3,
            linewidth=1,
            label="Trajectory",
        )

        # Initialize lines for body frame
        (line_x,) = ax.plot([], [], [], "r-", linewidth=4, label="X-axis")
        (line_y,) = ax.plot([], [], [], "g-", linewidth=4, label="Y-axis")
        (line_z,) = ax.plot([], [], [], "b-", linewidth=4, label="Z-axis")
        (trajectory_line,) = ax.plot([], [], [], "k-", linewidth=3, label="Path")
        (drone_center,) = ax.plot([], [], [], "ko", markersize=12)
        time_text = ax.text2D(
            0.05,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.legend(loc="upper right")

        frame_size = 0.5

        def animate(frame_idx):
            frame_idx = frame_idx * frame_skip
            if frame_idx >= len(self.positions):
                frame_idx = len(self.positions) - 1

            # Current position and orientation
            pos = self.positions[frame_idx]
            quat = self.quaternions[frame_idx]

            # Get rotation matrix
            R_mat = quat_to_rotation_matrix(*quat)

            # Body frame axes
            x_axis = np.array([frame_size, 0, 0])
            y_axis = np.array([0, frame_size, 0])
            z_axis = np.array([0, 0, frame_size])

            # Transform to world frame
            x_world = R_mat @ x_axis
            y_world = R_mat @ y_axis
            z_world = R_mat @ z_axis

            # Create line segments
            x_line = np.array([pos, pos + x_world])
            y_line = np.array([pos, pos + y_world])
            z_line = np.array([pos, pos + z_world])

            # Update visualization
            line_x.set_data_3d(x_line[:, 0], x_line[:, 1], x_line[:, 2])
            line_y.set_data_3d(y_line[:, 0], y_line[:, 1], y_line[:, 2])
            line_z.set_data_3d(z_line[:, 0], z_line[:, 1], z_line[:, 2])

            # Update trajectory
            traj = self.positions[: frame_idx + 1]
            trajectory_line.set_data_3d(traj[:, 0], traj[:, 1], traj[:, 2])

            # Update center
            drone_center.set_data_3d([pos[0]], [pos[1]], [pos[2]])

            # Update time
            time_text.set_text(f"Time: {self.times[frame_idx]:.2f}s")

            return (
                line_x,
                line_y,
                line_z,
                trajectory_line,
                drone_center,
                time_text,
            )

        num_frames = len(self.positions) // frame_skip
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=num_frames,
            interval=50,
            blit=False,
            repeat=True,
        )

        if save_path:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=20, metadata=dict(artist="Hexarotor"), bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")

        plt.show()
