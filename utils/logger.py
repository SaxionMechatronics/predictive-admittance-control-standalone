"""
CSV logging module for NMPC experiments
Provides modular and extensible logging functionality for experiment results
"""

import csv
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class ExperimentLogger:
    """Logger for saving experiment results to CSV files"""

    def __init__(self, robot_config, controller_config, sim_config):
        """
        Initialize experiment logger

        Args:
            robot_config: Robot configuration object
            controller_config: Controller configuration object
            sim_config: Simulation configuration object
        """
        self.robot_config = robot_config
        self.controller_config = controller_config
        self.sim_config = sim_config

        # Determine state dimensions
        self.nx = robot_config.nx
        self.nu = robot_config.nu
        self.num_rotors = robot_config.num_rotors

    def log_results(
        self,
        results: dict[str, Any],
        experiment_name: str = "experiment",
        reference_trajectory: Callable | None = None,
    ) -> str:
        """
        Save simulation results to CSV file

        Args:
            results: Dictionary containing simulation data
            experiment_name: Name prefix for the CSV file
            reference_trajectory: Optional reference trajectory function

        Returns:
            Path to the saved CSV file
        """
        # Create log directory if it doesn't exist
        log_dir = Path(self.sim_config.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}.csv"
        filepath = log_dir / filename

        # Write CSV file
        with open(filepath, "w", newline="") as csvfile:
            self._write_header(csvfile, experiment_name)
            self._write_data(csvfile, results, reference_trajectory)

        return str(filepath)

    def _write_header(self, csvfile, experiment_name: str):
        """Write metadata header and column names"""
        writer = csv.writer(csvfile)

        # Metadata section
        writer.writerow(["# Experiment Metadata"])
        writer.writerow(["# Name", experiment_name])
        writer.writerow(["# Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(["# Robot Type", self.robot_config.name])
        writer.writerow(["# Mass (kg)", self.robot_config.mass])
        writer.writerow(["# Number of Rotors", self.num_rotors])
        writer.writerow(["#"])
        writer.writerow(["# Controller Configuration"])
        writer.writerow(["# Timestep (s)", self.controller_config.dt])
        writer.writerow(["# Horizon", self.controller_config.horizon])
        writer.writerow([
            "# Position Weight",
            str(list(self.controller_config.position_weight)),
        ])
        writer.writerow(
            ["# Quaternion Weight", self.controller_config.quaternion_weight]
        )
        writer.writerow([
            "# Velocity Weight",
            str(list(self.controller_config.velocity_weight)),
        ])
        writer.writerow(["# Control Weight", self.controller_config.control_weight])
        writer.writerow(["#"])
        writer.writerow(["# Simulation Configuration"])
        writer.writerow(["# Simulation dt (s)", self.sim_config.dt])
        writer.writerow(["# Decimation", self.sim_config.decimation])
        writer.writerow(["# Number of Steps", self.sim_config.num_steps])
        writer.writerow(["#"])

        # Column headers
        headers = ["time"]

        # State columns
        headers.extend(["pos_x", "pos_y", "pos_z"])
        headers.extend(["quat_w", "quat_x", "quat_y", "quat_z"])
        headers.extend(["vel_x", "vel_y", "vel_z"])
        headers.extend(["omega_x", "omega_y", "omega_z"])

        # Rotor forces
        for i in range(self.num_rotors):
            headers.append(f"F{i + 1}")

        # Control inputs
        for i in range(self.num_rotors):
            headers.append(f"dF{i + 1}")

        # Solver performance
        headers.append("solve_time")

        # Reference trajectory columns (if logging enabled)
        if self.sim_config.log_reference:
            headers.extend(["ref_pos_x", "ref_pos_y", "ref_pos_z"])
            headers.extend(["ref_quat_w", "ref_quat_x", "ref_quat_y", "ref_quat_z"])

        writer.writerow(headers)

    def _write_data(
        self,
        csvfile,
        results: dict[str, Any],
        reference_trajectory: Callable | None,
    ):
        """Write time-series data rows"""
        writer = csv.writer(csvfile)

        states = results["states"]
        controls = results["controls"]
        times = results["times"]
        solve_times = results["solve_times"]

        # Number of data rows (use controls length since it's one less than states)
        num_rows = len(controls)

        for i in range(num_rows):
            row = self._format_state_row(
                time=times[i],
                state=states[i],
                control=controls[i],
                solve_time=solve_times[i],
                reference_trajectory=reference_trajectory,
            )
            writer.writerow(row)

    def _format_state_row(
        self,
        time: float,
        state: np.ndarray,
        control: np.ndarray,
        solve_time: float,
        reference_trajectory: Callable | None,
    ) -> list:
        """Format a single data row for CSV"""
        row = [time]

        # Extract state components
        # States: [pos(3), quat(4), vel(3), omega(3), forces(num_rotors)]
        pos = state[0:3]
        quat = state[3:7]
        vel = state[7:10]
        omega = state[10:13]
        forces = state[13 : 13 + self.num_rotors]

        # Add state data
        row.extend(pos)
        row.extend(quat)
        row.extend(vel)
        row.extend(omega)
        row.extend(forces)

        # Add control inputs
        row.extend(control)

        # Add solve time
        row.append(solve_time)

        # Add reference trajectory if enabled
        if self.sim_config.log_reference and reference_trajectory is not None:
            ref = reference_trajectory(time)
            ref_pos = ref.get("position", np.zeros(3))
            ref_quat = ref.get("quaternion", np.array([1.0, 0.0, 0.0, 0.0]))
            row.extend(ref_pos)
            row.extend(ref_quat)

        return row


def load_experiment_log(filepath: str) -> dict[str, Any]:
    """
    Load experiment data from CSV file

    Args:
        filepath: Path to the CSV log file

    Returns:
        Dictionary containing experiment data and metadata
    """
    metadata = {}
    data = []
    headers = []

    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)

        in_metadata = True
        for row in reader:
            if not row:
                continue

            # Parse metadata
            if row[0].startswith("#"):
                if len(row) >= 2 and not row[1].startswith("Experiment"):
                    # Extract metadata key-value pairs
                    key = row[0].replace("#", "").strip()
                    if len(row) > 1:
                        value = row[1].strip()
                        if key and value:
                            metadata[key] = value
                continue

            # Parse headers
            if in_metadata:
                headers = row
                in_metadata = False
                continue

            # Parse data rows
            data.append([float(x) if x else 0.0 for x in row])

    # Convert to numpy array
    data_array = np.array(data)

    # Create dictionary with named columns
    result = {"metadata": metadata}
    for i, header in enumerate(headers):
        result[header] = data_array[:, i]

    return result
