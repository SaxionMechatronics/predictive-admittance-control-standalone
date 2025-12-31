# Predictive Admittance Control (PAC)

![PAC Simulation](extra/render.gif)

A standalone implementation of Predictive Admittance Control (PAC) for Aerial Physical Interaction with fully actuated hexarotors. This project uses **Acados** for real-time nonlinear model predictive control (NMPC) and **MuJoCo** for high-fidelity physics simulation.

## 📚 References

This implementation is based on:
- **[1]** A. Alharbat, et al., "Predictive Admittance Control for Aerial Physical Interaction," in IEEE RA-L 2025
- **[2]** A. Alharbat, et al., "Three Fundamental Paradigms for Aerial Physical Interaction Using Nonlinear Model Predictive Control," in ICUAS 2022

## ✨ Features

- **Standalone Implementation**: Self-contained code for easy understanding and modification, with references to the papers
- **Real-time Optimization**: Powered by Acados with efficient QP solvers
- **High-fidelity Simulation**: MuJoCo physics engine with contact dynamics
- **Comprehensive Logging**: Detailed data logging of the simulations, generating `.pkl` file and visualization of the logged data, and video rendering of the simulation.
- **Extra**: a standalone implementation of a trajectory tracking NMPC in `extra/trajectory_track_mpc_standalone.py`

## 📦 Installation

**Quick Setup (One Command!):**

```bash
pixi run setup
```

**See [SETUP.md](SETUP.md) for detailed installation instructions.**

This automatically handles:
- Git submodule initialization (acados)
- Environment creation
- Acados compilation and installation
- All Python dependencies

## 🚁 Running the Example

```bash
# Direct run
pixi run python pac_standalone.py

# Or activate the environment first
pixi shell
python pac_standalone.py
```

## 📊 Outputs

The simulation generates comprehensive logs in the `logs/` directory:
- **`.pkl` files**: Complete simulation data (states, controls, predictions, config)
- **`.csv` files**: Tabular data for external analysis
- **`.png` files**: Trajectory and control plots
- **`.mp4` files**: Simulation videos (optional)

See the docstring in [pac_standalone.py](pac_standalone.py) for details on the log file format.