# Predictive Admittance Control (PAC)

[![OS: Ubuntu](https://img.shields.io/badge/OS-ubuntu_&_macOS-blue)]()
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![simulator: MuJoCo](https://img.shields.io/badge/simulator-MuJoCo-brightgreen?link=https%3A%2F%2Fgithub.com%2Fgoogle-deepmind%2Fmujoco)](https://github.com/google-deepmind/mujoco)
[![Solver: ACADOS](https://img.shields.io/badge/solver-ACADOS-important)](https://github.com/acados/acados)


![PAC Simulation](extra/render.gif)

A standalone implementation of Predictive Admittance Control (PAC) for Aerial Physical Interaction with fully actuated hexarotors. This project uses **Acados** for real-time nonlinear model predictive control (NMPC) and **MuJoCo** for high-fidelity physics simulation.

The goal of this project is twofold:
* Provide a simple educational implementation of the method proposed in [1]
* A tutorial on how to use **Acados** and **MuJoCo**

## ✨ Features

- **Standalone Implementation**: Self-contained code for easy understanding and modification, with references to the relevant equations in the papers
- **Real-time Optimization**: Powered by Acados with efficient QP solvers
- **High-fidelity Simulation**: MuJoCo physics engine with contact dynamics
- **Comprehensive Logging/Visualization**
  - Detailed data logging of the simulations
  - Generating `.pkl` file and visualization of the logged data
  - Video rendering of the simulation
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

## Acknowledgments

This work was partially supported by Regioorgaan SIA under projects **MAESTRO-Drone** (RAAK.MKB21.026) and **AEROWIND** (RAAK.PRO06.091).

## 📚 References

This implementation is based on:
- **[1]** A. Alharbat, et al., "Predictive Admittance Control for Aerial Physical Interaction," in IEEE RA-L 2025. [[Preprint]](https://ayhamalharbat.github.io/publication/2025-4-ral/2025-4-RAL.pdf)
- **[2]** A. Alharbat, et al., "Three Fundamental Paradigms for Aerial Physical Interaction Using Nonlinear Model Predictive Control," in ICUAS 2022. [[Preprint]](https://ris.utwente.nl/ws/portalfiles/portal/286552943/Three_Fundamental_Paradigms_for_Aerial_Physical_Interaction_Using_Nonlinear_Model_Predictive_Control.pdf)

## Maintainer

[Ayham Alharbat](https://ayhamalharbat.github.io/)
