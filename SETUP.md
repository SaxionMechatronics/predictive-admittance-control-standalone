# Setup Instructions

This document provides detailed instructions for setting up the Predictive Admittance Control environment using Pixi.

---

## 🚀 **Pixi Setup**

[Pixi](https://pixi.sh) is a modern, fast, and cross-platform package manager.

### Prerequisites
- Git

### Installation Steps

1. **Install Pixi** (one-time setup):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

   Or on macOS with Homebrew:
   ```bash
   brew install pixi
   ```

2. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd predictive-admittance-control
   ```

3. **Run setup** (this single command does everything):
   ```bash
   pixi run setup
   ```

   This will:
   - Initialize git submodules (acados)
   - Create the environment
   - Build and install acados
   - Install all Python dependencies

4. **Run the example**:
   ```bash
   pixi run python pac_standalone.py
   ```

   Or enter the shell and run commands manually:
   ```bash
   pixi shell
   python pac_standalone.py
   ```

### Why Pixi?
- ✅ **Single command setup**: Everything is configured in `pixi.toml`
- ✅ **Fast**: Better caching and parallel downloads
- ✅ **Reproducible**: Lock file ensures everyone gets the same environment
- ✅ **Isolated**: Each project has its own environment
- ✅ **Modern tooling**: Better dependency resolution

---

## 🧹 Cleaning Up

To clean build artifacts:
```bash
pixi run clean
```

To remove the entire environment:
```bash
rm -rf .pixi
```

---

## 🔧 Troubleshooting

### Acados build fails
- Ensure you have CMake 3.20+ installed
- Check that you have a C/C++ compiler (gcc/g++)
- Try cleaning and rebuilding:
  ```bash
  pixi run clean
  pixi run setup
  ```

### Environment variables not set
- They're automatically set when you run `pixi shell` or `pixi run`

### Python packages not found
- Run `pixi install` to ensure all dependencies are installed

---
