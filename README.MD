# Physio


A modular, high-performance toolkit for **Modeling and Simulation of Soft Tissue** using the Material Point Method (MPM), designed to study and optimize robot–tissue interaction dynamics in surgical robotics and haptic applications.

---

## 🌟 Features

- **MPM-based soft tissue modeling**: Realistic viscoelastic and hyperelastic material models
- **Configurable primitives**: Support for multiple geometric shapes (boxes, spheres, meshes)
- **Collision & contact handling**: Robust grid- and particle-based collision schemes
- **Adaptive time-stepping**: Automatic adjustment for stability under high deformation
- **Sensor–robot coupling**: Integrate real-time force/position sensor data for haptic feedback
- **Multi-platform**: Compatible with Windows, Linux, and macOS
- **Easy-to-extend**: Plugin architecture for custom material laws and control strategies


## 📦 Installation

### ❗ Prerequisites

- **Python 3.8+**
- **Git** to clone the repository
- (Optional) NVIDIA GPU + CUDA for accelerated Taichi backends

### 1. Clone the repository

```bash
git clone https://github.com/<username>/physio.git
cd physio
```

### 2. Create and activate a virtual environment

#### Windows (PowerShell)
```powershell
python -m venv physioenv     # Create venv in `physioenv`
.\physioenv\Scripts\Activate.ps1  # Activate environment
```

#### Linux / macOS (bash)
```bash
python3 -m venv physioenv
source physioenv/bin/activate
```

### 3. Install package and dependencies

```bash
pip install -e .            # Installs in editable mode
pip install -r requirements.txt  # Install additional requirements
```

### 4. (Optional) GPU acceleration

If you have an NVIDIA GPU and CUDA installed, enable the Taichi CUDA backend in `physio/config.yaml`:
```yaml
backend: cuda
```

---

## 🚀 Quick Start

1. **Setup simulation**: Modify your config file in `configs/` (e.g., `box_collision.yaml`).
2. **Run demo**:
   ```bash
   physio run configs/box_collision.yaml
   ```
3. **Visualize results**: Outputs stored in `output/`; use `--visualize` flag to view in real time:
   ```bash
   physio run configs/box_collision.yaml --visualize
   ```


## 📂 Repository Structure

```
physio/
├── configs/            # Example YAML configurations
├── physio/             # Core simulator package
│   ├── mpm.py          # MPM solver implementation
│   ├── primitives.py   # Geometric primitives & SDFs
│   ├── collision.py    # Contact routines
│   └── utils.py        # Logging, visualization helpers
├── examples/           # Sample scripts & Jupyter notebooks
├── tests/              # Unit and integration tests
├── requirements.txt    # Additional dependencies
└── setup.py            # Package installation script
```

---

## 🎓 Usage Examples

- **Basic box–tissue interaction**:
  ```bash
  physio run configs/box_collision.yaml
  ```
- **Surgical tool insertion** (sphere primitive + custom stiffness model):
  ```bash
  physio run configs/sphere_insert.yaml
  ```
- **Real-time haptic feedback**:
  ```bash
  physio run configs/haptic_feedback.yaml --force-sensor /dev/ttyUSB0
  ```


## 🛠️ Configuration

All simulation parameters live in a YAML file under `configs/`. Key fields include:

| Field                 | Description                                          | Example           |
|-----------------------|------------------------------------------------------|-------------------|
| `dim`                 | Simulation dimension (2 or 3)                        | `2`               |
| `n_particles`         | Total number of MPM particles                        | `10000`           |
| `material`            | Material model (`neo_hookean`, `corotated`, etc.)   | `neo_hookean`     |
| `yield_stress`        | Plastic yield threshold (Pa)                         | `1e4`             |
| `time_step`           | Simulation time step (s)                             | `1e-4`            |
| `backend`             | Taichi backend (`cpu`, `cuda`)                       | `cpu`             |

Refer to `configs/example.yaml` for the full schema.

---

## 📚 References

If you use this package in your research, please cite:

```bibtex
@article{Gabani2025Physio,
  title={Physio: MPM-based Soft Tissue Simulation for Robot–Tissue Interaction},
  author={Gabani, Krushang},
  journal={Journal of Soft Robotics Simulations},
  year={2025},
}
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/xyz`)
3. Commit your changes (`git commit -am 'Add xyz'`)
4. Push to branch (`git push origin feature/xyz`)
5. Open a Pull Request

Please ensure all tests pass and code is linted (`flake8`).

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

Questions or feedback? Open an issue or email me at <krushang@buffalo.edu>.

Happy simulating! 🚀

