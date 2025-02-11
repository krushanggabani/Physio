# Soft Body Massage Physics Simulator (Physio)
HILS LAB

**Author**: Krushang Gabani

## Setup

This project is designed for Ubuntu with Python 3.7.

1. **Create and activate the virtual environment:**

    ```bash
    python3 -m venv physioenv
    source physioenv/bin/activate
    ```

2. **Install the required packages:**

    ```bash
    pip install -e .
    ```

Your environment is now ready!

<!-- ## FAQ

### 1. CUDA Installation
   If the graphics card driver is incompatible with your OS, skip its installation. Instead, install `torch` and `cuda-python` within the virtual environment:

    ```bash
    pip install cuda-python torch
    ```

### 2. Taichi Function Error
   If encountering module issues in Taichi, try these solutions:

    - Replace `ext_arr()` with `types.ndarray()`
    - Replace `complex_kernel` with `ad.grad_replaced()`
    - Replace `complex_kernel_grad` with `ad.grad_for()` -->