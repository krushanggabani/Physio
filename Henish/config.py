import math
import numpy as np
from dataclasses import dataclass

@dataclass
class Config:
    # Simulation Dimensions
    dim: int = 3
    n_particles: int = 50000
    n_grid: int = 32

    # Time Stepping
    dt: float = 1e-5

    # Neo-Hookean material properties
    E: float = 0.1e4
    nu: float = 0.2

    # Environment
    floor_level: float = 0.0
    roller_radius: float = 0.025
    soft_radius: float = 0.2

    # Arm parameters (unused in MPM kernels)
    L1: float = 0.12
    L2: float = 0.10
    k: float = 10.0
    b: float = 0.5

    # Rendering optimization parameters
    particle_point_size: float = 8.0   # Point size for point rendering
    particle_render_mode: str = "small_spheres"  # Default to round spheres instead of square points
    max_render_particles: int = 25000  # LOD limit
    camera_distance_lod: float = 1.0   # Distance threshold for LOD
    enable_frustum_culling: bool = True
    sphere_subdivisions: int = 1  # For sphere rendering modes
    sphere_radius: float = 0.005  # Radius for sphere particles

    dx = 1.0/n_grid
    inv_dx = float(n_grid)
    p_vol = pow(dx, 0.5) ** dim
    p_rho = 1.0
    p_mass = p_rho * p_vol
    mu_0 = E / (2 * (1 + nu))
    lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))