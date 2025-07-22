# --------------------------------------------------------------------------------
# Copyright (c) 2025 Krushang Gabani
# All rights reserved.
#
# Configuration module for the Material Point Method (MPM) simulation,
# physics parameters, and rendering settings using PyRender.
#
# Author: Krushang Gabani
# Date: July 7, 2025
# --------------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    """
    Configuration for the MPM simulation, including physics, solver,
    and rendering settings. Tweak these parameters to control simulation
    fidelity, performance, and visual output.
    """

    # -------------------------- Simulation Dimensions --------------------------
    dim: int = 3  # Spatial dimensionality: 2 for planar, 3 for full 3D
    n_particles: int = 9_000  # Total number of particles in the simulation
    n_grid: int = 32  # Grid resolution per axis (grid dimension = n_grid^dim)
    max_steps: int = 1_000  # Maximum number of time steps before simulation ends
    dtype: str = "float32"  # Numeric precision: "float32" or "float64"

    # ---------------------------- Time Stepping -------------------------------
    dt: float = 1e-4  # Fixed time step size (in seconds)
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.8)  # Gravity vector (m/s^2)

    # ------------------------ Material Properties -----------------------------
    shape:str = "sphere"
    size: float = 0.2
    p_rho: float = 1.0  # Particle density (mass per unit volume)
    youngs_modulus: float = 0.185e4  # Young's modulus (Pa) for Neo-Hookean model
    poisson_ratio: float = 0.2  # Poisson's ratio (dimensionless)

    # ------------------------------ Floor ------------------------------------
    floor_level: float = 0.0  # Z-height of the floor plane
    floor_size: float = 2.0  # Half-size of the floor square (in meters)
    friction: float = 1.5  # Coefficient of friction between particles & floor

    # --------------------------- Engine Settings -----------------------------
    engine: str = "BASE_MPM"  # MPM solver variant identifier
    collision_model: str = "BASE_MODEL"  # Collision handling strategy

    # --------------------------- Rendering Settings --------------------------
    render_model: str = "PyRender"  # Rendering backend (e.g., "PyRender", "OpenGL")
    solid_mode: str = "solid"       # solid, mesh
    camera_pose: Tuple[float, float, float] = (0.5, -1.5, 2.0)
    # Camera position in world coordinates (X, Y, Z)
    camera_rotation: Tuple[float, float] = (0.0, 45.0)
    # Camera orientation: (yaw in degrees, pitch in degrees)

    light_pose: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Light position in world coordinates (X, Y, Z)
    light_rotation: Tuple[float, float] = (0.0, -30.0)
    # Light orientation: (yaw in degrees, pitch in degrees)
