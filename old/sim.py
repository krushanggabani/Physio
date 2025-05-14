import taichi as ti
import numpy as np

@ti.data_oriented
class MPMSimulator:

    def __init__(self,dt,cfg):
        
        # Simulation parameters (tunable)
        self.dt  = dt
        self.dim = cfg.dim

        quality = cfg.quality
        n_particles = self.n_particles = cfg.n_particles
        n_grid = self.n_grid = int(128 * quality)


        dx = 1.0 / n_grid
        inv_dx = float(n_grid)
        dt = 1e-4 / quality