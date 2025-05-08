import taichi as ti
import numpy as np


@ti.data_oriented
class MPM:
    def __init__(self,cfg):
        
        dim = self.dim = cfg.dim
        dtype = self.dtype = cfg.dtype

        # self._yield_stress = cfg.yield_stress
        # self.ground_friction = cfg.ground_friction
        self.default_gravity = cfg.gravity
        # self.n_primitive = len(primitives)

        quality = 1
        if self.dim == 3:
            quality = quality * 0.5
        n_particles = self.n_particles = cfg.n_particles
        n_grid = self.n_grid = int(128 * quality)

        self.dx, self.inv_dx = 1 / n_grid, float(n_grid)
        self.dt = cfg.dt
        self.p_vol, self.p_rho = (self.dx * 0.5) ** 2, 1
        self.p_mass = self.p_vol * self.p_rho


        # material
        self.material_model = cfg.material_model
        E, nu = cfg.E, cfg.nu
        self._mu, self._lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
        

        self.mu = ti.field(dtype=self.dtype, shape=n_particles)
        self.lam = ti.field(dtype=self.dtype, shape=n_particles)
        self.yield_stress = ti.field(dtype=self.dtype, shape=n_particles)

        
        self.x = ti.Vector.field(dim, dtype=dtype, shape=n_particles)  # position
        self.v = ti.Vector.field(dim, dtype=dtype, shape=n_particles)  # velocity
        self.C = ti.Matrix.field(dim, dim, dtype=dtype, shape=n_particles)  # affine velocity field
        self.F = ti.Matrix.field(dim, dim, dtype=dtype, shape=n_particles)  # deformation gradient

        # Fields: grid
        self.grid_v = ti.Vector.field(2, float, (n_grid, n_grid))  # node momentum/velocity
        self.grid_m = ti.field(float, (n_grid, n_grid))           # node mass
        self.grid_A = ti.field(int, (n_grid, n_grid))             # active under rigid box
        
        self.initialize()

    @ti.kernel
    def initialize(self):
        for i in range(self.n_particles):
            self.x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.2, ti.random() * 0.2]
            self.v[i] = ti.Vector.zero(self.dtype, self.dim)
            self.F[i] = ti.Matrix.identity(self.dtype, self.dim) #ti.Matrix([[1, 0], [0, 1]])
            self.C[i] = ti.Matrix.zero(self.dtype, self.dim, self.dim)

        # self.gravity[None] = self.default_gravity
        # # self.yield_stress.fill(self._yield_stress)
        # self.mu.fill(self._mu)
        # self.lam.fill(self._lam)

    # @ti.kernel
    # def substep(self):

    #     # t[None] += dt


    
    
        