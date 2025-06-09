"""
Defines the SoftBody base class and various shape subclasses.
Handles Taichi fields for particle positions, velocities, deformation gradients, etc.
"""
import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan)  # or ti.cpu for 2D

class SoftBody:
    """
    Base class for soft bodies. Manages Taichi fields and MPM kernels.
    Subclasses implement init_particles() for different shapes.
    """
    def __init__(self, n_particles, dim, bbox):
        self.n = n_particles
        self.dim = dim
        self.bmin, self.bmax = bbox  # numpy arrays of length dim
        # Taichi fields
        self.x = ti.Vector.field(dim, ti.f32, shape=self.n)  # positions
        self.v = ti.Vector.field(dim, ti.f32, shape=self.n)  # velocities
        self.F = ti.Matrix.field(dim, dim, ti.f32, shape=self.n)  # deformation
        self.C = ti.Matrix.field(dim, dim, ti.f32, shape=self.n)  # affine
        self.J = ti.field(ti.f32, shape=self.n)  # det(F)
        # Initialize particles
        self.init_particles()

    @ti.kernel
    def init_box(self):
        for i in range(self.n):
            # Uniform random sampling in axis-aligned box
            for d in ti.static(range(self.dim)):
                self.x[i][d] = ti.random() * (self.bmax[d] - self.bmin[d]) + self.bmin[d]
            self.v[i] = ti.Vector.zero(ti.f32, self.dim)
            self.F[i] = ti.Matrix.identity(ti.f32, self.dim)
            self.J[i] = 1.0
            self.C[i] = ti.Matrix.zero(ti.f32, self.dim, self.dim)

    @ti.kernel
    def init_sphere(self):
        for i in range(self.n):
            # Uniform sampling in unit sphere scaled to bbox
            while True:
                # sample in [-1,1]^dim until inside unit sphere
                pos = ti.Vector([ti.random() * 2 - 1 for _ in range(self.dim)])
                if pos.norm() <= 1.0:
                    break
            # scale and shift into bbox
            for d in ti.static(range(self.dim)):
                self.x[i][d] = (pos[d] * 0.5 + 0.5) * (self.bmax[d] - self.bmin[d]) + self.bmin[d]
            self.v[i] = ti.Vector.zero(ti.f32, self.dim)
            self.F[i] = ti.Matrix.identity(ti.f32, self.dim)
            self.J[i] = 1.0
            self.C[i] = ti.Matrix.zero(ti.f32, self.dim, self.dim)

    def init_particles(self):
        # Default to box initialization
        self.init_box()
