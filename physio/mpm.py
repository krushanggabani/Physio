import taichi as ti
import numpy as np
from .arm import PassiveArm

@ti.data_oriented
class MPMSimulator:
    def __init__(self,
                 arch=ti.vulkan,
                 arm = PassiveArm(),
                 n_particles=20000,
                 n_grid=128,
                 dt=2e-4,
                 E=5e4,
                 nu=0.2):
        # Simulation basics
        self.dim = 2
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.dx = 1.0 / n_grid
        self.inv_dx = float(n_grid)
        self.dt = dt

        # Material (neo-Hookean)
        p_rho = 1.0
        self.p_vol = (self.dx * 0.5)**2
        self.p_mass = p_rho * self.p_vol
        self.mu_0 = E / (2 * (1 + nu))
        self.lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

        # Boundaries
        self.floor_level = 0.0
        self.floor_friction = 0.4

        # Roller/contact fields
        self.roller_radius = 0.025
        self.roller_center = ti.Vector.field(self.dim, ti.f32, shape=1)
        self.roller_velocity = ti.Vector.field(self.dim, ti.f32, shape=1)
        self.contact_force = ti.Vector.field(self.dim, ti.f32, shape=1)

        # MPM fields
        self.x = ti.Vector.field(self.dim, ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(self.dim, ti.f32, shape=self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, ti.f32, shape=self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, ti.f32, shape=self.n_particles)
        self.J = ti.field(ti.f32, shape=self.n_particles)
        self.grid_v = ti.Vector.field(self.dim, ti.f32,
                                     shape=(self.n_grid, self.n_grid))
        self.grid_m = ti.field(ti.f32, shape=(self.n_grid, self.n_grid))

        self.ee = arm.ee
        self.init_mpm()

    @ti.func
    def neo_hookean_stress(self, F_i):
        J = F_i.determinant()
        FinvT = F_i.inverse().transpose()
        return self.mu_0 * (F_i - FinvT) + self.lambda_0 * ti.log(J) * FinvT

    @ti.kernel
    def init_mpm(self):
        # Clear grid
        # for I in ti.grouped(self.grid_m):
        #     self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
        #     self.grid_m[I] = 0.0
        # Initialize particles
        for p in range(self.n_particles):
            u = ti.random()
            r = 0.2 * ti.sqrt(u)
            theta = ti.random() *  3.141592653589793
            self.x[p] = ti.Vector([0.5 + r * ti.cos(theta),
                                   self.floor_level + r * ti.sin(theta)])
            self.v[p] = ti.Vector.zero(ti.f32, self.dim)
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.J[p] = 1.0
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        # Roller initial
        # self.roller_center[0] = ti.Vector([0.5, 0.4])
        
        self.roller_center[0] = self.ee
        self.roller_velocity[0] = ti.Vector.zero(ti.f32, self.dim)
        self.contact_force[0] = ti.Vector.zero(ti.f32, self.dim)

    @ti.kernel
    def p2g(self):
        # Particle to grid
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
            self.grid_m[I] = 0.0
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx)**2,
                 0.75 - (fx - 1.0)**2,
                 0.5 * (fx - 0.5)**2]
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx**2) * \
                     self.neo_hookean_stress(self.F[p])
            affine = stress + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offs = ti.Vector([i, j])
                dpos = (offs.cast(ti.f32) - fx) * self.dx
                wt = w[i].x * w[j].y
                self.grid_v[base + offs] += wt * (
                    self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offs] += wt * self.p_mass

    @ti.kernel
    def apply_grid_forces_and_detect(self):
        # Apply forces & detect roller/floor collisions
        self.contact_force[0] = ti.Vector.zero(ti.f32, self.dim)
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v_old = self.grid_v[I] / m
                v_new = v_old + self.dt * ti.Vector([0.0, -9.8])
                pos = I.cast(ti.f32) * self.dx
                # Roller
                rel = pos - self.roller_center[0]
                if rel.norm() < self.roller_radius:
                    rv = self.roller_velocity[0]
                    n = rel.normalized()
                    v_norm = n * n.dot(rv)
                    v_tan = v_old - n * n.dot(v_old)
                    v_new = v_tan + v_norm
                    delta = v_new - v_old
                    f_imp = m * delta / self.dt
                    self.contact_force[0] += f_imp
                # Floor
                if pos.y < self.floor_level + self.dx:
                    if v_new.y < 0: v_new.y = 0
                    v_new.x = 0
                # Walls
                if pos.x < self.dx or pos.x > 1 - self.dx:
                    v_new.x = 0
                self.grid_v[I] = v_new * m

    @ti.kernel
    def g2p(self):
        # Grid to particle
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx)**2,
                 0.75 - (fx - 1.0)**2,
                 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offs = ti.Vector([i, j])
                dpos = (offs.cast(ti.f32) - fx) * self.dx
                wt = w[i].x * w[j].y
                gv = self.grid_v[base + offs] / self.grid_m[base + offs]
                new_v += wt * gv
                new_C += 4 * self.inv_dx * wt * gv.outer_product(dpos)
            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v
            # Boundary clamping
            if self.x[p].y < self.floor_level:
                self.x[p].y = self.floor_level
                self.v[p].y = 0
                self.v[p].x = 0
            if self.x[p].x < self.dx:
                self.x[p].x = self.dx
                self.v[p].x = 0
            if self.x[p].x > 1 - self.dx:
                self.x[p].x = 1 - self.dx
                self.v[p].x = 0
            if self.x[p].y > 1 - self.dx:
                self.x[p].y = 1 - self.dx
                self.v[p].y = 0
            # Update deformation
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) +
                         self.dt * self.C[p]) @ self.F[p]
            self.J[p] = self.F[p].determinant()