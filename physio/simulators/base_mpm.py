# --------------------------------------------------------------------------------
# Copyright (c) 2025 Krushang Gabani
# All rights reserved.
#
# BASE_MPM: Taichi-based Material Point Method solver
#
# Author: Krushang Gabani
# Date: July 7, 2025
# --------------------------------------------------------------------------------


import taichi as ti
import numpy as np
import math

@ti.data_oriented
class BASE_MPM:
    def __init__(self,cfg):

        self.cfg         = cfg
        # simulation parameters
        self.dim         = cfg.dim
        self.dtype       = ti.f32
        self.n_particles = cfg.n_particles
        self.n_grid      = cfg.n_grid
        self.dt          = cfg.dt
        
        # derived quantities
        self.dx          = 1.0 / self.n_grid
        self.inv_dx      = float(self.n_grid)
        self.p_vol       = (self.dx * 0.5) ** 3
        self.p_mass      = cfg.p_rho * self.p_vol

        # gravity and floor
        self.gravity     = cfg.gravity       # tuple (gx,gy,gz)
        self.floor_level = cfg.floor_level   # z-coordinate of floor
        self.friction    = cfg.friction      # friction coefficient
        self.g = ti.Vector([self.gravity[0], self.gravity[1], self.gravity[2]])

        # material (Neo-Hookean) parameters
        E, nu = cfg.youngs_modulus, cfg.poisson_ratio
        self.mu_0     = E / (2 * (1 + nu))
        self.lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

        # MPM fields
        n, g, d = cfg.n_particles, cfg.n_grid, cfg.dim
        self.x       = ti.Vector.field(d, dtype=ti.f32, shape=n)
        self.v       = ti.Vector.field(d, dtype=ti.f32, shape=n)
        self.F       = ti.Matrix.field(d, d, dtype=ti.f32, shape=n)
        self.C       = ti.Matrix.field(d, d, dtype=ti.f32, shape=n)
        self.J       = ti.field(dtype=ti.f32, shape=n)
        self.grid_v  = ti.Vector.field(d, dtype=ti.f32, shape=(g, g, g))
        self.grid_m  = ti.field(dtype=ti.f32, shape=(g, g, g))

        # initialize particles
        self._init_particles()


    @ti.kernel
    def _init_particles(self):
        # initialize shape and state
        if self.cfg.shape == "box":
            for p in range(self.n_particles):
                self.x[p] = ti.Vector([ti.random() * 0.3,
                                       ti.random() * 0.3,
                                       ti.random() * 0.2 +0.5])
        else:  # sphere
            for p in range(self.n_particles):
                u = ti.random()
                v = ti.random()
                w = ti.random()
                # inverse‐CDF radius
                r = self.cfg.size * u ** (1/3)
                # angles
                theta = math.pi * v            # constant * Expr is OK
                phi   = ti.acos(2 * w - 1)         # use taichi-acos
                self.x[p] = ti.Vector([
                    r * ti.sin(phi) * ti.cos(theta)+0.5,
                    r * ti.cos(phi) +0.5,
                    r * ti.sin(phi) * ti.sin(theta) 
                ])

        # initialize the rest of the MPM state
        for p in range(self.n_particles):
            self.v[p] = ti.Vector.zero(self.dtype, self.dim)
            self.F[p] = ti.Matrix.identity(self.dtype, self.dim)
            self.J[p] = 1.0
            self.C[p] = ti.Matrix.zero(self.dtype, self.dim, self.dim)

    @ti.func
    def neo_hookean(self, F):
        J = F.determinant()
        FinvT = F.inverse().transpose()
        return self.mu_0 * (F - FinvT) + self.lambda_0 * ti.log(J) * FinvT

    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    @ti.kernel
    def clear_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
            self.grid_m[I] = 0.0

    @ti.kernel
    def p2g(self):
        for p in range(self.n_particles):
            xp = self.x[p] * self.inv_dx


            base = ti.cast(xp - 0.5, ti.i32)
            fx = xp - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            # print(f"set xp={xp} base={base}")
            
            stress = ti.Matrix.zero(self.dtype, self.dim, self.dim)
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx**2) * self.neo_hookean(self.F[p])
            affine = stress + self.p_mass * self.C[p]
            
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(self.dtype) - fx) * self.dx
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                idx = base + offset

                # if 0 <= idx[0] < self.n_grid and 0 <= idx[1] < self.n_grid and 0 <= idx[2] < self.n_grid:
                #     pass
                # else:
                #     print(f"set idx={idx} base={base}")

                self.grid_v[idx] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[idx] += weight * self.p_mass


    @ti.kernel 
    def grid_op(self):
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v_old = self.grid_v[I] / m
                v_new = v_old + self.dt * self.g
                pos = I.cast(self.dtype) * self.dx

                # impulse-based floor collision
                if pos[2] < self.floor_level + self.dx:
                    n = ti.Vector([0.0, 0.0, 1.0])
                    # normal and tangential components
                    v_n = v_new.dot(n)
                    v_t = v_new - v_n * n
                    if v_n < 0:
                        # normal impulse magnitude
                        jn = -(1 + 0.5) * v_n * m
                        # tangential impulse vector
                        jt = -m * v_t
                        jt_mag = jt.norm()
                        jt_max = self.friction * jn
                        # clamp tangential impulse
                        if jt_mag > jt_max:
                            jt = jt * (jt_max / jt_mag)
                        # apply impulses
                        p_new = m * v_old + jn * n + jt
                        v_new = p_new / m
                    
                # Floor clamp
                if pos.z < self.floor_level + self.dx:
                    v_new = ti.Vector.zero(ti.f32, self.dim)

                # optional wall clamps
                for d in ti.static(range(self.dim)):
                    if pos[d] < self.dx or pos[d] > 1 - self.dx:
                        v_new[d] = 0.0
                self.grid_v[I] = v_new * m


    @ti.kernel
    def g2p(self):
        for p in range(self.n_particles):
            xp = self.x[p] * self.inv_dx
            base = ti.cast(xp - 0.5, ti.i32)
            fx = xp - base.cast(ti.f32)


            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)

            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(self.dtype) - fx
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                idx = base + offset
                m_val = self.grid_m[idx]
                if m_val > 0:
                    gv = self.grid_v[idx] / m_val
                    new_v += weight * gv
                    new_C += 4 * self.inv_dx * weight * gv.outer_product(dpos)

            # Update particle
            self.v[p] = new_v
            self.x[p] += self.dt * new_v

            # particle-floor collision: enforce non-penetration
            if self.x[p].z < self.floor_level:
                self.x[p].z = self.floor_level
                self.v[p]    = ti.Vector.zero(ti.f32, self.dim)

                # self.v[p][2] = 0.0
                # # tangential friction at particle-level
                # self.v[p][0] *= (1 - self.friction)
                # self.v[p][1] *= (1 - self.friction)

            # Wall clamp
            # for d in ti.static(range(self.dim)):
            #     self.x[p][d] = ti.min(ti.max(self.x[p][d], self.dx), 1 - self.dx)
            #     # zero velocity at the boundary
            #     if self.x[p][d] <= self.dx or self.x[p][d] >= 1 - self.dx:
            #         self.v[p][d] = 0.0

            # Update deformation
            self.C[p] = new_C
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.dt * new_C) @ self.F[p]
            self.J[p] = self.F[p].determinant()
    
    def fk(self):
        pass


    def step(self,n_substeps:int = 1):
        
        for _ in range(n_substeps):
            self.clear_grid()
            self.p2g()
            self.grid_op()
            self.g2p()
            self.fk()

    
