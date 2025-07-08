#!/bin/usr/python3

import math
import time
import numpy as np
import taichi as ti


@ti.data_oriented
class MPM3DSim:
    def __init__(self, cfg):
        self.dim = cfg.dim
        self.dtype       = ti.f32
        self.n_particles = cfg.n_particles
        self.n_grid      = cfg.n_grid
        self.dt          = cfg.dt
        self.dx          = cfg.dx
        self.inv_dx      = cfg.inv_dx
        self.p_vol       = cfg.p_vol
        self.p_mass      = cfg.p_mass
        self.mu_0        = cfg.mu_0
        self.lambda_0    = cfg.lambda_0
        self.floor_level = cfg.floor_level
        self.roller_radius = cfg.roller_radius
        self.soft_radius = cfg.soft_radius

        # Performance tracking
        self.performance_stats = {
            'clear_grid': 0.0,
            'p2g': 0.0,
            'grid_op': 0.0,
            'g2p': 0.0,
            'fk': 0.0
        }
        self.frame_count = 0

        # MPM fields
        n, g, d = cfg.n_particles, cfg.n_grid, cfg.dim
        self.x       = ti.Vector.field(d, dtype=ti.f32, shape=n)
        self.v       = ti.Vector.field(d, dtype=ti.f32, shape=n)
        self.F       = ti.Matrix.field(d, d, dtype=ti.f32, shape=n)
        self.C       = ti.Matrix.field(d, d, dtype=ti.f32, shape=n)
        self.J       = ti.field(dtype=ti.f32, shape=n)
        self.grid_v  = ti.Vector.field(d, dtype=ti.f32, shape=(g, g, g))
        self.grid_m  = ti.field(dtype=ti.f32, shape=(g, g, g))

        # Roller & contact
        self.roller_center   = ti.Vector.field(d, dtype=ti.f32, shape=1)
        self.roller_velocity = ti.Vector.field(d, dtype=ti.f32, shape=1)
        self.contact_force   = ti.Vector.field(d, dtype=ti.f32, shape=1)

        # Robot arm parameters
        self.L1 , self.L2 = 0.12, 0.10
        self.theta1 = np.array([0.0], dtype=np.float32)
        self.theta2 = np.array([0.0], dtype=np.float32)
        self.dtheta1 = np.zeros(1, dtype=np.float32)
        self.dtheta2 = np.zeros(1, dtype=np.float32)
        self.theta1_rest = self.theta1.copy()
        self.theta2_rest = self.theta2.copy()
        self.k   = 10    
        self.b   = 0.5

        self.I1 = self.L1**2 / 12.0
        self.I2 = self.L2**2 / 12.0

        self.base_x , self.base_y , self.base_z = 0.15, 0.150, 0.0
        self.z0 = 0.5
        self.A , self.w  = 0.2, 2.0
        self.base_z = self.z0
        self.time_t = 0.0
    
    @ti.func
    def neo_hookean(self, F):
        J = F.determinant()
        FinvT = F.inverse().transpose()
        return self.mu_0 * (F - FinvT) + self.lambda_0 * ti.log(J) * FinvT
    
    @ti.kernel
    def init(self):
        for p in range(self.n_particles):
            # More efficient random initialization
            self.x[p] = ti.Vector([ti.random() * 0.3, ti.random() * 0.3, ti.random() * 0.3])
            self.v[p] = ti.Vector.zero(ti.f32, self.dim)
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.J[p] = 1.0
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        
        # Initialize robot arm
        base = ti.Vector([self.base_x, self.base_y, self.base_z])
        j2 = base + ti.Vector([ 
            ti.sin(self.theta1[0]),
            0.0,
            -ti.cos(self.theta1[0])
        ]) * self.L1

        ee = j2 + ti.Vector([
            ti.sin(self.theta1[0] + self.theta2[0]),
            0.0,
            -ti.cos(self.theta1[0] + self.theta2[0])
        ]) * self.L2
        
        self.roller_center[0]   = ee
        self.roller_velocity[0] = ti.Vector.zero(ti.f32, self.dim)
        self.contact_force[0]   = ti.Vector.zero(ti.f32, self.dim)
        
    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))
    
    @ti.kernel
    def clear_grid(self):
        ti.loop_config(parallelize=8, block_dim=128)  # Increased block size
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
            self.grid_m[I] = 0.0

    @ti.kernel
    def p2g(self):
        ti.loop_config(parallelize=8, block_dim=256)  # Optimized block size
        for p in range(self.n_particles):
            xp = self.x[p] * self.inv_dx
            base = ti.cast(xp - 0.5, ti.i32)
            fx = xp - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            stress = (-self.dt * self.p_vol * 4 * self.inv_dx**2) * self.neo_hookean(self.F[p])
            affine = stress + self.p_mass * self.C[p]
            
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(self.dtype) - fx) * self.dx
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]

                idx = base + offset
                self.grid_v[idx] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[idx] += weight * self.p_mass

    @ti.kernel 
    def grid_op(self):
        self.contact_force[0] = ti.Vector.zero(ti.f32, self.dim)
        ti.loop_config(parallelize=8, block_dim=128)
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v_old = self.grid_v[I] / m
                v_new = v_old + self.dt * ti.Vector([0.0, 0.0, -9.8])
                pos = I.cast(ti.f32) * self.dx

                # Roller contact
                rel = pos - self.roller_center[0]
                rel_norm = rel.norm()
                if rel_norm < self.roller_radius:
                    n = rel / rel_norm  # Avoid normalization if possible
                    rv = self.roller_velocity[0]
                    v_new = (v_old - n * n.dot(v_old)) + n * n.dot(rv)
                    f_imp = m * (v_new - v_old) / self.dt
                    self.contact_force[0] += f_imp

                # Floor clamp
                if pos.z < self.floor_level + self.dx:
                    v_new = ti.Vector.zero(ti.f32, self.dim)
                
                # Wall clamp
                for d in ti.static(range(self.dim)):
                    if pos[d] < self.dx or pos[d] > 1 - self.dx:
                        v_new[d] = 0.0

                self.grid_v[I] = v_new * m

    @ti.kernel
    def g2p(self):
        ti.loop_config(parallelize=8, block_dim=256)
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

            # Floor clamp
            if self.x[p].z < self.floor_level:
                self.x[p].z = self.floor_level
                self.v[p] = ti.Vector.zero(ti.f32, self.dim)

            # Update deformation
            self.C[p] = new_C
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.dt * new_C) @ self.F[p]
            self.J[p] = self.F[p].determinant()

    def fk(self):
        self.time_t += self.dt * 10
        self.base_z = self.base_z + self.A * np.cos(self.w * self.time_t)

        Fc = self.contact_force[0].to_numpy()
        base = np.array([self.base_x, self.base_y, self.base_z], dtype=np.float32)

        j2 = base + np.array([np.sin(self.theta1[0]), 0, -np.cos(self.theta1[0])]) * self.L1
        ee_old = self.roller_center[0].to_numpy()
        ee_new = j2 + np.array([np.sin(self.theta1[0] + self.theta2[0]), 0,
                                -np.cos(self.theta1[0] + self.theta2[0])]) * self.L2
        rv = (ee_new - ee_old) / self.dt

        self.roller_center[0] = ee_new.tolist()
        self.roller_velocity[0] = rv.tolist()
        
        r1 = ee_new - base
        r2 = ee_new - j2
        tau1_c = r1[1] * Fc[0] - r1[0] * Fc[1]
        tau2_c = r2[1] * Fc[0] - r2[0] * Fc[1]
        tau1 = tau1_c - self.k * (self.theta1[0] - self.theta1_rest[0]) - self.b * self.dtheta1[0]
        tau2 = tau2_c - self.k * (self.theta2[0] - self.theta2_rest[0]) - self.b * self.dtheta2[0]
        alpha1 = tau1 / self.I1
        alpha2 = tau2 / self.I2
        
        self.dtheta1[0] += alpha1 * self.dt
        self.theta1[0] += self.dtheta1[0] * self.dt
        self.dtheta2[0] += alpha2 * self.dt
        self.theta2[0] += self.dtheta2[0] * self.dt

    def substep(self):
        start_time = time.time()
        self.clear_grid()
        self.performance_stats['clear_grid'] = time.time() - start_time
        
        start_time = time.time()
        self.p2g()
        self.performance_stats['p2g'] = time.time() - start_time
        
        start_time = time.time()
        self.grid_op()
        self.performance_stats['grid_op'] = time.time() - start_time
        
        start_time = time.time()
        self.g2p()
        self.performance_stats['g2p'] = time.time() - start_time
        
        start_time = time.time()
        self.fk()
        self.performance_stats['fk'] = time.time() - start_time

        # Print performance stats every 100 frames
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            stats = self.performance_stats
            total_time = sum(stats.values())
            print(f"Performance (ms): Clear:{stats['clear_grid']*1000:.2f} | "
                  f"P2G:{stats['p2g']*1000:.2f} | Grid:{stats['grid_op']*1000:.2f} | "
                  f"G2P:{stats['g2p']*1000:.2f} | FK:{stats['fk']*1000:.2f} | "
                  f"Total:{total_time*1000:.2f}")

    def step(self, n_substeps: int = 1):
        for _ in range(n_substeps):
            self.substep()

    def get_particle_data_for_rendering(self, max_particles=None):
        """Optimized particle data extraction for rendering"""
        pts = self.x.to_numpy()
        
        if max_particles and len(pts) > max_particles:
            # LOD: subsample particles for rendering
            indices = np.random.choice(len(pts), max_particles, replace=False)
            pts = pts[indices]
        
        return pts


# Initialize Taichi with optimizations
ti.init(arch=ti.cuda, debug=False, fast_math=True, device_memory_fraction=0.8)