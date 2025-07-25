import math
from dataclasses import dataclass

import numpy as np
import taichi as ti
import pyvista as pv
import trimesh
import time

np.infty = np.inf





# Initialize Taichi
ti.init(arch=ti.vulkan, debug=False, fast_math=True)

@dataclass
class Config:
    # Simulation dimensions
    dim: int = 3
    n_particles: int = 9000
    n_grid: int = 32

    # Time stepping
    dt: float = 1e-4

    # Neo‐Hookean material properties
    E: float = 0.185e4
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

# Build derived constants
cfg = Config()
cfg.dx       = 1.0 / cfg.n_grid
cfg.inv_dx   = float(cfg.n_grid)
cfg.p_vol    = (cfg.dx * 0.5) ** cfg.dim
cfg.p_rho    = 1.0
cfg.p_mass   = cfg.p_rho * cfg.p_vol
cfg.mu_0     = cfg.E / (2 * (1 + cfg.nu))
cfg.lambda_0 = cfg.E * cfg.nu / ((1 + cfg.nu) * (1 - 2 * cfg.nu))



@ti.data_oriented
class MPM3DSim:
    def __init__(self, cfg):
        # Copy primitives into self for kernel use
        self.dim         = cfg.dim
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

        self.base_x , self.base_y , self.base_z = 0.15,0.150,0.0
        self.z0 = 0.5
        self.A , self.w  = 0.2,2.0
        self.base_z = self.z0
        self.time_t = 0.0


    @ti.func
    def neo_hookean(self, F):
        J = F.determinant()
        FinvT = F.inverse().transpose()

        # stress = self.mu_0*(F @ F.transpose()) + ti.Matrix.identity(self.dtype, self.dim) * (self.lambda_0 * ti.log(J) - self.mu_0)
        return self.mu_0 * (F - FinvT) + self.lambda_0 * ti.log(J) * FinvT

    @ti.kernel
    def init(self):
        # Initialize particles in a sphere and set roller at origin
        for p in range(self.n_particles):
            u, v, w = ti.random(), ti.random(), ti.random()
            r = cfg.soft_radius * u ** (1/3)
            theta = math.pi * v
            phi = ti.acos(2*w - 1)
            self.x[p] = ti.Vector([
                r * ti.sin(phi) * ti.cos(theta),
                r * ti.cos(phi),
                r * ti.sin(phi) * ti.sin(theta)
            ])
            self.x[p] = [ti.random() * 0.3 , ti.random() * 0.3 , ti.random() * 0.2]

            # i = p % res_x
            # j = (p // res_x) % res_y
            # k = p // (res_x * res_y)
            # # place in the center of each cell
            # self.x[p] = ti.Vector([
            #     (i + 0.5) * cell_size_x,
            #     (j + 0.5) * cell_size_y,
            #     (k + 0.5) * cell_size_z,
            # ])

            self.v[p] = ti.Vector.zero(ti.f32, self.dim)
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.J[p] = 1.0
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)

        # Roller initial state
        base = ti.Vector([self.base_x, self.base_y, self.base_z])
        j2 = base + ti.Vector([ 
            ti.sin(self.theta1[0]),    # Δx
            0.0,                   # Δy
            -ti.cos(self.theta1[0])     # Δz
        ]) * self.L1

        # end effector: same pattern but with θ1+θ2
        ee = j2 + ti.Vector([
            ti.sin(self.theta1[0] + self.theta2[0]),
            0.0,
            -ti.cos(self.theta1[0] +self.theta2[0])
        ]) * self.L2
        self.roller_center[0]   = ee
        self.roller_velocity[0] = ti.Vector.zero(ti.f32, self.dim)
        self.contact_force[0]   = ti.Vector.zero(ti.f32, self.dim)


    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    @ti.kernel
    def clear_grid(self):
        # Reset grid
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

            stress = ti.Matrix.zero(self.dtype, self.dim, self.dim)
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

        # Grid → apply forces & detect
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v_old = self.grid_v[I] / m
                v_new = v_old + self.dt * ti.Vector([0.0, 0.0,-9.8])
                pos = I.cast(ti.f32) * self.dx

                # Roller contact
                rel = pos - self.roller_center[0]
                if rel.norm() < self.roller_radius:
                    n = rel.normalized()
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
        for p in range(self.n_particles):
            xp = self.x[p] * self.inv_dx
            base = ti.cast(xp - 0.5, ti.i32)
            fx = xp - base.cast(ti.f32)

            # w0 = 0.5 * (1.5 - fx)**2
            # w1 = 0.75 - (fx - 1.0)**2
            # w2 = 0.5 * (fx - 0.5)**2

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
                self.v[p]    = ti.Vector.zero(ti.f32, self.dim)

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
        self.time_t += self.dt*15
        self.base_z = self.z0 + self.A * np.cos(self.w * self.time_t)

        # print(self.base_z)
        Fc = self.contact_force[0].to_numpy()
        base = np.array([self.base_x, self.base_y, self.base_z], dtype=np.float32)

        j2 = base + np.array([np.sin(self.theta1[0]), 0, -np.cos(self.theta1[0])]) * self.L1
        ee_old = self.roller_center[0].to_numpy()
        ee_new = j2 + np.array([np.sin(self.theta1[0] + self.theta2[0]), 0 ,
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

        self.clear_grid()
        self.p2g()
        self.grid_op()
        self.g2p()
        self.fk()

        

    def step(self, n_substeps: int = 1):
        for _ in range(n_substeps):
            self.substep()




class Renderer3D:
    def __init__(self,sim: MPM3DSim):
        self.sim = sim
        self.plotter = pv.Plotter(window_size=(800, 600))
        self.plotter.add_axes()
        self.plotter.set_background("white")

        # Floor
        self.floor = pv.Cube(center=(0, 0, -0.005), x_length=2, y_length=2, z_length=0.01)
        self.plotter.add_mesh(self.floor, color='lightgray')

        # # Point cloud initialization
        # self.pcd = pv.PolyData(sim.x.to_numpy())
        # self.cloud_actor = self.plotter.add_mesh(self.pcd, color='hotpink', point_size=5, render_points_as_spheres=True)

        self.solid_mesh = pv.Sphere(radius=0.01).triangulate()
        self.mesh_actor = self.plotter.add_mesh(self.solid_mesh, color='hotpink', opacity=1.0)


        # Robot arm placeholders (cylinders and roller sphere)
        self.link1 = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.005, height=1.0)
        self.link2 = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=0.005, height=1.0)
        self.roller = pv.Sphere(center=(0, 0, 0), radius=sim.roller_radius)

        self.link1_actor = self.plotter.add_mesh(self.link1, color='blue')
        self.link2_actor = self.plotter.add_mesh(self.link2, color='blue')
        self.roller_actor = self.plotter.add_mesh(self.roller, color='red')

        # Time text
        self.text_actor = self.plotter.add_text("t = 0.0000s", position='upper_left', font_size=14)

        self.plotter.show(auto_close=False, interactive_update=True)

    def update_cylinder(self, cyl: pv.PolyData, start, end):
        center = (start + end) / 2
        direction = end - start
        height = np.linalg.norm(direction)
        if height < 1e-6:
            direction = np.array([0, 0, 1])
            height = 1e-6

        new_cyl = pv.Cylinder(center=center, direction=direction, radius=0.005, height=height)
        cyl.deep_copy(new_cyl)  # <- this replaces the old geometry



    def render(self, step_i=0):
        # pts = self.sim.x.to_numpy()

        # # Update point cloud
        # self.pcd.points = pts
        # self.cloud_actor.mapper.Update()

        pts = self.sim.x.to_numpy()
        cloud = pv.PolyData(pts).clean()

        try:
            volume = cloud.delaunay_3d(alpha=1)
            shell = volume.extract_geometry()
            shell = shell.smooth(n_iter=10)
            self.solid_mesh.deep_copy(shell)
        except Exception as e:
            print(f"[Warning] Failed surface reconstruction: {e}")

        # # Remove previous robot links
        # for act in self.link_actors:
        #     self.plotter.remove_actor(act)
        # self.link_actors = []

        # Robot geometry
        base_pt = np.array([self.sim.base_x, self.sim.base_y, self.sim.base_z], dtype=np.float32)
        j2 = base_pt + np.array([np.sin(self.sim.theta1[0]), 0, -np.cos(self.sim.theta1[0])]) * self.sim.L1
        ee = self.sim.roller_center[0].to_numpy()

        # # Add links and roller as cylinders/spheres
        # self.link_actors.append(self.plotter.add_mesh(pv.Cylinder(center=(base_pt + j2) / 2, direction=j2 - base_pt, radius=0.005, height=np.linalg.norm(j2 - base_pt)), color='blue'))
        # self.link_actors.append(self.plotter.add_mesh(pv.Cylinder(center=(j2 + ee) / 2, direction=ee - j2, radius=0.005, height=np.linalg.norm(ee - j2)), color='blue'))
        # self.link_actors.append(self.plotter.add_mesh(pv.Sphere(center=ee, radius=self.sim.roller_radius), color='red'))

         # Update robot links
        self.update_cylinder(self.link1, base_pt, j2)
        self.update_cylinder(self.link2, j2, ee)

        # Update roller
        # self.roller.center = ee

        # Time display
        t = step_i *1e-4
        # self.text_actor.SetInput(f"t = {t:.4f}s")
        self.text_actor.SetText(2, f"t = {t:.4f}s")



        self.plotter.update()


        







if __name__ == "__main__":
    sim = MPM3DSim(cfg)
    sim.init()
    renderer = Renderer3D(sim)
    renderer.render(step_i=1)
    time.sleep(1)
    print("Init")

    for i in range(1000):
        sim.step(n_substeps=25)

        # ti.profiler_print()

        renderer.render(step_i=i)
        # time.sleep(1)
        # print("updated")
