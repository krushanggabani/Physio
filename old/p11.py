import taichi as ti
import numpy as np
import pyvista as pv
import time

import vtk
# Turn off all VTK warnings
vtk.vtkObject.GlobalWarningDisplayOff()

# ----------------------------
# 1) Configuration
# ----------------------------
class Config:
    dim = 3
    n_particles = 8000
    n_grid = 32
    dt = 1e-4
    E  = 0.185e4       # Young's modulus
    nu = 0.2          # Poisson ratio
    eta = 5.0         # Viscous damping coeff (simulates viscoelasticity)
    gravity = np.array([0.0, 0.0, -9.8], dtype=np.float32)

    # Soft block extents in [0,1]^3
    block_min = np.array([0.2, 0.0, 0.2], dtype=np.float32)
    block_max = np.array([0.8, 0.4, 0.8], dtype=np.float32)

    # Rigid sphere
    sphere_radius = 0.05
    sphere_mass = 1.0
    sphere_init_center = np.array([0.5, 1.0, 0.5], dtype=np.float32)
    sphere_init_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

cfg = Config()

# ----------------------------
# 2) Initialize Taichi
# ----------------------------

ti.init(arch=ti.vulkan, debug=False, fast_math=True)

# ----------------------------
# 3) MPM Simulator
# ----------------------------
@ti.data_oriented
class MPM3D:
    def __init__(self, cfg):
        self.cfg = cfg
        d = cfg.dim
        self.dim         = cfg.dim
        self.dtype       = ti.f32
        self.dt = cfg.dt
        self.gravity = ti.Vector(cfg.gravity)
        self.eta = cfg.eta

        # material constants
        self.mu     = cfg.E / (2 * (1 + cfg.nu))
        self.lmbda = cfg.E * cfg.nu / ((1 + cfg.nu) * (1 - 2 * cfg.nu))

        # grid / particle geometry
        self.n_grid = cfg.n_grid
        self.dx     = 1.0 / cfg.n_grid
        self.inv_dx = float(cfg.n_grid)
        self.p_vol  = (self.dx * 0.5) ** d
        self.p_mass = self.p_vol  # assume density=1

        # Taichi fields
        self.x       = ti.Vector.field(d, dtype=ti.f32, shape=cfg.n_particles)
        self.v       = ti.Vector.field(d, dtype=ti.f32, shape=cfg.n_particles)
        self.F       = ti.Matrix.field(d, d, dtype=ti.f32, shape=cfg.n_particles)
        self.C       = ti.Matrix.field(d, d, dtype=ti.f32, shape=cfg.n_particles)
        self.J       = ti.field(dtype=ti.f32, shape=cfg.n_particles)
        self.grid_v  = ti.Vector.field(d, dtype=ti.f32, shape=(cfg.n_grid,)*d)
        self.grid_m  = ti.field(dtype=ti.f32, shape=(cfg.n_grid,)*d)

        # sphere (rigid) fields
        self.sphere_center   = ti.Vector.field(d, dtype=ti.f32, shape=1)
        self.sphere_velocity = ti.Vector.field(d, dtype=ti.f32, shape=1)
        self.contact_force   = ti.Vector.field(d, dtype=ti.f32, shape=1)

    @ti.kernel
    def init(self):
        # 3D block of particles
        for i in range(self.cfg.n_particles):        

            self.x[i] = [ti.random() * 0.3 , ti.random() * 0.3 , ti.random() * 0.2]

            self.v[i] = ti.Vector.zero(ti.f32, self.cfg.dim)
            self.F[i] = ti.Matrix.identity(ti.f32, self.cfg.dim)
            self.C[i] = ti.Matrix.zero(ti.f32, self.cfg.dim, self.cfg.dim)
            self.J[i] = 1.0

        # sphere initial state
        self.sphere_center[0]   = ti.Vector(self.cfg.sphere_init_center)
        self.sphere_velocity[0] = ti.Vector(self.cfg.sphere_init_velocity)
        self.contact_force[0]   = ti.Vector.zero(ti.f32, self.cfg.dim)

    @ti.func
    def neo_hookean(self, F):
        J = F.determinant()
        FinvT = F.inverse().transpose()
        return self.mu * (F - FinvT) + self.lmbda * ti.log(J) * FinvT

    @ti.kernel
    def clear_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(ti.f32, self.cfg.dim)
            self.grid_m[I] = 0.0

    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))
    
    @ti.kernel
    def p2g(self):
        for p in range(self.x.shape[0]):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base.cast(ti.f32)
            # B-spline weights
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2
            ]
            # Elastic stress → affine
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * self.neo_hookean(self.F[p])
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
        # reset contact
        self.contact_force[0] = ti.Vector.zero(ti.f32, self.cfg.dim)
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v_old = self.grid_v[I] / m
                # gravity + viscous damping
                v_new = v_old + self.dt * self.gravity - self.eta * v_old
                pos = I.cast(ti.f32) * self.dx

                # sphere contact
                rel = pos - self.sphere_center[0]
                if rel.norm() < self.cfg.sphere_radius:
                    n = rel.normalized()
                    # simple elastic reflection
                    snapped = v_old - 2 * (v_old.dot(n)) * n
                    v_new = snapped
                    # accumulate impulse for sphere update
                    self.contact_force[0] += m * (v_new - v_old) / self.dt

                # floor bounce for block
                if pos.y < 0:
                    v_new.y = -v_new.y * 0.5

                self.grid_v[I] = v_new * m

    @ti.kernel
    def g2p(self):
        for p in range(self.x.shape[0]):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base.cast(ti.f32)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2
            ]
            new_v = ti.Vector.zero(ti.f32, self.cfg.dim)
            new_C = ti.Matrix.zero(ti.f32, self.cfg.dim, self.cfg.dim)
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
            self.v[p] = new_v
            self.x[p] += self.dt * new_v

            # Floor clamp
            if self.x[p].z < 0.0:
                self.x[p].z = 0.0
                self.v[p]    = ti.Vector.zero(ti.f32, self.dim)

            self.C[p] = new_C
            self.F[p] = (ti.Matrix.identity(ti.f32, self.cfg.dim) + self.dt * new_C) @ self.F[p]
            self.J[p] = self.F[p].determinant()

    def step(self):
        # One MPM substep
        self.clear_grid()
        self.p2g()
        self.grid_op()
        self.g2p()

        # Now update sphere in Python
        f = self.contact_force[0].to_numpy()
        v = self.sphere_velocity[0].to_numpy()
        # acceleration = gravity + contact_force/mass
        a = cfg.gravity + f / cfg.sphere_mass
        v = v + cfg.dt * a
        pos = self.sphere_center[0].to_numpy() + cfg.dt * v
        # bounce on floor
        if pos[1] < cfg.sphere_radius:
            pos[1] = cfg.sphere_radius
            v[1] = -v[1] * 0.5
        self.sphere_velocity[0] = ti.Vector(v)
        self.sphere_center[0]   = ti.Vector(pos)

# ----------------------------
# 4) Live PyVista Renderer
# ----------------------------
class Renderer3D:
    def __init__(self, sim: MPM3D):
        self.sim = sim
        # set up plotter
        self.plotter = pv.Plotter(window_size=(800, 600))
        self.plotter.add_axes()
        self.plotter.set_background("white")
        # Floor
        self.floor = pv.Cube(center=(0, 0, -0.005), x_length=2, y_length=2, z_length=0.01)
        self.plotter.add_mesh(self.floor, color='lightgray')

        # Soft mesh placeholder
        self.pcd = pv.PolyData(sim.x.to_numpy())
        self.cloud_actor = self.plotter.add_mesh(self.pcd, color='hotpink', point_size=5, render_points_as_spheres=True)

        # self.soft_mesh = pv.Sphere(radius=0.01).triangulate()
        # self.soft_actor = self.plotter.add_mesh(self.soft_mesh, color='pink', opacity=1.0)

        # Sphere placeholder
        c0 = sim.sphere_center[0].to_numpy()
        self.bouncer = pv.Sphere(center=c0, radius=cfg.sphere_radius)
        self.bouncer_actor = self.plotter.add_mesh(self.bouncer, color='red')
        # Time text
        self.text_actor = self.plotter.add_text("t = 0.0000 s", position='upper_left', font_size=14)
        self.plotter.show(auto_close=False, interactive_update=True)

    def render(self, frame):
        # reconstruct soft-body surface
        pts = self.sim.x.to_numpy()
        # cloud = pv.PolyData(pts)
        self.pcd.points = pts
        self.cloud_actor.mapper.Update()

        # try:
        #     vol   = cloud.delaunay_3d(alpha=0.1)
        #     shell = vol.extract_geometry().smooth(n_iter=5)
        #     self.soft_mesh.deep_copy(shell)
        # except:
        #     pass
        # update sphere pos
        center = self.sim.sphere_center[0].to_numpy()
        new_ball = pv.Sphere(center=center, radius=cfg.sphere_radius)
        self.bouncer.deep_copy(new_ball)
        # update time
        # t = frame * cfg.dt
        # self.text_actor.SetText(f"t = {t:.4f} s")
        # redraw
        self.plotter.update()

# ----------------------------
# 5) Main Loop
# ----------------------------
if __name__ == "__main__":
    sim = MPM3D(cfg)
    sim.init()
    renderer = Renderer3D(sim)

    for i in range(2000):
        sim.step()
        renderer.render(i)
        # slow down if too fast
        time.sleep(0.005)
