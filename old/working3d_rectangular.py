import taichi as ti
import numpy as np
np.infty = np.inf
import pyrender
import trimesh
import time

# ─── Init Taichi ──────────────────────────────────────────────────────────
ti.init(arch=ti.vulkan)

# ─── Constants ─────────────────────────────────────────────────────────────
PI = 3.141592653589793

# ─── Simulation parameters ────────────────────────────────────────────────
dim         = 3
n_particles = 50000
n_grid      = 64
x_min, x_max = 0.0, 1.0

dx     = (x_max - x_min) / n_grid
inv_dx = float(n_grid) / (x_max - x_min)
dt     = 2e-4

# ─── Particle properties ──────────────────────────────────────────────────
p_rho  = 1.0
p_vol  = (dx * 0.5)**dim
p_mass = p_vol * p_rho

# ─── Material (Neo-Hookean) ──────────────────────────────────────────────
E        = 5e3
nu       = 0.2
mu_0     = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# ─── Roller (end-effector) ────────────────────────────────────────────────
roller_radius = 0.01
roller_center = ti.Vector.field(dim, dtype=ti.f32, shape=())

# ─── Roller state & contact ───────────────────────────────────────────────
state          = ti.field(ti.i32, shape=())   # 0=descend,1=roll
contact_height = ti.field(ti.f32, shape=())
contact_force  = ti.field(ti.f32, shape=())

v_desc = 0.5  # m/s
v_roll = 0.2  # m/s
v_target = ti.Vector([0.,0.,0.])

# ─── Floor parameters ────────────────────────────────────────────────────
floor_level    = 0.0
floor_friction = 0.4

# ─── Soft-body box params ─────────────────────────────────────────────────
box_min = ti.Vector([0.0, 0.0, 0.0])
box_max = ti.Vector([0.2, 0.2, 0.2])

# ─── Taichi fields ────────────────────────────────────────────────────────
x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
J = ti.field(dtype=ti.f32, shape=n_particles)

grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

# ─── Neo-Hookean stress ───────────────────────────────────────────────────
@ti.func
def neo_hookean_stress(F_i):
    J_det = F_i.determinant()
    FinvT = F_i.inverse().transpose()
    return mu_0 * (F_i - FinvT) + lambda_0 * ti.log(J_det) * FinvT

# ─── Initialize box soft-body & roller ────────────────────────────────────
@ti.kernel
def init_particles():
    for p in range(n_particles):
        x[p] = ti.Vector([
            ti.random() * (box_max.x - box_min.x) + box_min.x,
            ti.random() * (box_max.y - box_min.y) + box_min.y,
            ti.random() * (box_max.z - box_min.z) + box_min.z
        ])
        v[p] = ti.Vector.zero(ti.f32, dim)
        F[p] = ti.Matrix.identity(ti.f32, dim)
        J[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)
    state[None]          = 0
    contact_height[None] = 0.0
    contact_force[None]  = 0.0
    roller_center[None]  = ti.Vector([0.1, 0.1, 0.4])  # start above box

# ─── Particle→Grid (P2G) ─────────────────────────────────────────────────
@ti.kernel
def p2g():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, dim)
        grid_m[I] = 0.0
    for p in range(n_particles):
        Xp = (x[p] - x_min) * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * neo_hookean_stress(F[p])
        affine = stress + p_mass * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k], ti.f32)
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y * w[k].z
            idx = base + ti.Vector([i, j, k])
            grid_v[idx] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[idx] += weight * p_mass

# ─── Grid Forces & Contact ────────────────────────────────────────────────
@ti.kernel
def apply_grid_forces_and_detect():
    contact_force[None] = 0.0
    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 0:
            v_old = grid_v[I] / m
            # gravity downwards in -Z
            v_new = v_old + dt * ti.Vector([0.0, 0.0, -9.8])
            pos = I.cast(ti.f32) * dx + ti.Vector([x_min, 0.0, x_min])
            # Roller contact if within radius
            rel = pos - roller_center[None]
            print(rel)
            if rel.norm() < roller_radius:
                # choose target: descent in -Z, then roll in +X
                if state[None] == 0:
                    v_target = ti.Vector([0.0, 0.0, -v_desc])
                else:
                    v_target = ti.Vector([v_roll, 0.0, 0.0])
                n = rel.normalized()
                v_norm = n * n.dot(v_target)
                v_tan = v_old - n * n.dot(v_old)
                v_new = v_tan + v_norm
                f_imp = m * (v_new - v_old) / dt
                contact_force[None] += f_imp.norm()
            # floor at Z=floor_level
            if pos.z < floor_level + dx:
                if v_new.z < 0:
                    v_new.z = 0
                # friction in XY plane
                v_new.x *= (1 - floor_friction)
                v_new.y *= (1 - floor_friction)
            # side walls in X and Y
            if pos.x < x_min + dx: v_new.x = 0
            if pos.x > x_max - dx: v_new.x = 0
            if pos.y < 0.0 + dx:   v_new.y = 0
            if pos.y > 1.0 - dx:   v_new.y = 0
            grid_v[I] = v_new * m
    # switch phase once threshold
    if state[None] == 0 and contact_force[None] >= 0.20:
        state[None] = 1
        contact_height[None] = roller_center[None].z

# ─── Grid→Particle (G2P) ─────────────────────────────────────────────────
@ti.kernel
def g2p():
    for p in range(n_particles):
        Xp = (x[p] - x_min) * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)
        for i, j, k in ti.static(ti.ndrange(3,3,3)):
            idx = base + ti.Vector([i,j,k])
            dpos = (ti.Vector([i,j,k],ti.f32)-fx)*dx
            weight = w[i].x*w[j].y*w[k].z
            g_v = grid_v[idx]/grid_m[idx]
            new_v += weight*g_v
            new_C += 4*inv_dx*weight*g_v.outer_product(dpos)
        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v
        # particle floor clamp
        if x[p].z < floor_level:
            x[p].z = floor_level; v[p].z = 0
            x[p].x *= (1 - floor_friction)
            x[p].y *= (1 - floor_friction)
        # side walls X and Y
        if x[p].x < x_min+dx: x[p].x = x_min+dx; v[p].x = 0
        if x[p].x > x_max-dx: x[p].x = x_max-dx; v[p].x = 0
        if x[p].y < 0.0+dx:   x[p].y = 0.0+dx;   v[p].y = 0
        if x[p].y > 1.0-dx:   x[p].y = 1.0-dx;   v[p].y = 0
        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]
        J[p] = F[p].determinant()

# ─── Roller Update Function ──────────────────────────────────────────────
def update_roller_position():
    c = roller_center[None]
    if state[None] == 0:
        c.z -= v_desc * dt
    else:
        c.x += v_roll * dt
        c.z = contact_height[None]
    roller_center[None] = c


# ─── PYRENDER SETUP ──────────────────────────────────────────────────────
# ─── VISUALIZER CLASS USING REFERENCE SETUP ───────────────────────────────
class Visualizer:
    def __init__(self, link_size):
        # Camera setup
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi/6, aspectRatio=1.0)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.camera_pose[:3, 3] = [0, -2.7, 2]
        # rotation around X-axis by 60°
        angle = np.deg2rad(60)
        self.camera_pose[:3, :3] = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]], dtype=np.float32)
        
        # Light setup
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self.light_pose = np.eye(4, dtype=np.float32)
        angle_l = np.deg2rad(-30)
        self.light_pose[:3, :3] = np.array([[1,0,0],[0,np.cos(angle_l),-np.sin(angle_l)],[0,np.sin(angle_l),np.cos(angle_l)]], dtype=np.float32)
        
        # Floor mesh
        floor_size = 2.0
        vs = np.array([
            [ floor_size,  floor_size, 0],
            [-floor_size,  floor_size, 0],
            [-floor_size, -floor_size, 0],
            [ floor_size, -floor_size, 0]
        ], dtype=np.float32)
        fs = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        floor_tm = trimesh.Trimesh(vertices=vs, faces=fs)
        self.floor_mesh = pyrender.Mesh.from_trimesh(floor_tm, smooth=False)
        
        # Link mesh
        box_tm = trimesh.creation.uv_sphere(radius=roller_radius)
        red_rgba = [1.0, 0.0, 0.0, 1.0]
        red_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=red_rgba,
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        self.box_mesh = pyrender.Mesh.from_trimesh(box_tm, material=red_mat, smooth=True)
        self.box_pose = np.eye(4, dtype=np.float32)
        self.box_pose[:3, 3] = [-0.5, 0.0, 2]
        
        # Scene
        self.scene = pyrender.Scene()
        self.floor_node = self.scene.add(self.floor_mesh)
        self.cam_node   = self.scene.add(self.camera, pose=self.camera_pose)
        self.light_node = self.scene.add(self.light,   pose=self.light_pose)
        self.box_node   = None
        self.cloud_node = None
        self.viewer     = None

    def view(self, points: np.ndarray, colors: np.ndarray, box_pose: np.ndarray):
        if self.viewer is None:
            self.viewer = pyrender.Viewer(
                self.scene,
                use_raymond_lighting=True,
                run_in_thread=True,
                fullscreen=True,
                window_title="Robot Tissue Interaction",
                viewport_size=(1920,1080)
            )
            time.sleep(0.001)
        cloud = pyrender.Mesh.from_points(points, colors=colors)
        self.box_pose[:3,3] = box_pose
        with self.viewer.render_lock:
            if self.cloud_node is not None:
                self.scene.remove_node(self.cloud_node)
                self.scene.remove_node(self.box_node)
            self.cloud_node = self.scene.add(cloud)
            self.box_node   = self.scene.add(self.box_mesh, pose=self.box_pose)
        time.sleep(1e-3)


# ─── INIT & RUN ──────────────────────────────────────────────────────────
init_particles()
visualizer = Visualizer(link_size=[0.1,0.02,0.02])  # example link size
while True:
    for _ in range(10):
        p2g(); apply_grid_forces_and_detect(); update_roller_position(); g2p()
    pts = x.to_numpy()
    colors = np.tile([0.4,0.7,1.0,1.0], (n_particles,1))
    box_pose = roller_center[None].to_numpy()
    visualizer.view(pts, colors, box_pose)