import taichi as ti
import numpy as np
from yacs.config import CfgNode as CN
import os, math, time , trimesh , pyrender


os.environ['PYOPENGL_PLATFORM'] = 'egl'
np.infty = np.inf  # pyrender compatibility

def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    yaw_mat = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ], dtype=np.float32)
    pitch_mat = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch),  math.cos(pitch)]
    ], dtype=np.float32)
    return yaw_mat @ pitch_mat

class Render:
    def __init__(self, config):
        self.config = config
        
        # --- Camera setup ---
        cam_height = 10.0
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi/6, aspectRatio=1.0)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.camera_pose[:3, 3] = [0, -2.7, 2]
        self.camera_pose[:3, :3] = rotation_matrix(0, 60)

        # --- Light setup ---s
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self.light_pose = np.eye(4, dtype=np.float32)
        self.light_pose[:3,:3] = rotation_matrix(0, -30)

         # --- Floor mesh ---
        floor_size = 2.0
        vs = np.array([
            [ floor_size, 0, floor_size],
            [-floor_size, 0, floor_size],
            [-floor_size, 0, -floor_size],
            [ floor_size, 0, -floor_size]
        ], dtype=np.float32)
        fs = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        floor_tm = trimesh.Trimesh(vertices=vs, faces=fs)
        self.floor_mesh = pyrender.Mesh.from_trimesh(floor_tm, smooth=False)


        ############################
        # -- Link mesh ---
        box_tm = trimesh.creation.box(extents=config.link.size)
        red_rgba = [1.0, 0.0, 0.0, 1.0]
        red_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=red_rgba,
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        self.box_mesh = pyrender.Mesh.from_trimesh(box_tm, material=red_mat, smooth=True)
        self.box_pose = np.eye(4, dtype=np.float32)

        self.box_pose[:3, 3] = [-0.5, 0.0, 2]

        # --- Scene setup ---
        self.scene = pyrender.Scene()
        # Add static nodes

        self.floor_node = self.scene.add(self.floor_mesh)
        self.cam_node   = self.scene.add(self.camera,    pose=self.camera_pose)
        self.light_node = self.scene.add(self.light,     pose=self.light_pose)

        # Will be created on first view()
        self.box_node   = None
        self.cloud_node = None
        self.viewer     = None


    def view(self, points: np.ndarray, colors: np.ndarray):
        """
        Update point cloud and render scene in a background thread.
        Prints camera position & orientation each call.
        """

        # Launch the viewer on first call
        if self.viewer is None:
            self.viewer = pyrender.Viewer(
                self.scene,
                use_raymond_lighting=True,
                run_in_thread=True,
                fullscreen=True,
                window_title = "Robot Tissue Interaction",
                viewport_size=(1920, 1080)
            )
            # Give the window a moment to initialize
            time.sleep(0.1)

        # Create a new point cloud mesh
        cloud = pyrender.Mesh.from_points(points, colors=colors)




        # Safely swap out the old cloud under render_lock
        with self.viewer.render_lock:
            if self.cloud_node is not None:
                self.scene.remove_node(self.cloud_node)
                # self.scene.remove_node(self.box_node)
            self.cloud_node = self.scene.add(cloud)

            # self.box_node   = self.scene.add(self.box_mesh, pose=self.box_pose)

        # Brief pause so the render thread can catch up
        time.sleep(1e-3)


ti.init(arch=ti.gpu)

# dim, n_grid, steps, dt = 2, 128, 20, 2e-4
# dim, n_grid, steps, dt = 2, 256, 32, 1e-4
dim, n_grid, steps, dt = 3, 32, 25, 4e-4
# dim, n_grid, steps, dt = 3, 64, 25, 2e-4
# dim, n_grid, steps, dt = 3, 128, 25, 8e-5

n_particles = n_grid**dim // 2 ** (dim - 1)
dx = 1 / n_grid

p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
F_J = ti.field(float, n_particles)

F_grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)
F_grid_m = ti.field(float, (n_grid,) * dim)

neighbour = (3,) * dim


@ti.kernel
def substep():
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E * p_vol * (F_J[p] - 1) / dx**2
        affine = ti.Matrix.identity(float, dim) * stress + p_mass * F_C[p]
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I][1] -= dt * gravity
        cond = (I < bound) & (F_grid_v[I] < 0) | (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        F_J[p] *= 1 + dt * new_C.trace()
        F_C[p] = new_C


@ti.kernel
def init():
    for i in range(n_particles):
        F_x[i] = ti.Vector([ti.random() for i in range(dim)]) * 0.4 + 0.5
        F_J[i] = 1


def T(a):
    if dim == 2:
        return a

    phi, theta = np.radians(28), np.radians(32)

    a = a - 0.5
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    x, z = x * cp + z * sp, z * cp - x * sp
    u, v = x, y * ct + z * st
    return np.array([u, v]).swapaxes(0, 1) + 0.5





init()
cfg = CN()
cfg.link         = CN()
cfg.link.size    = (0.02,0.02,0.3)
cfg.link.pos     = (0.5,0,0)

render = Render(cfg)

while True:

    for s in range(steps):
        substep()
    pos = F_x.to_numpy()
    colors = np.ones((n_particles, 4), dtype=np.float32)
    colors[:, :3] = (pos + 1.0) * 0.5
    render.view(pos , colors)



