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
            [ floor_size,  floor_size, 0],
            [-floor_size,  floor_size, 0],
            [-floor_size, -floor_size, 0],
            [ floor_size, -floor_size, 0]
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


    def view(self, points: np.ndarray, colors: np.ndarray, box_pose :np.ndarray):
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
            time.sleep(0.001)

        # Create a new point cloud mesh
        cloud = pyrender.Mesh.from_points(points, colors=colors)


        self.box_pose[:3, 3] = box_pose

        # Safely swap out the old cloud under render_lock
        with self.viewer.render_lock:
            if self.cloud_node is not None:
                self.scene.remove_node(self.cloud_node)
                self.scene.remove_node(self.box_node)
            self.cloud_node = self.scene.add(cloud)

            self.box_node   = self.scene.add(self.box_mesh, pose=self.box_pose)

        # Brief pause so the render thread can catch up
        time.sleep(1e-3)


# -------------------------------------------------------------------
# Taichi init
# -------------------------------------------------------------------
ti.init(arch=ti.gpu, fast_math=True, device_memory_GB=9)

# -------------------------------------------------------------------
# Simulation parameters
# -------------------------------------------------------------------
quality     = 1
dim         = 3
n_particles = 50000 * quality**2
n_grid      = 32 * quality

dx     = 1.0 / n_grid
inv_dx = float(n_grid)
dt     = 1e-4 / quality

# 3D particle volume & mass
p_vol = (dx * 0.5)**2
p_rho = 1.0
p_mass = p_vol * p_rho

# Neo-Hookean material
E, nu    = 0.1e4, 0.2
mu_0     = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# Rigid-box & gravity
gravity   = 9.8
amplitude = 0.1
omega     = 10.0

# -------------------------------------------------------------------
# Taichi fields
# -------------------------------------------------------------------
x       = ti.Vector.field(dim, float, n_particles)
v       = ti.Vector.field(dim, float, n_particles)
C       = ti.Matrix.field(dim, dim, float, n_particles)
F       = ti.Matrix.field(dim, dim, float, n_particles)

grid_v  = ti.Vector.field(dim, float, (n_grid,)*dim)
grid_m  = ti.field(float,          (n_grid,)*dim)
grid_A  = ti.field(int,            (n_grid,)*dim)

t        = ti.field(float, shape=())             # time
r_vel    = ti.Vector.field(dim, float, shape=())  # box velocity
to_box   = ti.Vector.field(dim, float, shape=())  # box min corner
from_box = ti.Vector.field(dim, float, shape=())  # box max corner



# -------------------------------------------------------------------
# Initialization kernel
# -------------------------------------------------------------------
@ti.kernel
def initialize():
    # Initialize particles in a blob
    for i in range(n_particles):
        x[i] = [ti.random() * 0.3 + 0.2, ti.random() * 0.3 + 0.2, ti.random() * 0.2]
        v[i] = [0.0, 0.0, 0.0]
        F[i] = ti.Matrix.identity(float, dim)
        C[i] = ti.Matrix.zero(float, dim, dim)

    # Rigid‐box initial velocity (stationary in x/y, moving in z)
    r_vel[None]     = [0.0, 0.0, 0.0]
    # Rigid‐box initial corners (xmin, ymin, zmin) and (xmax, ymax, zmax)
    to_box[None]    = [0.290, 0.290, 0.30]   # lower‐front‐left
    from_box[None]  = [0.310, 0.310, 0.70]   # upper‐back‐right


@ti.kernel
def substep():
    # Advance rigid-box position
    t[None] += dt

    
    v_z = -amplitude*omega *(ti.sin(omega * t[None]))
    r_vel[None]   = ti.Vector([0.0, 0.0, v_z])
    to_box[None] += r_vel[None] * dt
    from_box[None] += r_vel[None] * dt

    # Build active-mask in 3D
    for i, j, k in grid_A:
        pos = ti.Vector([i, j, k]) * dx
        inside = (pos[0] >= to_box[None][0]) & (pos[0] <= from_box[None][0]) \
               & (pos[1] >= to_box[None][1]) & (pos[1] <= from_box[None][1]) \
               & (pos[2] >= to_box[None][2]) & (pos[2] <= from_box[None][2])
        grid_A[i, j, k] = 1 if inside else 0


    # # Clear grid fields
    zero = ti.Vector.zero(float, dim)
    for I in ti.grouped(grid_m):
        grid_v[I] = zero
        grid_m[I] = 0.0

    ti.loop_config(block_dim=n_grid)

    # P2G: scatter particles to grid
    for p in range(n_particles):

        Xp = x[p]/dx

        base = int(Xp-0.5)

        fx = Xp - base
        # Quadratic B-spline weights for each axis
        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2
        ]

        

        stress = ti.Matrix.zero(float, dim, dim)

        # Update deformation gradient
        F[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ F[p]
        # Neo-Hookean stress
        h      = 0.5
        mu, la = mu_0 * h, lambda_0 * h
        U, sigma, V = ti.svd(F[p])
        J = sigma[0, 0] * sigma[1, 1] * (sigma[2, 2] if dim == 3 else 1.0)
        P = (2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose()
             + ti.Matrix.identity(float, dim) * la * J * (J - 1))
        stress = -dt * p_vol * 4 * inv_dx**2 * P
        affine = stress + p_mass * C[p]


        # Scatter mass & momentum
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
            dpos = (offset.cast(float) - fx) * dx
            weight = ti.cast(1.0, float)
            for d in ti.static(range(dim)):
                weight *= w[offset[d]][d]

            # x = base + offset
            # print(base + offset)

            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    # Inject rigid-box velocity into grid nodes inside it  # Grid update: momentum → velocity, apply gravity & BCs
    for I in ti.grouped(grid_m):
        if grid_A[I] == 1 and grid_m[I] > 1e-10:
            grid_v[I] = r_vel[None] * grid_m[I]

    
    for I, J, K in grid_m:
        m = grid_m[I, J, K]
        if m > 1e-10:
            v_ijk = grid_v[I, J, K] / m
            v_ijk[1] -= dt * gravity  # apply gravity on y-axis
            # simple bounce at boundaries
            if I < 3    and v_ijk[0] < 0: v_ijk[0] = 0
            if I > n_grid - 3 and v_ijk[0] > 0: v_ijk[0] = 0
            if J < 3    and v_ijk[1] < 0: v_ijk[1] = 0
            if J > n_grid - 3 and v_ijk[1] > 0: v_ijk[1] = 0
            if K < 3    and v_ijk[2] < 0: v_ijk[2] = 0
            if K > n_grid - 3 and v_ijk[2] > 0: v_ijk[2] = 0
            grid_v[I, J, K] = v_ijk * m

    # G2P: update particles from grid
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)

        # if base < 0 | base > 1 :
        #     print(base)
        fx   = x[p] * inv_dx - base.cast(float)
        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2
        ]
        new_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)

        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
            idx = base+ offset
            dpos = offset.cast(float) - fx

            g_v = grid_v[idx]
            weight = ti.cast(1.0, float)
            if grid_A[idx] == 1:
                g_v = r_vel[None]
            
            for d in ti.static(range(dim)):
                    weight *=w[offset[d]][d]
            
           
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            

        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]

# ----- Setup & main loop -----
initialize()

cfg = CN()
cfg.link         = CN()
cfg.link.size    = (0.02,0.02,0.3)
cfg.link.pos     = (0.5,0,0)

render = Render(cfg)



while True:
    
    substep()

 
    arr3 = x.to_numpy()
    colors = np.ones((n_particles, 4), dtype=np.float32)
    colors[:, :3] = (arr3 + 1.0) * 0.5

    box_pos = (to_box[None] + from_box[None])*0.5
    box_pose = box_pos.to_numpy()
    render.view(arr3 , colors,box_pos)

