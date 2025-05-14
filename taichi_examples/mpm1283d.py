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
               
            self.cloud_node = self.scene.add(cloud)


        # Brief pause so the render thread can catch up
        time.sleep(1e-3)


ti.init(arch=ti.gpu)  # Try to run on GPU

quality = 1  # Use a larger value for higher-res simulations
dim     = 3
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(dim, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(dim, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)  # deformation gradient

material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid,)*dim)  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid,)*dim)  # grid node mass
gravity = ti.Vector.field(dim, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(dim, dtype=float, shape=())


@ti.kernel
def substep():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, dim) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:
            # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, dim) * ti.sqrt(J)
        elif material[p] == 2:
            # Reconstruct elastic deformation gradient after plasticity
            F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, dim) * la * J * (
            J - 1
        )
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
            # Loop over 3x3 grid node neighborhood
            # offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = ti.cast(1.0, float)
            for d in ti.static(range(dim)):
                weight *= w[offset[d]][d]

            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    
    for I, J, K in grid_m:
        m = grid_m[I, J, K]
        if m > 1e-10:
            v_ijk = grid_v[I, J, K] / m
            v_ijk -= dt * gravity[None]  # apply gravity on y-axis
            # simple bounce at boundaries
            if I < 3    and v_ijk[0] < 0: v_ijk[0] = 0
            if I > n_grid - 3 and v_ijk[0] > 0: v_ijk[0] = 0
            if J < 3    and v_ijk[1] < 0: v_ijk[1] = 0
            if J > n_grid - 3 and v_ijk[1] > 0: v_ijk[1] = 0
            if K < 3    and v_ijk[2] < 0: v_ijk[2] = 0
            if K > n_grid - 3 and v_ijk[2] > 0: v_ijk[2] = 0
            grid_v[I, J, K] = v_ijk * m

    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float,dim)
        new_C = ti.Matrix.zero(float, dim,dim)

        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * dim)))):
            # loop over 3x3 grid node neighborhood
            dpos = offset.cast(float) - fx
            g_v = grid_v[base + offset]

            weight = ti.cast(1.0, float)
            for d in ti.static(range(dim)):
                weight *= w[offset[d]][d]

            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


@ti.kernel
def reset():
    group_size = n_particles
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
            ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size),
            ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
        ]
        material[i] = 2 # 0: fluid 1: jelly 2: snow
        v[i] = [0, 0,0]
        F[i] = ti.Matrix([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, dim, dim)


reset()

gravity[None] = [0,0, 10]

cfg = CN()
cfg.link         = CN()
cfg.link.size    = (0.02,0.02,0.3)
cfg.link.pos     = (0.5,0,0)

render = Render(cfg)



for frame in range(20000):
    
    for s in range(int(2e-3 // dt)):
        substep()
    

    # print()
    arr = x.to_numpy()
    # zeros = np.zeros((arr.shape[0], 1))

    # arr3 = np.hstack([arr, zeros])

    colors = np.ones((n_particles, 4), dtype=np.float32)
    colors[:, :3] = (arr + 1.0) * 0.5

    render.view(arr , colors)

