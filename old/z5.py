import os
import math
import time
import numpy as np
import taichi as ti
import trimesh
import pyrender
import pyglet
from yacs.config import CfgNode as CN


ti.init(arch=ti.gpu, fast_math=True, device_memory_GB=1)
# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    """
    Build combined yaw (around Y) and pitch (around X) rotation matrix (in degrees).
    """
    y = math.radians(yaw)
    p = math.radians(pitch)
    yaw_mat = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]
    ], dtype=np.float32)
    pitch_mat = np.array([
        [1, 0, 0],
        [0, math.cos(p), -math.sin(p)],
        [0, math.sin(p),  math.cos(p)]
    ], dtype=np.float32)
    return yaw_mat @ pitch_mat

# ------------------------------------------------------------
# Renderer
# ------------------------------------------------------------
class Renderer:
    def __init__(self, link_size=(0.02, 0.02, 0.3), viewport=(600,800), title="MPM Viewer"):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        np.infty = np.inf
        self.viewport = viewport
        self._build_scene(link_size)
        self.viewer = None
        self.label = None  # pyglet text label

    def _build_scene(self, link_size):
        # Camera
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi/6, aspectRatio=1.0)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.camera_pose[:3,3] = [0, -2.7, 2]
        self.camera_pose[:3,:3] = rotation_matrix(0, 60)
        # Light
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self.light_pose = np.eye(4, dtype=np.float32)
        self.light_pose[:3,:3] = rotation_matrix(0, -30)
        # Floor
        vs = np.array([[ 2.0,  2.0,0],[-2.0,2.0,0],[-2.0,-2.0,0],[ 2.0,-2.0,0]], np.float32)
        fs = np.array([[0,1,2],[0,2,3]], np.int32)
        floor_tm = trimesh.Trimesh(vertices=vs, faces=fs)
        self.floor_mesh = pyrender.Mesh.from_trimesh(floor_tm, smooth=False)
        # Box
        box_tm = trimesh.creation.box(extents=link_size)
        red = [1,0,0,1]
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=red,
                                                 metallicFactor=0,
                                                 roughnessFactor=1)
        self.box_mesh = pyrender.Mesh.from_trimesh(box_tm, material=mat, smooth=True)
        # Scene
        self.scene = pyrender.Scene()
        self.scene.add(self.floor_mesh)
        self.scene.add(self.camera, pose=self.camera_pose)
        self.scene.add(self.light,  pose=self.light_pose)
        self.box_node = self.scene.add(self.box_mesh,
                                        pose=np.eye(4, dtype=np.float32))
        self.cloud_node = None

    def update(self, points: np.ndarray, colors: np.ndarray, box_pose: np.ndarray,step :int = 0):
        # Initialize viewer
        if self.viewer is None:
            self.viewer = pyrender.Viewer(self.scene,
                                          use_raymond_lighting=True,
                                          run_in_thread=True,
                                          fullscreen=True,
                                          viewport_size=self.viewport,
                                          window_title="Robot Tissue Interaction")
            
            
            time.sleep(0.01)
        
       
        
        # Swap point cloud
        mesh = pyrender.Mesh.from_points(points, colors=colors)
        with self.viewer.render_lock:
            if self.cloud_node:
                self.scene.remove_node(self.cloud_node)
            self.cloud_node = self.scene.add(mesh)
            # Update box pose
            self.scene.set_pose(self.box_node, box_pose)
            self.viewer._message_text =f"Step: {step}"
            time.sleep(1e-3)

# ------------------------------------------------------------
# MPM Simulator
# ------------------------------------------------------------
@ti.data_oriented
class MPMSim:

    def __init__(self,env_dt=2e-3):
        dim =self.dim = 3
        dtype = self.dtype = ti.f64

        self._yield_stress   = 30
        self.ground_friction = 20
        self.default_gravity = (0.0 , -9.8, 0.0)

        quality =1
        n_particles = self.n_particles = 50000 * quality ** 2
        n_grid      = self.n_grid      = int(128 * quality)

        self.dx, self.inv_dx = 1 / n_grid, float(n_grid)
        self.dt = 2e-4
        self.p_vol, self.p_rho = (self.dx * 0.5) ** 2, 1
        self.p_mass = self.p_vol * self.p_rho


        # material
        self.ptype = 0
        self.material_model = 0
        E, nu = 3e3,0.2
        self._mu, self._lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters


        self.mu = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)
        self.lam = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)
        self.yield_stress = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)

        max_steps = self.max_steps = 100
        self.substeps = int(env_dt / self.dt)
        self.x = ti.Vector.field(dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # position
        self.v = ti.Vector.field(dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # velocity
        self.C = ti.Matrix.field(dim, dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # affine velocity field
        self.F = ti.Matrix.field(dim, dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # deformation gradient

        self.F_tmp = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles), needs_grad=True)  # deformation gradient
        self.U = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)
        self.V = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)
        self.sig = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)

        self.res = res = (n_grid, n_grid) if dim == 2 else (n_grid, n_grid, n_grid)
        self.grid_v_in = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)  # grid node momentum/velocity
        self.grid_m = ti.field(dtype=dtype, shape=res, needs_grad=True)  # grid node mass
        self.grid_v_out = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)  # grid node momentum/velocity

        self.gravity = ti.Vector.field(dim, dtype=dtype, shape=()) # gravity ...
        # self.primitives = primitives
        # self.primitives_contact = [True for _ in range(self.n_primitive)]
        # self.rigid_velocity_control = rigid_velocity_control


        # collision
        self.collision_type = 2 # 0 for grid, 1 for particle, 2 for mixed
        self.grid_v_mixed = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)
        self.v_tmp = ti.Vector.field(dim, dtype=dtype, shape=(n_particles), needs_grad=True)
        self.v_tgt = ti.Vector.field(dim, dtype=dtype, shape=(n_particles), needs_grad=True)


        # Rigid box state
        self.to_box   = ti.Vector.field(dim, self.dtype, ())
        self.from_box = ti.Vector.field(dim, self.dtype, ())
        self.r_vel    = ti.Vector.field(dim, self.dtype, ())
        self.to_box[None], self.from_box[None] = [-0.01, -0.01, 0.40], [ 0.01,  0.01, 0.70]
        self.r_vel[None] = [0.0, 0.0,0.0]

        self.t = ti.field(ti.f32, ())
        self.t[None] = 0.0
        amp, omega = 0.2, 10.0

        self.gravity[None] = self.default_gravity
        self.yield_stress.fill(self._yield_stress)
        self.mu.fill(self._mu)
        self.lam.fill(self._lam)



    @ti.kernel
    def initialize(self):
        for i in range(self.n_particles):
            x_val = ti.random() * 0.4 -0.2
            y_val = ti.random() * 0.2 - 0.1
            z_val = ti.random() * 0.2
            self.x[0, i] = ti.Vector([x_val, y_val,z_val])
            self.v[0, i] = ti.Vector([0.0, 0.0, 0.0])
            self.F[0, i] = ti.Matrix.identity(self.dtype, self.dim)
            self.C[0, i] = ti.Matrix.zero(self.dtype, self.dim, self.dim)


    def run(self, render: Renderer, steps_per_frame: int=5):
        
        frame = 0
        while True:
            
            for s in range(1,steps_per_frame):
                self.substep(s)
                frame = frame + 1
            pts = self.get_x(0)
            cols = np.ones((self.n_particles, 4), np.float32)
            cols[:, :3] = (pts + 1.0) * 2
            box_mid = (self.to_box[None] + self.from_box[None]) * 0.5

            pose4 = np.eye(4, dtype=np.float32)
            pose4[:3, 3] = box_mid.to_numpy()
            render.update(pts, cols, pose4,frame)

    @ti.kernel
    def get_x_kernel(self, f: ti.i32, x: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.x[f, i][j]

    def get_x(self, f):
        x = np.zeros((self.n_particles, self.dim), dtype=np.float64)
        self.get_x_kernel(f, x)
        return x

    @ti.kernel
    def clear_grid(self):
        zero = ti.Vector.zero(self.dtype, self.dim)
        for I in ti.grouped(self.grid_m):
            self.grid_v_in[I] = zero
            self.grid_v_out[I] = zero
            self.grid_m[I] = 0

            self.grid_v_in.grad[I] = zero
            self.grid_v_out.grad[I] = zero
            self.grid_m.grad[I] = 0

           
            self.grid_v_mixed[I] = zero
            self.grid_v_mixed.grad[I] = zero

        for p in range(0, self.n_particles):
            self.v_tmp[p] = zero
            self.v_tmp.grad[p] = zero
            self.v_tgt[p] = zero
            self.v_tgt.grad[p] = zero

    def substep(self,s:int = 1):

    
        self.t[None] += self.dt *s


        self.clear_grid()
        # self.initialize()
        # self.compute_F_tmp(s)
        # self.svd()
        # self.p2g(s)

        # # if self.rigid_velocity_control:
        # #     for i in range(self.n_primitive):
        # #         self.primitives[i].forward_kinematics(s, self.dt)

        # self.grid_op_mixed(s)
        
        # self.g2p(s)

        # compute new r_vel, advance to_box/from_box…

        # then P2G, grid update, G2P, etc.
        # --------------------------------------------------------
# Main entry
# ------------------------------------------------------------
if __name__ == "__main__":

    sim    = MPMSim()
    sim.initialize()
    render = Renderer()
    sim.run(render)
