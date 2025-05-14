import os
import math
import time
import numpy as np
import taichi as ti
import trimesh
import pyrender
from yacs.config import CfgNode as CN

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
        [0, math.sin(p), math.cos(p)]
    ], dtype=np.float32)
    return yaw_mat @ pitch_mat

# ------------------------------------------------------------
# Renderer
# ------------------------------------------------------------
class Renderer:
    def __init__(self, link_size, viewport=(1920,1080), title="MPM Viewer"):
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        np.infty = np.inf
        self.viewport = viewport
        self._build_scene(link_size)
        self.viewer = None

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
        vs = np.array([[ 2.0,  2.0,0],[-2.0,2.0,0],[-2.0,-2.0,0],[2.0,-2.0,0]], np.float32)
        fs = np.array([[0,1,2],[0,2,3]], np.int32)
        floor_tm = trimesh.Trimesh(vertices=vs, faces=fs)
        self.floor_mesh = pyrender.Mesh.from_trimesh(floor_tm, smooth=False)
        # Box
        box_tm = trimesh.creation.box(extents=link_size)
        red = [1,0,0,1]
        mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=red, metallicFactor=0, roughnessFactor=1)
        self.box_mesh = pyrender.Mesh.from_trimesh(box_tm, material=mat, smooth=True)
        # Scene
        self.scene = pyrender.Scene()
        self.scene.add(self.floor_mesh)
        self.scene.add(self.camera, pose=self.camera_pose)
        self.scene.add(self.light,  pose=self.light_pose)
        self.box_node = self.scene.add(self.box_mesh, pose=np.eye(4, dtype=np.float32))
        self.cloud_node = None

    def update(self, points: np.ndarray, colors: np.ndarray, box_pose: np.ndarray):
        # Initialize viewer
        if self.viewer is None:
            self.viewer = pyrender.Viewer(self.scene,
                                          use_raymond_lighting=True,
                                          run_in_thread=True,
                                          fullscreen=True,
                                          viewport_size=self.viewport,
                                          window_title="Robot Tissue Interaction")
            time.sleep(0.1)
        # Swap point cloud
        mesh = pyrender.Mesh.from_points(points, colors=colors)
        with self.viewer.render_lock:
            if self.cloud_node:
                self.scene.remove_node(self.cloud_node)
            self.cloud_node = self.scene.add(mesh)
            # Update box
            self.scene.set_pose(self.box_node, box_pose)
        time.sleep(1e-3)

# ------------------------------------------------------------
# MPM Simulator
# ------------------------------------------------------------
@ti.data_oriented
class MPMSimulator:
    def __init__(self, cfg: CN):
        ti.init(arch=ti.gpu, fast_math=True, device_memory_GB=1)
        self._setup(cfg)
        self._declare_fields()
        self._init_kernel()

    def _setup(self, cfg: CN):
        self.dim         = cfg.dim
        self.n_particles = cfg.n_particles
        self.n_grid      = cfg.n_grid
        self.dx          = 1.0 / self.n_grid
        self.inv_dx      = float(self.n_grid)
        self.dt          = cfg.dt
        self.p_vol       = (self.dx * 0.5)**self.dim
        self.p_rho       = 1.0
        self.p_mass      = self.p_vol * self.p_rho
        self.mu0         = cfg.E / (2*(1+cfg.nu))
        self.la0         = cfg.E*cfg.nu/((1+cfg.nu)*(1-2*cfg.nu))
        self.amplitude   = cfg.amplitude
        self.omega       = cfg.omega

    def _declare_fields(self):
        D, N, G = self.dim, self.n_particles, self.n_grid
        self.x   = ti.Vector.field(D, float, N)
        self.v   = ti.Vector.field(D, float, N)
        self.C   = ti.Matrix.field(D, D, float, N)
        self.F   = ti.Matrix.field(D, D, float, N)
        self.grid_v = ti.Vector.field(D, float, (G,)*D)
        self.grid_m = ti.field(float,           (G,)*D)
        self.grid_A = ti.field(int,             (G,)*D)
        self.t       = ti.field(float, shape=())
        self.r_vel   = ti.Vector.field(D, float, shape=())
        self.to_box  = ti.Vector.field(D, float, shape=())
        self.from_box= ti.Vector.field(D, float, shape=())

    def _init_kernel(self):
        @ti.kernel
        def init():
            for i in range(self.n_particles):
                self.x[i] = [ti.random()*0.4-0.2,
                             ti.random()*0.2-0.1,
                             ti.random()*0.2]
                self.v[i] = [0]*self.dim
                self.F[i] = ti.Matrix.identity(float, self.dim)
                self.C[i] = ti.Matrix.zero(float, self.dim, self.dim)
            self.t[None]          = 0.0
            self.r_vel[None]      = [0]*self.dim
            self.to_box[None]     = [-0.010, -0.010, 0.40] 
            self.from_box[None]   =  [0.010,  0.010, 0.70]
        init()

    def substep(self):
        self._substep_kernel()

    def run(self, render: Renderer, steps_per_frame: int=50):
        while True:
            for _ in range(steps_per_frame):
                self.substep()
            pts = self.x.to_numpy()
            cols = np.ones((self.n_particles,4), np.float32)
            cols[:,:3] = (pts+1.0)*0.5
            box_mid = (self.to_box[None]+self.from_box[None])*0.5
            pose4 = np.eye(4, dtype=np.float32)
            pose4[:3,3] = box_mid.to_numpy()
            render.update(pts, cols, pose4)

    @ti.kernel
    def _substep_kernel(self):
        G = self.n_grid
        # Clear grid & build mask
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = [0]*self.dim
            self.grid_m[I] = 0.0
            pos = I * self.dx
            self.grid_A[I] = 1 if (pos >= self.to_box[None]).all() and (pos <= self.from_box[None]).all() else 0
        # P2G
        for p in range(self.n_particles):
            base = (self.x[p]*self.inv_dx-0.5).cast(int)
            fx   = self.x[p]*self.inv_dx-base.cast(float)
            w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2
        ]
            self.F[p] = (ti.Matrix.identity(float,self.dim)+self.dt*self.C[p])@self.F[p]
            # neo-Hookean
            h   = 0.5
            mu,la = self.mu0*h, self.la0*h
            U,S,V = ti.svd(self.F[p])
            J = 1.0
            for d in ti.static(range(self.dim)):
                J *= S[d,d]
            P = 2*mu*(self.F[p]-U@V.transpose())@self.F[p].transpose() + ti.Matrix.identity(float,self.dim)*la*J*(J-1)
            stress = -self.dt*self.p_vol*4*self.inv_dx*self.inv_dx*P
            affine = stress + self.p_mass*self.C[p]

            for offs in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.dim)))):
                
                dpos= (offs.cast(float)-fx)*self.dx
                idx = base+offs
                weight = ti.cast(1.0, float)
                for d in ti.static(range(self.dim)):
                    weight *= w[offs[d]][d]
                # x = base + offset
                self.grid_v[idx] += weight*(self.p_mass*self.v[p]+affine@dpos)
                self.grid_m[idx] += weight*self.p_mass

        # Grid update & G2P
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m>0:
                v_ = self.grid_v[I]/m
                if self.grid_A[I]==1:
                    v_ = self.r_vel[None]
                v_[1] -= self.dt*cfg.gravity
                # boundary bounce
                for d in ti.static(range(self.dim)):
                    if I[d]<3 and v_[d]<0 or I[d]>G-3 and v_[d]>0:
                        v_[d] = 0
                self.grid_v[I] = v_*m
        for p in range(self.n_particles):
            base = (self.x[p]*self.inv_dx-0.5).cast(int)
            fx   = self.x[p]*self.inv_dx-base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2,            0.75 - (fx - 1.0) ** 2,            0.5 * (fx - 0.5) ** 2        ]
            new_v = ti.Vector.zero(float,self.dim)
            new_C = ti.Matrix.zero(float,self.dim,self.dim)

            for offs in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.dim)))):
                idx = base+ offs
                
                dpos = offs.cast(float) - fx    
                weight = ti.cast(1.0, float)
                for d in ti.static(range(self.dim)):
                    weight *= w[offs[d]][d]
              
                g_v = self.grid_v[base+offs]
                if self.grid_A[base+offs]==1:
                    g_v = self.r_vel[None]

                new_v += weight*g_v
                new_C += 4*self.inv_dx*weight*g_v.outer_product(dpos)
            self.v[p], self.C[p] = new_v, new_C
            self.x[p] += self.dt*self.v[p]

# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
if __name__ == "__main__":
    # Build config
    cfg = CN()
    cfg.dim        = 3
    cfg.n_particles= 9000
    cfg.n_grid     = 128
    cfg.dt         = 1e-4
    cfg.E          = 5e3
    cfg.nu         = 0.2
    cfg.amplitude  = 0.2
    cfg.omega      = 10.0
 
    cfg.gravity    = 10.0
    cfg.link       = CN()
    cfg.link.size  = (0.02,0.02,0.3)

    sim    = MPMSimulator(cfg)
    render = Renderer(cfg.link.size)
    sim.run(render)
