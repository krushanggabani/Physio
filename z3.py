import os
import math
import time
import threading
import numpy as np
import taichi as ti
import trimesh
import pyrender


# Headless EGL rendering for PyOpenGL
os.environ['PYOPENGL_PLATFORM'] = 'egl'
np.infty = np.inf

# ─── Utility Functions ────────────────────────────────────────────
def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    """Yaw (Y-axis) + Pitch (X-axis) rotation matrix."""
    yaw, pitch = map(math.radians, (yaw, pitch))
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


def create_sphere(center, radius=0.025, color=[1, 0, 0, 1]):
    """Create a simple icosphere mesh."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color)
    return pyrender.Mesh.from_trimesh(sphere, smooth=False, material=material), center


class PyRenderer3D:
    def __init__(self):

        self.solid_mode = "mesh"       # solid, mesh
        self.particles = None
        self.roller_center = np.array([0,0,0])
        self.viewer, self.object_node = None, None
        self.robot_node1 = self.robot_node2 = None
        self.roller_mesh,self.roller_pose = None,None

        self.running = True


        # Scene Setup
        self.scene = pyrender.Scene()
        self._setup_camera_light()
        self._setup_floor()

    def _setup_camera_light(self):
        cam_pose = np.eye(4, dtype=np.float32)
        cam_pose[:3, 3] = np.array((0.5, -2, 0.35), dtype=np.float32)
        cam_pose[:3, :3] = rotation_matrix(0.0, 95.0)
        self.camera_node = self.scene.add(pyrender.PerspectiveCamera(yfov=np.pi/6), pose=cam_pose)

        light_pose = np.eye(4, dtype=np.float32)
        light_pose[:3, 3] = np.array((0, 0, 1), dtype=np.float32)
        light_pose[:3, :3] = rotation_matrix(0.0, -30.0)
        self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=light_pose)

    def _setup_floor(self):
        vs = np.array([[2, 2, 0], [-2, 2, 0], [-2, -2, 0], [2, -2, 0]], dtype=np.float32)
        fs = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        floor_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=vs, faces=fs))
        self.scene.add(floor_mesh)

    def set_particles(self, particles):
        self.particles = particles


    def _reconstruct_solid(self):
        hull = trimesh.Trimesh(vertices=self.particles).convex_hull
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.47, 0.0, 0.0, 1])
        return pyrender.Mesh.from_trimesh(hull, smooth=True,material=material)
    
    def _monitor_viewer(self):
        while self.viewer.is_active:
            time.sleep(0.1)
        self.running = False
    
    def update_robot(self,roller_center):
        self.roller_center = roller_center


    def render(self):
        
        if self.viewer is None:
            self.viewer = pyrender.Viewer(
                self.scene, use_raymond_lighting=True, run_in_thread=True,
                window_title="Robot Tissue Interaction"
            )
            threading.Thread(target=self._monitor_viewer, daemon=True).start()



        # pose = self.viewer._camera_node.matrix
        # position = pose[:3, 3]
        # print("Current Camera Position:", position)

        if self.solid_mode == "solid":
            cloud = self._reconstruct_solid()
        else:
            N = len(self.particles) 
            sphere_tm = trimesh.creation.uv_sphere(radius=0.002)
            RGBA = np.array([60, 160, 255, 255], dtype=np.uint8)   # pale blue
            sphere_tm.visual.vertex_colors = np.tile(RGBA, (sphere_tm.vertices.shape[0], 1))
            tfs = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], N, axis=0)
            tfs[:, :3, 3] = self.particles.astype(np.float32) 
            cloud = pyrender.Mesh.from_trimesh(sphere_tm, poses=tfs, smooth=False)
        

        # Roller as sphere
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=roller_radius)
        T = np.eye(4)
        T[:3, 3] = self.roller_center
        self.roller_pose = T
        
        self.roller_mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1, 0, 0, 1]))
        
       
        with self.viewer.render_lock:
            if self.object_node is None:
                self.object_node = self.scene.add(cloud, name="particles")
            else:
                self.object_node.mesh = cloud

            
            if self.robot_node1 is None:
                self.robot_node1 = self.scene.add(self.roller_mesh,pose=self.roller_pose)
            else:
                self.scene.set_pose(self.robot_node1, self.roller_pose)

                

##########################################################################

# ─── Taichi init ───────────────────────────────────────────────────────────
ti.init(arch=ti.vulkan)

# ─── Simulation parameters ─────────────────────────────────────────────────
dim         = 3
n_particles = 9000
n_grid      = 32
dx          = 1.0 / n_grid
inv_dx      = float(n_grid)
dt          = 2e-4
gravity     = ti.Vector([0.0,0.0,-9.8])

# ─── Material (Neo‐Hookean) ────────────────────────────────────────────────
p_rho   = 1.0
p_vol   = (dx * 0.5)**2
p_mass  = p_rho * p_vol
E       = 5e3
nu      = 0.2
mu_0     = E / (2*(1 + nu))
lambda_0 = E * nu / ((1 + nu)*(1 - 2*nu))

# ─── Floor & domain boundaries ─────────────────────────────────────────────
floor_level    = 0.0
floor_friction = 0.4


# ─── Roller & contact fields (single hand) ─────────────────────────────────
roller_radius     = 0.025
roller_center     = ti.Vector.field(dim, dtype=ti.f32, shape=())
roller_velocity   = ti.Vector.field(dim, dtype=ti.f32, shape=())

state          = ti.field(ti.i32, shape=())   # 0=descend, 1=roll
contact_height = ti.field(ti.f32, shape=())
contact_force  = ti.field(ti.f32, shape=())

# ─── Speeds ───────────────────────────────────────────────────────────────
v_desc = 0.5  # vertical descent m/s
v_roll = 0.2  # horizontal roll m/s


# ─── MPM fields ────────────────────────────────────────────────────────────
x      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
F      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
C      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
J      = ti.field(dtype=ti.f32, shape=n_particles)
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid,n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid,n_grid))


half_radius   = 0.2
soft_center = ti.Vector.field(dim, dtype=ti.f32, shape=())
soft_center   = ti.Vector([0.5, 0.5, 0.0])

# ─── Neo-Hookean stress ───────────────────────────────────────────────────
@ti.func
def neo_hookean_stress(F_i):
    J_det = F_i.determinant()
    FinvT = F_i.inverse().transpose()
    return mu_0 * (F_i - FinvT) + lambda_0 * ti.log(J_det) * FinvT

# ─── Initialize semi-circular soft-body & roller ───────────────────────────
@ti.kernel
def init_particles():
    for p in range(n_particles):
        u1 = ti.random()
        u2 = ti.random()
        u3 = ti.random()
        r = half_radius * u1 ** (1 / 3)
        theta = math.pi * u2
        phi   = ti.acos(2 * u3 - 1)
        x[p] = ti.Vector([
            r * ti.sin(phi) * ti.cos(theta) + 0.5,
            r * ti.cos(phi) + 0.5,
            r * ti.sin(phi) * ti.sin(theta)])
        

        v[p] = ti.Vector([0.0, 0.0, 0.0])
        F[p] = ti.Matrix.identity(ti.f32, dim)
        J[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)

    state[None]          = 0
    contact_height[None] = 0.0
    contact_force[None]  = 0.0
    roller_center[None] = ti.Vector([0.5, 0.5, 0.5])



# ─── Particle to Grid (P2G) ────────────────────────────────────────────────
@ti.kernel
def p2g():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, dim)
        grid_m[I] = 0.0
    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * neo_hookean_stress(F[p])
        affine = stress + p_mass * C[p]

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos   = (offset.cast(ti.f32) - fx) * dx
            weight = w[i].x * w[j].y * w[k].z
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass






# ─── Update roller position ────────────────────────────────────────────────
def update_roller_position():
    c = roller_center[None]
    if state[None] == 0:
        c.z -= v_desc * dt
    else:
        c.x += v_roll * dt
        c.z = contact_height[None]
    roller_center[None] = c


#  ─── Grid forces, roller (normal-only), floor, walls, force detect ─────────

@ti.kernel
def apply_grid_forces_and_detect():
    contact_force[None] = 0.0
    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 0:
            v_old = grid_v[I] / m
            v_new = v_old + dt * ti.Vector([0.0, 0.0,-9.8])
            pos   = I.cast(ti.f32) * dx

            # Roller contact (normal only)
            if (pos - roller_center[None]).norm() < roller_radius:
                v_target = ti.Vector.zero(ti.f32, dim)
                if state[None] == 0:
                    v_target = ti.Vector([0.0, 0.0,-v_desc])
                else:
                    v_target = ti.Vector([v_roll, 0.0, 0.0])
                n       = (pos - roller_center[None]).normalized()
                v_norm  = n * (n.dot(v_target))
                v_tang  = v_old - n * (n.dot(v_old))
                v_new   = v_tang + v_norm
                f_imp   = m * (v_new - v_old) / dt
                contact_force[None] += f_imp.norm()

            # Floor contact
            if pos.z < floor_level + dx:
                if v_new.z < 0: v_new.z = 0
                v_new.x *= (1 - floor_friction)
                v_new.y *= (1 - floor_friction)

            # Walls on grid
            if pos.x < dx:      v_new.x = 0
            if pos.x > 1-dx:    v_new.x = 0

            if pos.y < dx:      v_new.y = 0
            if pos.y > 1-dx:    v_new.y = 0

            grid_v[I] = v_new * m

    if state[None] == 0 and contact_force[None] >= 2.0:
        state[None] = 1
        contact_height[None] = roller_center[None].z

# ─── Grid to Particle (G2P) with boundaries ────────────────────────────────
@ti.kernel
def g2p():
    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]

        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)
        for i, j , k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos   = (offset.cast(ti.f32) - fx) * dx
            weight = w[i].x * w[j].y * w[k].z
            g_v    = grid_v[base + offset] / grid_m[base + offset]
            new_v  += weight * g_v
            new_C  += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v

        # Particle-level boundaries
        if x[p].z< floor_level:
            x[p].z = floor_level
            v[p].z = 0; 
            v[p].x *= (1 - floor_friction)
            v[p].y *= (1 - floor_friction)
            

        if x[p].x < dx:
            x[p].x = dx; v[p].x = 0
        if x[p].x > 1 - dx:
            x[p].x = 1 - dx; v[p].x = 0

        if x[p].y > 1 - dx:
            x[p].y = 1 - dx; v[p].y = 0


        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * new_C) @ F[p]
        J[p] = F[p].determinant()




# ─── Main Loop ────────────────────────────────────────────────────────────
init_particles()
scene = PyRenderer3D()


while scene.running:
    

    for _ in range(20):
        p2g()
        apply_grid_forces_and_detect()
        update_roller_position()
        g2p()



    pts = x.to_numpy()
    scene.set_particles(pts)
    scene.update_robot(roller_center[None].to_numpy())
    scene.render()
    time.sleep(0.01)