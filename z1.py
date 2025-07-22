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




def create_link_mesh(start, end, radius=0.01, color=[0.1, 0.1, 0.8, 1.0]):
    """Create a cylindrical mesh between two 3D points."""
    start, end = np.array(start), np.array(end)
    vec = end - start
    length = np.linalg.norm(vec)

    # Create cylinder aligned with Z-axis
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
    cylinder.visual.vertex_colors = color

    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, vec)
    angle = np.arccos(np.clip(np.dot(z_axis, vec), -1.0, 1.0))
    rot = np.eye(3) if np.linalg.norm(axis) < 1e-6 else \
          trimesh.transformations.rotation_matrix(angle, axis / np.linalg.norm(axis))[:3, :3]

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = (start + end) / 2.0
    return pyrender.Mesh.from_trimesh(cylinder), T


def create_sphere(center, radius=0.025, color=[1, 0, 0, 1]):
    """Create a simple icosphere mesh."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color)
    return pyrender.Mesh.from_trimesh(sphere, smooth=False, material=material), center


class PyRenderer3D:
    def __init__(self):

        self.solid_mode = "mesh"       # solid, mesh
        self.particles = None
        self.robot_center = None
        self.viewer, self.object_node = None, None
        self.robot_node1 = self.robot_node2 = None
        self.link1, self.link2 = None, None
        self.LT1, self.LT2 = np.eye(4), np.eye(4)
        self.running = True


        # Scene Setup
        self.scene = pyrender.Scene()
        self._setup_camera_light()
        self._setup_floor()

    def _setup_camera_light(self):
        cam_pose = np.eye(4, dtype=np.float32)
        cam_pose[:3, 3] = np.array((0.5, -1.5, 2.0), dtype=np.float32)
        cam_pose[:3, :3] = rotation_matrix(0.0, 45.0)
        self.scene.add(pyrender.PerspectiveCamera(yfov=np.pi/6), pose=cam_pose)

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


    def update_robot_arm(self, base, joint2, ee):
        self.link1, self.LT1 = create_link_mesh (base, joint2, color=[0.0, 0.6, 1.0, 1.0])
        self.link2, self.LT2 = create_link_mesh(joint2, ee, color=[1.0, 0.0, 0.0, 1.0])



    def _reconstruct_solid(self):
        hull = trimesh.Trimesh(vertices=self.particles).convex_hull
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.47, 0.0, 0.0, 1])
        return pyrender.Mesh.from_trimesh(hull, smooth=True,material=material)
    


    def _monitor_viewer(self):
        while self.viewer.is_active:
            time.sleep(0.1)
        self.running = False
    

    def update_robot_arm(self, base, joint2, ee):
        """
        Render a 2-link arm using cylinders.
        base: np.array([x, y, z])
        joint2: np.array([x, y, z])
        ee: np.array([x, y, z])
        """
        self.link1,self.LT1 = create_link_mesh(base, joint2, radius=0.01, color=[0.0, 0.6, 1.0, 1.0])
        self.link2,self.LT2 = create_link_mesh(joint2, ee, radius=0.01, color=[1.0, 0.0, 0.0, 1.0])



    def render(self):
        
        if self.viewer is None:
            self.viewer = pyrender.Viewer(
                self.scene, use_raymond_lighting=True, run_in_thread=True,
                window_title="Robot Tissue Interaction"
            )
            threading.Thread(target=self._monitor_viewer, daemon=True).start()

            

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
        
       
        with self.viewer.render_lock:
            if self.object_node is None:
                self.object_node = self.scene.add(cloud, name="particles")
            else:
                self.object_node.mesh = cloud

            # Update Robot Links
            for node, mesh, pose in [(self.robot_node1, self.link1, self.LT1),
                                        (self.robot_node2, self.link2, self.LT2)]:
                if node is not None:
                    self.scene.remove_node(node)
                node = self.scene.add(mesh, pose=pose)
                


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
p_vol   = (dx * 0.5)**3
p_mass  = p_rho * p_vol
E       = 5.65e4
nu      = 0.185
mu_0    = E / (2 * (1 + nu))
lambda_0= E * nu / ((1 + nu) * (1 - 2 * nu))

# ─── Floor & domain boundaries ─────────────────────────────────────────────
floor_level    = 0.0
floor_friction = 0.4

# ─── Single‐hand 2‐link arm params ─────────────────────────────────────────
L1, L2       = 0.12, 0.10
theta1       = np.array([0.0], dtype=np.float32)
theta2       = np.array([0.0],     dtype=np.float32)
dtheta1      = np.zeros(1, dtype=np.float32)
dtheta2      = np.zeros(1, dtype=np.float32)
theta1_rest  = theta1.copy()
theta2_rest  = theta2.copy()

k       = 10       
b       = 0.2   
I1      = L1**2 / 12.0
I2      = L2**2 / 12.0

# ─── Moving base ────────────────────────────
time_t       = 0.0
base_pos      = ti.Vector([0.5, 0.5, 0.4])
A             = 0.1
ω             = 0.5
time_t        = 0.0

# ─── Roller & contact fields (single hand) ─────────────────────────────────
roller_radius     = 0.025
roller_center     = ti.Vector.field(dim, dtype=ti.f32, shape=1)
roller_velocity   = ti.Vector.field(dim, dtype=ti.f32, shape=1)
contact_force_vec = ti.Vector.field(dim, dtype=ti.f32, shape=1)

# ─── MPM fields ────────────────────────────────────────────────────────────
x      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
F      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
C      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
J      = ti.field(dtype=ti.f32, shape=n_particles)
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid,n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid,n_grid))




# ─── Neo‐Hookean stress ────────────────────────────────────────────────────
@ti.func
def neo_hookean_stress(F_i):
    J_det = F_i.determinant()
    FinvT = F_i.inverse().transpose()
    return mu_0 * (F_i - FinvT) + lambda_0 * ti.log(J_det) * FinvT


PI = 3.141592653589793
half_radius   = 0.2
# soft_center_x = 0.5
soft_center   = np.array([0.5, 0.5, 0.0])


# ─── Initialize particles + place roller via FK ────────────────────────────
@ti.kernel
def init_mpm():
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


    # place roller at hand’s FK position
    base = base_pos
    j2 = base + ti.Vector([L1 * ti.sin(theta1[0]),0, -L1 * ti.cos(theta1[0])])
    ee = j2 + ti.Vector([L2 * ti.sin(theta1[0] + theta2[0]), 0, -L2 * ti.cos(theta1[0] + theta2[0])])

    roller_center[0]   = ee
    roller_velocity[0] = ti.Vector.zero(ti.f32, dim)
    contact_force_vec[0] = ti.Vector.zero(ti.f32, dim)

@ti.func
def stencil_range():
    return ti.ndrange(*((3,) * dim))

# ─── Particle→Grid (P2G) ──────────────────────────────────────────────────
@ti.kernel
def p2g():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, dim)
        grid_m[I] = 0.0

    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - base.cast(ti.f32)
        w    = [0.5 * (1.5 - fx)**2,
                0.75 - (fx - 1.0)**2,
                0.5 * (fx - 0.5)**2]
        
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * neo_hookean_stress(F[p])
        affine = stress + p_mass * C[p]

        for offset in ti.static(ti.grouped(stencil_range())):
                dpos = (offset.cast(ti.f32) - fx) * dx
                weight = 1.0
                for d in ti.static(range(dim)):
                    weight *= w[offset[d]][d]
                idx = base + offset
                grid_v[idx] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[idx] += weight * p_mass



# ─── Grid forces & contact detect (normal‐only sliding) ────────────────────
@ti.kernel
def apply_grid_forces_and_detect():

    contact_force_vec[0] = ti.Vector.zero(ti.f32, dim)

    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 1e-10:    # No need for epsilon here, 1e-10 is to prevent potential numerical problems .
            v_old = grid_v[I] / m
            v_new = v_old + dt * gravity  # gravity
            pos   = I.cast(ti.f32) * dx

            # # Roller: enforce only normal component
            # rel = pos - roller_center[0]
            # if rel.norm() < roller_radius:
            #     rv     = roller_velocity[0]
            #     n      = rel.normalized()
            #     v_norm = n * n.dot(rv)
            #     v_tan  = v_old - n * (n.dot(v_old))
            #     v_new  = v_tan + v_norm
            #     delta_v= v_new - v_old
            #     f_imp  = m * delta_v / dt
            #     contact_force_vec[0] += f_imp

            # Floor: clamp vertical AND horizontal for nodes touching floor
            if pos.z < floor_level + dx:
                v_new = ti.Vector.zero(ti.f32, dim)

            # optional wall clamps
            for d in ti.static(range(dim)):
                if pos[d] < dx or pos[d] > 1 - dx:
                    v_new[d] = 0.0

            grid_v[I] = v_new * m


# ─── Grid→Particle (G2P) w/ floor & wall clamp ────────────────────────────
@ti.kernel
def g2p():
    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - base.cast(ti.f32)
        w    = [0.5 * (1.5 - fx)**2,
                0.75 - (fx - 1.0)**2,
                0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)


        for offset in ti.static(ti.grouped(stencil_range())):
            dpos = offset.cast(ti.f32) - fx
            weight = ti.cast(1.0, ti.f32)
            for d in ti.static(range(dim)):
                weight *= w[offset[d]][d]
            
            gv     = grid_v[base + offset] / grid_m[base + offset]
            new_v += weight * gv
            new_C += 4 * inv_dx * weight * gv.outer_product(dpos)

        # Update particle velocity and position
        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v

        # Floor: if particle touches floor, clamp vertical AND horizontal
        if x[p].z < floor_level:
            x[p].z = floor_level
            v[p] = ti.Vector.zero(ti.f32, dim)

        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * new_C) @ F[p]
        J[p] = F[p].determinant()


# ─── Update moving base & single passive arm dynamics ───────────────────────
def update_base_and_arm():
    global time_t, base_y, theta1, theta2, dtheta1, dtheta2

    # 1) Advance time and update vertical base_y
    time_t += dt * 15
    base_y = y0 + A * np.cos(ω * time_t)

    # 2) Read contact force (2D) from MPM kernel
    Fc   = contact_force_vec[0].to_numpy()
    base = np.array([base_x, base_y], dtype=np.float32)

    # 3) Forward kinematics (0 rad = downward)
    j2     = base + np.array([np.sin(theta1[0]), -np.cos(theta1[0])]) * L1
    ee_old = roller_center[0].to_numpy()
    ee_new = j2   + np.array([np.sin(theta1[0] + theta2[0]),
                               -np.cos(theta1[0] + theta2[0])]) * L2
    rv     = (ee_new - ee_old) / dt

    # 4) Update roller’s position & velocity
    roller_center[0]   = ee_new.tolist()
    roller_velocity[0] = rv.tolist()

    # 5) Compute torques via 2D cross‐product + passive spring‐damper
    r1     = ee_new - base       # lever arm from shoulder
    r2     = ee_new - j2         # lever arm from elbow
    tau1_c = r1[1] * Fc[0] - r1[0] * Fc[1]
    # Flip sign so elbow opens outward rather than inward:
    tau2_c = r2[1] * Fc[0] - r2[0] * Fc[1]

    tau1   = tau1_c - k1 * (theta1[0] - theta1_rest[0]) - b1 * dtheta1[0]
    tau2   = tau2_c - k2 * (theta2[0] - theta2_rest[0]) - b2 * dtheta2[0]

    # 6) Integrate joint accelerations (semi‐implicit Euler)
    alpha1      = tau1 / I1
    alpha2      = tau2 / I2
    dtheta1[0] += alpha1 * dt
    theta1[0]  += dtheta1[0] * dt
    dtheta2[0] += alpha2 * dt
    theta2[0]  += dtheta2[0] * dt

# __________Rendering__________________________________________


# ─── Main loop + GUI ──────────────────────────────────────────────────────
init_mpm()
scene = PyRenderer3D()
# viewer = pyrender.Viewer(scene, run_in_thread=True)

while scene.running:

    for _ in range(15):
        p2g()
        apply_grid_forces_and_detect()
        g2p()
    # #     update_base_and_arm()


    pts = x.to_numpy()
    scene.set_particles(pts)
    
    base = np.array([0.5, 0.5, 0.4])

    j2 = base + ti.Vector([L1 * ti.sin(theta1[0]),0, -L1 * ti.cos(theta1[0])])
    ee = j2 + ti.Vector([L2 * ti.sin(theta1[0] + theta2[0]), 0, -L2 * ti.cos(theta1[0] + theta2[0])])

    scene.update_robot_arm(base, j2, ee)
    
    scene.render()
    time.sleep(0.01)


    



