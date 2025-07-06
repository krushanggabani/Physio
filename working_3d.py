import taichi as ti
import numpy as np
import pyrender
import trimesh
import math


np.infty = np.inf

# ─── Taichi init ───────────────────────────────────────────────────────────
ti.init(arch=ti.vulkan)

# ─── Simulation parameters ─────────────────────────────────────────────────
dim         = 3
n_particles = 50000
n_grid      = 64
dx          = 1.0 / n_grid
inv_dx      = float(n_grid)
dt          = 2e-4



# ─── Material (Neo‐Hookean) ────────────────────────────────────────────────
p_rho   = 1.0
p_vol   = (dx * 0.5)**dim
p_mass  = p_rho * p_vol
E       = 5.65e4
nu      = 0.185
mu_0    = E / (2 * (1 + nu))
lambda_0= E * nu / ((1 + nu) * (1 - 2 * nu))


# ─── Floor & domain boundaries ─────────────────────────────────────────────
floor_level    = 0.0
floor_friction = 0.4


# ─── Single‐hand 2‐link arm params ─────────────────────────────────────────
L1, L2 = 0.12, 0.10
theta1 = np.array([0.0], dtype=np.float32)
theta2 = np.array([0.0], dtype=np.float32)
dtheta1 = np.zeros(1, dtype=np.float32)
dtheta2 = np.zeros(1, dtype=np.float32)
theta1_rest = theta1.copy()
theta2_rest = theta2.copy()
k   = 10    
b   = 0.5

I1 = L1**2 / 12.0
I2 = L2**2 / 12.0

# ─── Trajectory paramteres ─────────────────────────────────────────────────

base_x, base_y = 0.0 ,0.0 
z0 = 0.4
A = 0.1
ω = 0.5
base_z = z0
time_t = 0.0

# ─── Roller & contact fields (single hand) ─────────────────────────────────
roller_radius = 0.025
roller_center = ti.Vector.field(dim, dtype=ti.f32, shape=1)
roller_velocity = ti.Vector.field(dim, dtype=ti.f32, shape=1)
contact_force_vec = ti.Vector.field(dim, dtype=ti.f32, shape=1)


# ─── MPM fields ────────────────────────────────────────────────────────────
x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
J = ti.field(dtype=ti.f32, shape=n_particles)
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))



# ─── Neo‐Hookean stress ────────────────────────────────────────────────────
@ti.func
def neo_hookean_stress(F_i):
    J_det = F_i.determinant()
    FinvT = F_i.inverse().transpose()
    return mu_0 * (F_i - FinvT) + lambda_0 * ti.log(J_det) * FinvT



PI = 3.141592653589793
half_radius = 0.2
soft_center = np.array([0.0,0.0,0.0])


@ti.kernel
def init_mpm():
    for p in range(n_particles):
        u = ti.random()
        v_ = ti.random()
        w = ti.random()
        r = half_radius * u ** (1/3)
        theta =  PI * v_
        phi = ti.acos(2*w - 1)
        x[p] = ti.Vector([
            soft_center[0] + r * ti.sin(phi) * ti.cos(theta),
            soft_center[1] + r * ti.cos(phi),
            soft_center[2] + r * ti.sin(phi) * ti.sin(theta),
        ])
        v[p] = ti.Vector.zero(ti.f32, dim)
        F[p] = ti.Matrix.identity(ti.f32, dim)
        J[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)

    # base = ti.Vector([base_x, base_y, base_z])
    # j2 = base + ti.Vector([ti.sin(theta1[0]), -ti.cos(theta1[0]),0]) * L1
    # ee = j2 + ti.Vector([ti.sin(theta1[0] + theta2[0]),
    #                      -ti.cos(theta1[0] + theta2[0]), 0]) * L2

        # robot now moves in the X–Z plane (y is constant = base_y)
    base = ti.Vector([base_x, base_y, base_z])
    # joint2: x = sin(θ), y stays at base_y, z = –cos(θ)
    j2 = base + ti.Vector([ 
        ti.sin(theta1[0]),    # Δx
        0.0,                   # Δy
        -ti.cos(theta1[0])     # Δz
    ]) * L1

    # end effector: same pattern but with θ1+θ2
    ee = j2 + ti.Vector([
        ti.sin(theta1[0] + theta2[0]),
        0.0,
        -ti.cos(theta1[0] + theta2[0])
    ]) * L2

    roller_center[0] = ee
    roller_velocity[0] = ti.Vector.zero(ti.f32, dim)
    contact_force_vec[0] = ti.Vector.zero(ti.f32, dim)



@ti.kernel
def p2g():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, dim)
        grid_m[I] = 0.0
    for p in range(n_particles):
        Xp = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * neo_hookean_stress(F[p])
        affine = stress + p_mass * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offs = ti.Vector([i, j, k])
            dpos = (offs.cast(ti.f32) - fx) * dx
            wt = w[i].x * w[j].y * w[k].z
            grid_v[base + offs] += wt * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offs] += wt * p_mass

@ti.kernel
def apply_grid_forces_and_detect():
    contact_force_vec[0] = ti.Vector.zero(ti.f32, dim)
    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 0:
            v_old = grid_v[I] / m
            v_new = v_old + dt * ti.Vector([0.0, -9.8, 0.0])
            pos = I.cast(ti.f32) * dx
            rel = pos - roller_center[0]
            if rel.norm() < roller_radius:
                rv = roller_velocity[0]
                n = rel.normalized()
                v_norm = n * n.dot(rv)
                v_tan = v_old - n * (n.dot(v_old))
                v_new = v_tan + v_norm
                delta_v = v_new - v_old
                f_imp = m * delta_v / dt
                contact_force_vec[0] += f_imp

            if pos.y < floor_level + dx:
                if v_new.y < 0:
                    v_new.y = 0
                v_new.x = 0
                v_new.z = 0

            for d in ti.static(range(dim)):
                if pos[d] < dx: v_new[d] = 0
                if pos[d] > 1 - dx: v_new[d] = 0

            grid_v[I] = v_new * m

@ti.kernel
def g2p():
    for p in range(n_particles):
        Xp = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offs = ti.Vector([i, j, k])
            dpos = (offs.cast(ti.f32) - fx) * dx
            wt = w[i].x * w[j].y * w[k].z
            gv = grid_v[base + offs] / grid_m[base + offs]
            new_v += wt * gv
            new_C += 4 * inv_dx * wt * gv.outer_product(dpos)
        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v
        if x[p].y < floor_level:
            x[p].y = floor_level
            v[p].y = 0
            v[p].x = 0
            v[p].z = 0
        for d in ti.static(range(dim)):
            if x[p][d] < dx:
                x[p][d] = dx
                v[p][d] = 0
            if x[p][d] > 1 - dx:
                x[p][d] = 1 - dx
                v[p][d] = 0
        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * new_C) @ F[p]
        J[p] = F[p].determinant()

def update_base_and_arm():
    global time_t, base_y, theta1, theta2, dtheta1, dtheta2
    time_t += dt * 15
    base_y = z0 + A * np.cos(ω * time_t)
    Fc = contact_force_vec[0].to_numpy()
    base = np.array([base_x, base_y, base_z], dtype=np.float32)
    j2 = base + np.array([np.sin(theta1[0]), -np.cos(theta1[0]), 0]) * L1
    ee_old = roller_center[0].to_numpy()
    ee_new = j2 + np.array([np.sin(theta1[0] + theta2[0]),
                            -np.cos(theta1[0] + theta2[0]), 0]) * L2
    rv = (ee_new - ee_old) / dt
    roller_center[0] = ee_new.tolist()
    roller_velocity[0] = rv.tolist()
    r1 = ee_new - base
    r2 = ee_new - j2
    tau1_c = r1[1] * Fc[0] - r1[0] * Fc[1]
    tau2_c = r2[1] * Fc[0] - r2[0] * Fc[1]
    tau1 = tau1_c - k * (theta1[0] - theta1_rest[0]) - b * dtheta1[0]
    tau2 = tau2_c - k * (theta2[0] - theta2_rest[0]) - b * dtheta2[0]
    alpha1 = tau1 / I1
    alpha2 = tau2 / I2
    dtheta1[0] += alpha1 * dt
    theta1[0] += dtheta1[0] * dt
    dtheta2[0] += alpha2 * dt
    theta2[0] += dtheta2[0] * dt

init_mpm()





# Utility: build combined yaw (around Y-axis) and pitch (around X-axis) rotation matrix
def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
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

class Renderer:
    def __init__(self, camera_height: float = 10.0, floor_size: float = 2.0):
        """
        Set up camera, light, and floor mesh for scene visualization.
        camera_height: Z-axis position of the camera
        floor_size: half-extent of square floor in X and Y
        """
        # Perspective camera (field-of-view 30°)
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi/6, aspectRatio=1.0)
        # Build camera pose (translation + orientation)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.camera_pose[:3, 3] = np.array([0, 0, camera_height], dtype=np.float32)
        self.camera_pose[:3, :3] = rotation_matrix(yaw=0.0, pitch=0.0)

        # Directional light pitched downward by 30°
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self.light_pose = np.eye(4, dtype=np.float32)
        self.light_pose[:3, :3] = rotation_matrix(yaw=0.0, pitch=-math.pi/6)

        # Create a flat floor mesh in the X-Y plane at Z=0
        vs = np.array([
            [ floor_size,  floor_size, 0],
            [-floor_size,  floor_size, 0],
            [-floor_size, -floor_size, 0],
            [ floor_size, -floor_size, 0]
        ], dtype=np.float32)
        fs = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        floor_trimesh = trimesh.Trimesh(vertices=vs, faces=fs)
        self.floor_mesh = pyrender.Mesh.from_trimesh(floor_trimesh, smooth=False)




        # # Create an axis-aligned box (1 x 2 x 0.5)
        # box_trimesh = trimesh.creation.box(extents=(0.2, 0.2, 0.5))
        # self.box_mesh = pyrender.Mesh.from_trimesh(box_trimesh, smooth=True)
        # # Position box at (-3, 0, 0.25) so it sits on the floor
        # self.box_pose = np.eye(4, dtype=np.float32)
        # self.box_pose[:3, 3] = np.array([-0.5, 0.0, 0.25], dtype=np.float32)



    def render(self, points: np.ndarray, colors: np.ndarray):
        """
        Build the Pyrender scene and launch the viewer.

        points: (N,3) array of XYZ coords
        colors: (N,4) array of RGBA values
        """
        scene = pyrender.Scene()

        # Add point cloud as a mesh of point sprites
        cloud = pyrender.Mesh.from_points(points, colors=colors)
        scene.add(cloud)
        

        # Add floor, camera, and light
        scene.add(self.floor_mesh)
        scene.add(self.camera, pose=self.camera_pose)
        scene.add(self.light, pose=self.light_pose)



        # # Arm links as cylinders
        base_pt = np.array([base_x, base_y, base_z], dtype=np.float32)
        j2 = base_pt + np.array([np.sin(theta1[0]), 0, -np.cos(theta1[0])]) * L1
        ee = roller_center[0].to_numpy()



        def add_cylinder(start, end, radius, color):
            vec = end - start
            length = np.linalg.norm(vec)
            if length < 1e-7:
                return
            center = (start + end) / 2
            z = vec / length
            v = np.cross([0, 0, 1], z)
            c = np.dot([0, 0, 1], z)
            s = np.linalg.norm(v)
            if s < 1e-7:
                R = np.eye(3)
            else:
                vx = np.array([
                    [0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]
                ])
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = center
            cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
            mesh = pyrender.Mesh.from_trimesh(cyl, smooth=False, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color))
            scene.add(mesh, pose=T)

        add_cylinder(base_pt, j2, roller_radius, [0, 0, 0.3, 1])
        add_cylinder(j2, ee, roller_radius, [0, 0, 0.3, 1])

        # Roller as sphere
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=roller_radius)
        T = np.eye(4)
        T[:3, 3] = ee
        
        mesh = pyrender.Mesh.from_trimesh(sphere, smooth=False, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1, 0, 0, 1]))
        scene.add(mesh, pose=T)


        # Launch interactive viewer (blocks until closed)
        pyrender.Viewer(scene, use_raymond_lighting=True)



points_np = x.to_numpy()

colors = np.ones((n_particles, 4), dtype=np.float32)
colors[:, :3] = (points_np + 1.0) * 0.5


renderer = Renderer(camera_height=10.0, floor_size=2.0)
renderer.render(points_np, colors)


# for frame in range(1):
#     for _ in range(10):
#         p2g()
#         apply_grid_forces_and_detect()
#         g2p()
#         update_base_and_arm()
    

print("Simulation ended.")

