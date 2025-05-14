import taichi as ti
import numpy as np
import imageio
import pyrender
import trimesh
import math

# Initialize Taichi GPU
ti.init(arch=ti.gpu)

@ti.data_oriented
class MPMSim:

    def __init__(self,env_dt=2e-3):
        dim =self.dim = 2
        dtype = self.dtype = ti.f64

        self._yield_stress   = 30
        self.ground_friction = 20
        self.default_gravity = (0.0 , -9.8)

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
        self.to_box   = ti.Vector.field(2, self.dtype, ())
        self.from_box = ti.Vector.field(2, self.dtype, ())
        self.r_vel    = ti.Vector.field(2, self.dtype, ())
        self.to_box[None], self.from_box[None] = [0.48, 0.5], [0.52, 0.7]
        self.r_vel[None] = [0.0, 0.0]

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
            x_val = ti.random() * 0.4 + 0.3
            y_val = ti.random() * 0.2
            self.x[0, i] = ti.Vector([x_val, y_val])
            self.v[0, i] = ti.Vector([0.0, 0.0])
            self.F[0, i] = ti.Matrix.identity(self.dtype, self.dim)
            self.C[0, i] = ti.Matrix.zero(self.dtype, self.dim, self.dim)


    @ti.kernel
    def substep(self):
        # indentation fixed: everything under here
        # t = 2  # your time field
        self.t[None] += self.dt
        # compute new r_vel, advance to_box/from_box…

        # then P2G, grid update, G2P, etc.




class PyRenderer:
    def __init__(self):
        # camera
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 6, aspectRatio=1.0)
        self.camera_pose = np.eye(4)
        pitch, yaw = (-0.25, 0.24)
        pos = (1.0, 0.8, 2.5)
        self.camera_pose[:3, 3] = np.array(pos)
        self.camera_pose[:3, :3] = np.array([
            [np.cos(yaw),   0, np.sin(yaw)],
            [0,             1, 0          ],
            [-np.sin(yaw),  0, np.cos(yaw)],
        ]) @ np.array([
            [1, 0            , 0             ],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch) ],
        ])

        # light
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=8.)

        pitch, yaw = (-1 * math.pi / 6, 0)
        self.light_pose = np.eye(4)
        self.light_pose[:3, :3] = np.array([
            [np.cos(yaw),   0, np.sin(yaw)],
            [0,             1, 0          ],
            [-np.sin(yaw),  0, np.cos(yaw)],
        ]) @ np.array([
            [1, 0            , 0             ],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch) ],
        ])

        # floor
        floor_vertices = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]])
        floor_faces = np.array([[1, 2, 0], [2, 1, 3]])

        n_g = 4
        n_v = n_g + 1
        floor_vertices = np.array([[i / n_g, 0, j / n_g] for i in range(n_v) for j in range(n_v)])
        floor_faces = np.array([[i*n_v+j, i*n_v+j+1, i*n_v+j+n_v, i*n_v+j+n_v+1, i*n_v+j+n_v, i*n_v+j+1] \
            for i in range(n_g) for j in range(n_g)]).reshape(-1, 3)
        floor_colors = np.array([[0.4745, 0.5843, 0.6980, 1.0] if (i % n_g + i // n_g) % 2 == 0 \
            else [0.7706, 0.8176, 0.8569, 1.] for i in range(n_g * n_g)]).repeat(2, axis=0)

        floor_mesh = trimesh.Trimesh(vertices=floor_vertices, faces=floor_faces)
        floor_mesh.visual.face_colors = floor_colors
        self.floor = pyrender.Mesh.from_trimesh(floor_mesh, smooth=False)

        # primitives
        # self.primitives = primitives
        # self.meshes_rest = []
        # self.meshes = []
        # self.mesh_color = [100 / 255, 18 / 255, 22 / 255, 0.8]      # red
        # for i, primitive in enumerate(primitives):
        #     self.meshes_rest.append(primitive.mesh_rest)

        # particle
        self.particles = None
        self.particles_color = None

        # target
        self.target = None

        self.mode = "rgb_array"

    def set_particles(self, particles, colors):
        self.particles = particles
        self.particles_color = [
            (colors[0] >> 16 & 0xFF) / 127,
            (colors[0] >> 8 & 0xFF) / 127,
            (colors[0] & 0xFF) / 127,
            1.0
        ]

    def render(self):
        # particle
        p_mesh = trimesh.creation.uv_sphere(radius=0.002)
        p_mesh.visual.vertex_colors = self.particles_color
        tfs = np.tile(np.eye(4), (len(self.particles), 1, 1))
        tfs[:,:3,3] = self.particles
        particle = pyrender.Mesh.from_trimesh(p_mesh, poses=tfs)

        # scene
        scene = pyrender.Scene()

        scene.add(particle)

        scene.add(self.floor)
        if self.target is not None:
            scene.add(self.target)
        scene.add(self.light, pose=self.light_pose)
        scene.add(self.camera, pose=self.camera_pose)
        pyrender.Viewer(scene, use_raymond_lighting=True)

# Rendering
m = MPMSim()
m.initialize()  


r = PyRenderer()
r.render
