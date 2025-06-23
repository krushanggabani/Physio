import taichi as ti
import numpy as np


PI = 3.141592653589793

@ti.data_oriented
class MPMSim:
    """
    Material Point Method (MPM) simulation with a moving 2-link compliant arm and roller contact.
    """
    def __init__(self, args):
        # ----------------------------------------------------------------------
        # Store external arguments
        # ----------------------------------------------------------------------
        self.args = args
        self.dt = args.Time_step       # Simulation timestep
        self.dim = args.Dimension      # Spatial dimension (e.g., 2)

        # ----------------------------------------------------------------------
        # MPM grid and particle parameters
        # ----------------------------------------------------------------------
        self.n_particles = 20000       # Number of material points
        self.n_grid = 128              # Grid resolution per axis
        self.dx = 1.0 / self.n_grid    # Grid cell size
        self.inv_dx = float(self.n_grid)
        self.p_vol = (self.dx * 0.5) ** 2  # Particle volume
        self.p_rho = 1.0               # Particle density
        self.p_mass = self.p_vol * self.p_rho  # Particle mass

        # ----------------------------------------------------------------------
        # Material properties (linear elastic)
        # ----------------------------------------------------------------------
        self.E = 5.65e4                # Young's modulus
        self.nu = 0.185                # Poisson's ratio
        self.mu_0 = self.E / (2 * (1 + self.nu))                     # Shear modulus
        self.lambda_0 = self.E * self.nu / ((1 + self.nu)*(1 - 2*self.nu))  # Lame's first parameter

        # ----------------------------------------------------------------------
        # Boundary (floor) parameters
        # ----------------------------------------------------------------------
        self.floor_level = 0.0
        self.floor_friction = 0.4

        # ----------------------------------------------------------------------
        # Single‑hand 2‑link compliant arm parameters
        # ----------------------------------------------------------------------
        self.L1, self.L2 = 0.12, 0.10    # Link lengths
        # Joint angles & velocities (initial)
        self.theta1 = np.zeros(1, dtype=np.float32)
        self.theta2 = np.zeros(1, dtype=np.float32)
        self.dtheta1 = np.zeros(1, dtype=np.float32)
        self.dtheta2 = np.zeros(1, dtype=np.float32)
        # Rest angles
        self.theta1_rest = self.theta1.copy()
        self.theta2_rest = self.theta2.copy()
        # Spring-damper parameters at joints
        self.k = 10.0    # Stiffness
        self.b = 0.5      # Damping
        # Rotational inertias
        self.I1 = (self.L1**2) / 12.0
        self.I2 = (self.L2**2) / 12.0
        
        # Additional soft region parameters
        self.half_radius = 0.2        # Soft body radius
        self.soft_center_x = 0.5      # Soft body center x-coordinate
        # ----------------------------------------------------------------------
        # Moving base trajectory parameters
        # ----------------------------------------------------------------------
        
        self.y0 = 0.4                   # Mean base y-position
        self.A = 0.1                    # Amplitude
        self.omega = 0.5                # Angular frequency for sinusoidal motion
        self.time_t = 0.0               # Internal time counter
        self.base_x = 0.5               # Fixed base x-position
        self.base_y = self.y0           # Fixed base y-position     
        # ----------------------------------------------------------------------
        # Roller & contact fields
        # ----------------------------------------------------------------------
        self.roller_radius = 0.025
        # Vector fields for roller center and contact force (1 vector per roller)
        self.roller_center = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.roller_velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.contact_force_vec = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)

        # ----------------------------------------------------------------------
        # Core MPM data fields
        # ----------------------------------------------------------------------
        # Particle data: position, velocity, deformation gradient, affine C, and J
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)
        self.J = ti.field(dtype=ti.f32, shape=self.n_particles)
        # Grid fields: mass and velocity
        self.grid_v = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))


    # ─── Neo‐Hookean stress ────────────────────────────────────────────────────
    @ti.func
    def neo_hookean_stress(self, F_i):
        """
        Compute Neo-Hookean stress for a given deformation gradient.
        """
        J = F_i.determinant()
        FinvT = F_i.inverse().transpose()
        return self.mu_0 * (F_i - FinvT) + self.lambda_0 * ti.log(J) * FinvT

    # ─── Initialize particles + place roller via FK ────────────────────────────
    @ti.kernel
    def init_mpm(self):
        """
        Initialize particle positions in a disk and set initial states.
        """
        for p in range(self.n_particles):
            u = ti.random()
            r = self.half_radius * ti.sqrt(u)
            theta = ti.random() * PI
            self.x[p] = ti.Vector([self.soft_center_x + r * ti.cos(theta),
                                   self.floor_level + r * ti.sin(theta)])
            self.v[p] = ti.Vector.zero(ti.f32, self.dim)
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.J[p] = 1.0
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        # Place roller at end-effector via forward kinematics
        base = ti.Vector([self.base_x, self.base_y])
        j2 = base + ti.Vector([ti.sin(self.theta1[0]), -ti.cos(self.theta1[0])]) * self.L1
        ee = j2 + ti.Vector([ti.sin(self.theta1[0] + self.theta2[0]),
                              -ti.cos(self.theta1[0] + self.theta2[0])]) * self.L2
        self.roller_center[0] = ee
        self.roller_velocity[0] = ti.Vector.zero(ti.f32, self.dim)
        self.contact_force_vec[0] = ti.Vector.zero(ti.f32, self.dim)
        

    # ─── Particle→Grid (P2G) ──────────────────────────────────────────────────
    @ti.kernel
    def p2g(self):
        """
        Particle-to-grid transfer: scatter mass and momentum to grid nodes.
        """
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
            self.grid_m[I] = 0.0

        for p in range(self.n_particles):
            Xp   = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx   = Xp - base.cast(ti.f32)
            w    = [0.5 * (1.5 - fx)**2,
                    0.75 - (fx - 1.0)**2,
                    0.5 * (fx - 0.5)**2]
            stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * self.neo_hookean_stress(self.F[p])
            affine = stress + self.p_mass * self.C[p]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offs = ti.Vector([i, j])
                dpos = (offs.cast(ti.f32) - fx) * self.dx
                wt = w[i].x * w[j].y
                self.grid_v[base + offs] += wt * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offs] += wt * self.p_mass
            
    @ti.kernel
    def apply_grid_forces_and_detect(self):
        """
        Update grid velocities under gravity, floor, walls, and roller contact.
        """
        self.contact_force_vec[0] = ti.Vector.zero(ti.f32, self.dim)
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v_old = self.grid_v[I] / m
                v_new = v_old + self.dt * ti.Vector([0.0, -9.8])
                pos = I.cast(ti.f32) * self.dx
                # Roller contact: only normal component preserved
                rel = pos - self.roller_center[0]
                if rel.norm() < self.roller_radius:
                    rv = self.roller_velocity[0]
                    n = rel.normalized()
                    v_norm = n * n.dot(rv)
                    v_tan = v_old - n * (n.dot(v_old))
                    v_new = v_tan + v_norm
                    delta_v = v_new - v_old
                    f_imp = m * delta_v / self.dt
                    self.contact_force_vec[0] += f_imp
                # Floor and walls clamping
                if pos.y < self.floor_level + self.dx and v_new.y < 0:
                    v_new.y = 0
                    v_new.x = 0
                if pos.x < self.dx or pos.x > 1 - self.dx:
                    v_new.x = 0
                self.grid_v[I] = v_new * m

    @ti.kernel
    def g2p(self):
        """
        Grid-to-particle transfer: update particle states and enforce boundaries.
        """
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx)**2,
                 0.75 - (fx - 1.0)**2,
                 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offs = ti.Vector([i, j])
                dpos = (offs.cast(ti.f32) - fx) * self.dx
                wt = w[i].x * w[j].y
                # avoid divide-by-zero
                m_val = self.grid_m[base + offs]
                if m_val > 0:
                    gv = self.grid_v[base + offs] / m_val
                    new_v += wt * gv
                    new_C += 4 * self.inv_dx * wt * gv.outer_product(dpos)
            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v
            # enforce floor and walls on particles
            if self.x[p].y < self.floor_level:
                self.x[p].y = self.floor_level; self.v[p].y = 0; self.v[p].x = 0
            if self.x[p].x < self.dx:
                self.x[p].x = self.dx;       self.v[p].x = 0
            if self.x[p].x > 1 - self.dx:
                self.x[p].x = 1 - self.dx;   self.v[p].x = 0
            if self.x[p].y > 1 - self.dx:
                self.x[p].y = 1 - self.dx;   self.v[p].y = 0
            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.dt * new_C) @ self.F[p]
            self.J[p] = self.F[p].determinant()

    def update_base_and_arm(self):
        """
        Advance base motion and compute compliant arm response from contact forces.
        """
        # 1) Advance time and update base position
        self.time_t += self.dt * 15
        self.base_y = self.y0 + self.A * np.cos(self.omega * self.time_t)
        # 2) Read contact force
        Fc = self.contact_force_vec[0].to_numpy()
        base = np.array([self.base_x, self.base_y], dtype=np.float32)
        # 3) Forward kinematics
        j2 = base + np.array([np.sin(self.theta1[0]), -np.cos(self.theta1[0])]) * self.L1
        ee_old = self.roller_center[0].to_numpy()
        ee_new = j2 + np.array([np.sin(self.theta1[0] + self.theta2[0]),
                                -np.cos(self.theta1[0] + self.theta2[0])]) * self.L2
        rv = (ee_new - ee_old) / self.dt
        # 4) Update roller fields
        self.roller_center[0] = ee_new.tolist()
        self.roller_velocity[0] = rv.tolist()
        # 5) Compute joint torques
        r1 = ee_new - base
        r2 = ee_new - j2
        tau1_c = r1[1] * Fc[0] - r1[0] * Fc[1]
        tau2_c = r2[1] * Fc[0] - r2[0] * Fc[1]
        tau1 = tau1_c - self.k * (self.theta1[0] - self.theta1_rest[0]) - self.b * self.dtheta1[0]
        tau2 = tau2_c - self.k * (self.theta2[0] - self.theta2_rest[0]) - self.b * self.dtheta2[0]
        # 6) Integrate joint accelerations
        alpha1 = tau1 / self.I1
        alpha2 = tau2 / self.I2
        self.dtheta1[0] += alpha1 * self.dt
        self.theta1[0]  += self.dtheta1[0] * self.dt
        self.dtheta2[0] += alpha2 * self.dt
        self.theta2[0]  += self.dtheta2[0] * self.dt

    def step(self):
        self.p2g()
        self.apply_grid_forces_and_detect()
        self.g2p()
        self.update_base_and_arm()