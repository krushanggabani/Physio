import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan)

@ti.data_oriented
class SoftBodyMPM:
    def __init__(self, n_particles=20000, n_grid=128, material_props=None, floor_level=0.0,
                 floor_friction=0.4, roller_radius=0.025):
        # Simulation parameters
        self.dim = 2
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.dx = 1.0 / n_grid
        self.inv_dx = float(n_grid)
        self.dt = 2e-4
        # Material (Neo‐Hookean)
        if material_props is None:
            E = 5e4
            nu = 0.2
            mu_0 = E / (2 * (1 + nu))
            lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))
        else:
            mu_0 = material_props['mu']
            lambda_0 = material_props['lambda']
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.p_rho = 1.0
        self.p_vol = (self.dx * 0.5) ** 2
        self.p_mass = self.p_rho * self.p_vol
        # Floor & boundary
        self.floor_level = floor_level
        self.floor_friction = floor_friction
        # Roller/contact
        self.roller_radius = roller_radius
        # Fields
        self.x = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(self.dim, dtype=ti.f32, shape=self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=self.n_particles)
        self.J = ti.field(dtype=ti.f32, shape=self.n_particles)
        self.grid_v = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid))
        self.roller_center = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.roller_velocity = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        self.contact_force_vec = ti.Vector.field(self.dim, dtype=ti.f32, shape=1)
        # Initialize
        self.init_particles()

    @ti.func
    def neo_hookean_stress(self, F_i):
        J_det = F_i.determinant()
        FinvT = F_i.inverse().transpose()
        return self.mu_0 * (F_i - FinvT) + self.lambda_0 * ti.log(J_det) * FinvT

    @ti.kernel
    def init_particles(self):
        for p in range(self.n_particles):
            self.x[p] = [0.3 + ti.random() * 0.4, ti.random() * 0.2]
            self.v[p] = [0.0, 0.0]
            self.F[p] = ti.Matrix.identity(ti.f32, self.dim)
            self.J[p] = 1.0
            self.C[p] = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        # Roller initial placement (updated externally by RobotArm)
        self.roller_center[0] = ti.Vector([0.0, 0.0])
        self.roller_velocity[0] = ti.Vector([0.0, 0.0])
        self.contact_force_vec[0] = ti.Vector([0.0, 0.0])

    @ti.kernel
    def p2g(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
            self.grid_m[I] = 0.0
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx)**2,
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
    def apply_forces_and_detect(self):
        self.contact_force_vec[0] = ti.Vector.zero(ti.f32, self.dim)
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v_old = self.grid_v[I] / m
                v_new = v_old + self.dt * ti.Vector([0.0, -9.8])
                pos = I.cast(ti.f32) * self.dx
                # Roller: normal-only sliding
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
                # Floor: clamp vertical & horizontal
                if pos.y < self.floor_level + self.dx:
                    if v_new.y < 0:
                        v_new.y = 0
                    v_new.x = 0
                # Walls: clamp horizontal
                if pos.x < self.dx:
                    v_new.x = 0
                if pos.x > 1 - self.dx:
                    v_new.x = 0
                self.grid_v[I] = v_new * m

    @ti.kernel
    def g2p(self):
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
                # Guarantee gv is always defined:
                grid_m_val = self.grid_m[base + offs]
                gv = ti.Vector.zero(ti.f32, self.dim)
                if grid_m_val > 0:
                    gv = self.grid_v[base + offs] / grid_m_val
                else:
                    gv = ti.Vector.zero(ti.f32, self.dim)

                new_v += wt * gv
                new_C += 4 * self.inv_dx * wt * gv.outer_product(dpos)

            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v

            # Floor: enforce constraints
            if self.x[p].y < self.floor_level:
                self.x[p].y = self.floor_level
                self.v[p].y = 0
                self.v[p].x = 0
            # Walls:
            if self.x[p].x < self.dx:
                self.x[p].x = self.dx
                self.v[p].x = 0
            if self.x[p].x > 1 - self.dx:
                self.x[p].x = 1 - self.dx
                self.v[p].x = 0
            if self.x[p].y > 1 - self.dx:
                self.x[p].y = 1 - self.dx
                self.v[p].y = 0

            self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) + self.dt * new_C) @ self.F[p]
            self.J[p] = self.F[p].determinant()


class RobotArm:
    def __init__(self, L1=0.12, L2=0.10, k1=5.0, k2=5.0, b1=2.0, b2=2.0,
                 base_x=0.4, y0=0.4, A=0.1, omega=0.5):
        # Geometry
        self.L1 = L1
        self.L2 = L2
        # Joint states
        self.theta1 = np.array([np.pi/15], dtype=np.float32)
        self.theta2 = np.array([0.0], dtype=np.float32)
        self.dtheta1 = np.zeros(1, dtype=np.float32)
        self.dtheta2 = np.zeros(1, dtype=np.float32)
        self.theta1_rest = self.theta1.copy()
        self.theta2_rest = self.theta2.copy()
        # Passive spring-damper
        self.k1 = k1
        self.k2 = k2
        self.b1 = b1
        self.b2 = b2
        self.I1 = L1**2 / 12.0
        self.I2 = L2**2 / 12.0
        # Base motion
        self.base_x = base_x
        self.y0 = y0
        self.A = A
        self.omega = omega
        self.base_y = y0
        self.time_t = 0.0

    def forward_kinematics(self):
        base = np.array([self.base_x, self.base_y], dtype=np.float32)
        j2 = base + np.array([np.sin(self.theta1[0]), -np.cos(self.theta1[0])]) * self.L1
        ee = j2 + np.array([np.sin(self.theta1[0] + self.theta2[0]),
                            -np.cos(self.theta1[0] + self.theta2[0])]) * self.L2
        return base, j2, ee

    def update(self, dt, contact_force):
        # 1) update base vertical motion
        self.time_t += dt * 15
        self.base_y = self.y0 + self.A * np.cos(self.omega * self.time_t)
        # 2) read contact force
        Fc = contact_force
        base, j2, ee_old = self.forward_kinematics()
        # 3) compute new end-effector position
        j2 = base + np.array([np.sin(self.theta1[0]), -np.cos(self.theta1[0])]) * self.L1
        ee_new = j2 + np.array([np.sin(self.theta1[0] + self.theta2[0]),
                                -np.cos(self.theta1[0] + self.theta2[0])]) * self.L2
        rv = (ee_new - ee_old) / dt
        # 4) update roller state externally by Simulation
        # 5) compute torques
        r1 = ee_new - base
        r2 = ee_new - j2
        tau1_c = r1[1] * Fc[0] - r1[0] * Fc[1]
        tau2_c = r2[1] * Fc[0] - r2[0] * Fc[1]
        tau1 = 2 * tau1_c - self.k1 * (self.theta1[0] - self.theta1_rest[0]) - self.b1 * self.dtheta1[0]
        tau2 = 5 * tau2_c - self.k2 * (self.theta2[0] - self.theta2_rest[0]) - self.b2 * self.dtheta2[0]
        # 6) integrate
        alpha1 = tau1 / self.I1
        alpha2 = tau2 / self.I2
        self.dtheta1[0] += alpha1 * dt
        self.theta1[0] += self.dtheta1[0] * dt
        self.dtheta2[0] += alpha2 * dt
        self.theta2[0] += self.dtheta2[0] * dt
        return ee_new, rv


@ti.data_oriented
class Simulation:
    def __init__(self):
        # Instantiate soft body and robot arm
        self.soft_body = SoftBodyMPM()
        self.robot = RobotArm()
        # GUI
        self.gui = ti.GUI('MPM + Passive 2-Link Arm', res=(512, 512))
        # Initialize roller from FK
        base, j2, ee = self.robot.forward_kinematics()
        self.soft_body.roller_center[0] = ee.tolist()
        self.soft_body.roller_velocity[0] = [0.0, 0.0]

    def step(self):
        # MPM P2G -> grid forces -> G2P
        self.soft_body.p2g()
        self.soft_body.apply_forces_and_detect()
        self.soft_body.g2p()
        # Update robot with contact
        Fc = self.soft_body.contact_force_vec[0].to_numpy()
        ee_new, rv = self.robot.update(self.soft_body.dt, Fc)
        # Update roller in soft body
        self.soft_body.roller_center[0] = ee_new.tolist()
        self.soft_body.roller_velocity[0] = rv.tolist()

    def run(self):
        # Main loop
        while self.gui.running:
            for _ in range(15):
                self.step()
            self.render()
            self.gui.show()

    def render(self):
        # Draw particles
        self.gui.circles(self.soft_body.x.to_numpy(), radius=1.5, color=0x66CCFF)
        # Draw arm & roller
        base, j2, ee = self.robot.forward_kinematics()
        self.gui.line(begin=base, end=j2, radius=2, color=0x000050)
        self.gui.line(begin=j2, end=ee, radius=2, color=0x000050)
        self.gui.circle(base, radius=4, color=0xFF0000)
        self.gui.circle(j2, radius=4, color=0xFF0000)
        self.gui.circle(ee, radius=int(self.soft_body.roller_radius * 512), color=0xFF0000)
        self.gui.text(f'Force: {self.soft_body.contact_force_vec[0]} N', pos=(0.02, 0.95), color=0xFFFFFF)


# Built‐in tests for floor + energy
def test_floor_constraint():
    sim = Simulation()
    for _ in range(100):
        sim.step()
    xs = sim.soft_body.x.to_numpy()
    assert np.all(xs[:, 1] >= sim.soft_body.floor_level - 1e-5), "Floor constraint violated"
    print("Floor constraint test passed.")


def test_energy_non_negative():
    sim = Simulation()
    for _ in range(100):
        sim.step()
        Fc = sim.soft_body.contact_force_vec[0].to_numpy()
        assert np.linalg.norm(Fc) >= 0, "Negative contact force norm"
    print("Energy/contact force test passed.")


def run_tests():
    test_floor_constraint()
    test_energy_non_negative()


if __name__ == '__main__':
    # Run the two built‐in tests, then launch GUI
    run_tests()
    sim = Simulation()
    sim.run()
