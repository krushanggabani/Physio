import taichi as ti

@ti.data_oriented
class MPM2DSimulation:
    def __init__(self, n_particles=20000, n_grid=128):
        # Simulation parameters
        self.dim = 2
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.dx = 1.0 / n_grid
        self.inv_dx = float(n_grid)
        self.dt = 2e-4

        # Material properties
        self.p_rho = 1.0
        self.p_vol = (self.dx * 0.5)**2
        self.p_mass = self.p_vol * self.p_rho
        E = 5e3; nu = 0.2
        self.mu_0 = E / (2 * (1 + nu))
        self.lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

        # Roller
        self.roller_radius = 0.05
        self.v_desc = 0.5
        self.v_roll = 0.2

        # Floor
        self.floor_level = 0.0
        self.floor_friction = 0.4

        # Taichi fields
        self.x = ti.Vector.field(self.dim, ti.f32, shape=self.n_particles)
        self.v = ti.Vector.field(self.dim, ti.f32, shape=self.n_particles)
        self.F = ti.Matrix.field(self.dim, self.dim, ti.f32, shape=self.n_particles)
        self.C = ti.Matrix.field(self.dim, self.dim, ti.f32, shape=self.n_particles)
        self.J = ti.field(ti.f32, shape=self.n_particles)
        self.grid_v = ti.Vector.field(self.dim, ti.f32, shape=(n_grid, n_grid))
        self.grid_m = ti.field(ti.f32, shape=(n_grid, n_grid))
        self.roller_center = ti.Vector.field(self.dim, ti.f32, shape=())
        self.state = ti.field(ti.i32, shape=())
        self.contact_height = ti.field(ti.f32, shape=())
        self.contact_force = ti.field(ti.f32, shape=())

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
            self.x[p] = [ti.random() * 0.8 + 0.1, ti.random() * 0.2]
            self.v[p] = [0.0, 0.0]
            self.F[p] = ti.Matrix.identity(ti.f32, 2)
            self.J[p] = 1.0
            self.C[p] = ti.Matrix.zero(ti.f32, 2, 2)
        self.state[None] = 0
        self.contact_height[None] = 0.0
        self.contact_force[None] = 0.0
        self.roller_center[None] = ti.Vector([0.5, 0.5])

    @ti.kernel
    def p2g(self):
        for I in ti.grouped(self.grid_m):
            self.grid_v[I] = ti.Vector.zero(ti.f32, 2)
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
                offset = ti.Vector([i, j])
                dpos = (offset.cast(ti.f32) - fx) * self.dx
                weight = w[i].x * w[j].y
                self.grid_v[base + offset] += weight * (self.p_mass * self.v[p] + affine @ dpos)
                self.grid_m[base + offset] += weight * self.p_mass

    @ti.kernel
    def apply_forces(self):
        self.contact_force[None] = 0.0
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0:
                v_old = self.grid_v[I] / m
                v_new = v_old + self.dt * ti.Vector([0.0, -9.8])
                pos = I.cast(ti.f32) * self.dx
                # Roller contact (normal-only)
                if (pos - self.roller_center[None]).norm() < self.roller_radius:
                    if self.state[None] == 0:
                        v_target = ti.Vector([0.0, -self.v_desc])
                    else:
                        v_target = ti.Vector([self.v_roll, 0.0])
                    rel = pos - self.roller_center[None]
                    n = rel.normalized()
                    v_new = (v_old - n * (n.dot(v_old))) + n * (n.dot(v_target))
                    f_imp = m * (v_new - v_old) / self.dt
                    self.contact_force[None] += f_imp.norm()
                # Floor
                if pos.y < self.floor_level + self.dx:
                    if v_new.y < 0:
                        v_new.y = 0
                    v_new.x *= (1 - self.floor_friction)
                # Walls
                if pos.x < self.dx or pos.x > 1 - self.dx:
                    v_new.x = 0
                self.grid_v[I] = v_new * m
        if self.state[None] == 0 and self.contact_force[None] >= 2.0:
            self.state[None] = 1
            self.contact_height[None] = self.roller_center[None].y

    @ti.kernel
    def g2p(self):
        for p in range(self.n_particles):
            Xp = self.x[p] * self.inv_dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base.cast(ti.f32)
            w = [0.5 * (1.5 - fx)**2,
                 0.75 - (fx - 1.0)**2,
                 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(ti.f32, 2)
            new_C = ti.Matrix.zero(ti.f32, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(ti.f32) - fx) * self.dx
                weight = w[i].x * w[j].y
                g_v = self.grid_v[base + offset] / self.grid_m[base + offset]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v
            if self.x[p].y < self.floor_level:
                self.x[p].y = self.floor_level
                self.v[p].y = 0
                self.v[p].x *= (1 - self.floor_friction)
            if self.x[p].x < self.dx:
                self.x[p].x = self.dx
                self.v[p].x = 0
            if self.x[p].x > 1 - self.dx:
                self.x[p].x = 1 - self.dx
                self.v[p].x = 0
            self.F[p] = (ti.Matrix.identity(ti.f32, 2) + self.dt * self.C[p]) @ self.F[p]
            self.J[p] = self.F[p].determinant()

    def step(self):
        self.p2g()
        self.apply_forces()
        # update roller position
        c = self.roller_center[None]
        if self.state[None] == 0:
            c.y -= self.v_desc * self.dt
        else:
            c.x += self.v_roll * self.dt
            c.y = self.contact_height[None]
        self.roller_center[None] = c
        self.g2p()

    def run(self):
        gui = ti.GUI('MPM OOP 2D', res=(512, 512))
        while gui.running:
            for _ in range(20):
                self.step()
            gui.circles(self.x.to_numpy(), radius=1.5, color=0x66CCFF)
            gui.circle(self.roller_center[None].to_numpy(), radius=int(self.roller_radius * 512), color=0xFF0000)
            gui.text(f'Phase: {"Rolling" if self.state[None] else "Descending"}   '
                     f'Force: {self.contact_force[None]:.2f} N', pos=(0.02, 0.95), color=0xFFFFFF)
            gui.show()
