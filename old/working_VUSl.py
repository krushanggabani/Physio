import taichi as ti
ti.init(arch=ti.gpu)

# ─── Simulation Parameters ────────────────────────────────────────────────
dim = 2
n_particles = 8192
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2e-4

# ─── Material Parameters ──────────────────────────────────────────────────
E = 400      # Young's modulus
nu = 0.2     # Poisson's ratio
mu_0 = E / (2 * (1 + nu))                     # Shear modulus
la_0 = E * nu / ((1 + nu) * (1 - 2 * nu))     # Lamé's first parameter
eta = 0.2     # Viscous damping coefficient

p_vol = (dx * 0.5)**2
p_rho = 1
p_mass = p_vol * p_rho

# ─── Fields ───────────────────────────────────────────────────────────────
x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)      # position
v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)      # velocity
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles) # affine velocity field
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles) # deformation gradient

grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

@ti.kernel
def initialize():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.2 + 0.3, ti.random() * 0.2 + 0.3]
        v[i] = [0, 0]
        F[i] = ti.Matrix.identity(ti.f32, dim)
        C[i] = ti.Matrix.zero(ti.f32, dim, dim)

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = ti.Vector.zero(ti.f32, dim)
        grid_m[i, j] = 0

    # ─── P2G ──────────────────────────────────────────────────────────────
    for p in x:
        Xp = x[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1)**2,
             0.5 * (fx - 0.5)**2]

        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]
        J = F[p].determinant()
        r, s = ti.polar_decompose(F[p])

        # Neo-Hookean elastic stress
        sigma_elastic = 2 * mu_0 * (F[p] - r) @ F[p].transpose() + \
                        ti.Matrix.identity(ti.f32, dim) * la_0 * J * (J - 1)

        # Kelvin–Voigt viscous stress (strain-rate from C)
        sigma_viscous = eta * (C[p] + C[p].transpose())

        # Total stress
        stress = (sigma_elastic + sigma_viscous)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_idx = base + offset

                grid_v[grid_idx] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[grid_idx] += weight * p_mass

    # ─── Grid Update ──────────────────────────────────────────────────────
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j][1] -= dt * 9.8  # gravity

            # Boundary conditions
            if i < 3 or i > n_grid - 3:
                grid_v[i, j][0] = 0
            if j < 3 or j > n_grid - 3:
                grid_v[i, j][1] = 0

    # ─── G2P ──────────────────────────────────────────────────────────────
    for p in x:
        Xp = x[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1)**2,
             0.5 * (fx - 0.5)**2]

        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                g_v = grid_v[base + offset]

                new_v += weight * g_v
                # Use the vector method for outer product:
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]

# ─── Run ──────────────────────────────────────────────────────────────────
initialize()
gui = ti.GUI("v-USL MPM with Kelvin–Voigt", res=512)
while gui.running:
    for _ in range(25):
        substep()
    gui.circles(x.to_numpy(), radius=1.5, color=0x66ccff)
    gui.show()
