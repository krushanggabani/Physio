import taichi as ti

ti.init(arch=ti.cpu)

# — window —
window_size = 800
gui = ti.GUI("Viscoelastic MPM", (window_size, window_size))

# — MPM grid parameters —
n      = 80
dx     = 1.0 / n
inv_dx = float(n)
dt     = 2e-4
substeps_per_frame = int(1e-3 / dt)

# — SLS viscoelastic parameters —
E1  = 5e5       # parallel spring modulus
E2  = 5e4       # Maxwell spring modulus
eta = 1e3       # dashpot viscosity
tau = eta / E2  # relaxation time

# — particle patch —
center       = ti.Vector([0.5, 0.05], ti.f32)
width, height, spacing = 0.5, 0.25, 0.01
nx = max(1, int(width  / spacing) + 1)
ny = max(1, int(height / spacing) + 1)
num_particles = nx * ny

# — fields —
x         = ti.Vector.field(2, ti.f32, num_particles)
v         = ti.Vector.field(2, ti.f32, num_particles)
F         = ti.Matrix.field(2, 2, ti.f32, num_particles)
C         = ti.Matrix.field(2, 2, ti.f32, num_particles)
sigma_v   = ti.Matrix.field(2, 2, ti.f32, num_particles)
grid_v    = ti.Vector.field(2, ti.f32, (n+1, n+1))
grid_m    = ti.field(ti.f32,   (n+1, n+1))

@ti.kernel
def seed():
    I = ti.Matrix.identity(ti.f32,2)
    for p in range(num_particles):
        i = p // ny
        j = p % ny
        x[p]       = center + ti.Vector([-width*0.5 + i*spacing, j*spacing])
        v[p]       = ti.Vector([0.0, 0.0])
        v[p]       = ti.Vector([0.0, 0.0])
        F[p]       = I
        C[p]       = ti.Matrix.zero(ti.f32, 2, 2)
        sigma_v[p] = ti.Matrix.zero(ti.f32, 2, 2)


@ti.kernel
def substep():
    I2 = ti.Matrix.identity(ti.f32,2)
    # reset grid
    for I in ti.grouped(grid_m):
        grid_m[I] = 0.0
        grid_v[I] = ti.Vector([0.0, 0.0])

    # P2G
    for p in range(num_particles):
        
        Xp = x[p] * inv_dx
        base = ti.floor(Xp - 0.5).cast(int)
        fx = Xp - base.cast(ti.f32)

        # quadratic B-spline weights
        w0 = 0.5 * (1.5 - fx)**2
        w1 = 0.75 - (fx - 1.0)**2
        w2 = 0.5 * (fx - 0.5)**2
        w = [w0, w1, w2]

        # viscoelastic constitutive update
        eps_e   = F[p] - I2               # elastic strain
        sigma_e = E1 * eps_e             # spring-1 stress
        # Maxwell branch update
        sigma_v[p] = sigma_v[p] + dt * (E2 * eps_e - sigma_v[p]) / tau
        sigma_total = sigma_e + sigma_v[p]

        # first Piola stress (small-strain approx)
        P = sigma_total
        affine = P @ F[p].transpose() 

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = (ti.Vector([i, j]).cast(ti.f32) - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + ti.Vector([i, j])] += weight * (v[p] + affine @ offset)
            grid_m[base + ti.Vector([i, j])] += weight

    # grid update
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] = grid_v[I] / grid_m[I]
            # simple boundary conditions
            if I.x < 2 or I.x > n - 2 or I.y < 2:
                grid_v[I] = ti.Vector([0.0, 0.0])

    # G2P
    for p in range(num_particles):
        Xp = x[p] * inv_dx
        base = ti.floor(Xp - 0.5).cast(int)
        fx = Xp - base.cast(ti.f32)

        w0 = 0.5 * (1.5 - fx)**2
        w1 = 0.75 - (fx - 1.0)**2
        w2 = 0.5 * (fx - 0.5)**2
        w = [w0, w1, w2]

        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix.zero(ti.f32,2,2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = (ti.Vector([i, j]).cast(ti.f32) - fx) * dx
            weight = w[i].x * w[j].y
            gv = grid_v[base + ti.Vector([i, j])]
            new_v += weight * gv
            new_C += 4 * weight * ti.math.cross(gv, offset) / (dx * dx)

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]
        F[p] = (I2 + dt * C[p]) @ F[p]

# — main —
seed()
for _ in range(1000000):
    for __ in range(substeps_per_frame):
        substep()
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=2, color=0xED553B)
    gui.show()
