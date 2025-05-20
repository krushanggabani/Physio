import taichi as ti
import numpy as np

# Initialize Taichi GPU
ti.init(arch=ti.gpu)

# Simulation resolution
quality = 1
dim   =2
n_particles = 10000 * quality ** 2
n_grid = 128 * quality

dx = 1 / n_grid
inv_dx = float(n_grid)
dt = 1e-4 / quality

# Particle properties
p_vol = (dx * 0.5) ** 2
p_rho = 1
p_mass = p_vol * p_rho

# Material parameters (neo-Hookean)
E, nu = 0.1e4, 0.2
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# Fields: particles
x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
F = ti.Matrix.field(2, 2, float, n_particles)

# Fields: grid
grid_v = ti.Vector.field(2, float, (n_grid, n_grid))  # node momentum/velocity
grid_m = ti.field(float, (n_grid, n_grid))           # node mass
grid_A = ti.field(int, (n_grid, n_grid))             # active under rigid box

# Gravity and rigid-box parameters
gravity = 10
t = ti.field(float, shape=())                 # simulation time
r_v = 1  # rigid speed (units per second)
# Rigid-box velocity vector (downwards)
r_vel = ti.Vector.field(2, float, shape=())
# Rigid-box bounds: min and max corners
to_box = ti.Vector.field(2, float, shape=())  # lower-left corner
from_box = ti.Vector.field(2, float, shape=()) # upper-right corner


amplitude = 0.2
omega     = 10

@ti.kernel
def initialize():
    # Initialize particles in a blob
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.2]
        v[i] = [0.0, 0.0]
        F[i] = ti.Matrix.identity(float, 2)
        C[i] = ti.Matrix.zero(float, 2, 2)

    # Rigid-box initial velocity (downwards)
    r_vel[None] = [0.0, 0.0]
    # Rigid-box initial corners (axis-aligned)
    to_box[None] = [0.45, 0.5]   # lower-left corner
    from_box[None] = [0.55, 0.7] # upper-right corner

@ti.kernel
def substep():
    # Advance rigid-box position
    t[None] += dt
    v_y = -amplitude*omega *(ti.sin(omega * t[None]))
    r_vel[None]   = ti.Vector([0.0, v_y])
    to_box[None] += r_vel[None] * dt
    from_box[None] += r_vel[None] * dt

    # Build active-mask: mark grid nodes inside the box
    for i, j in grid_A:
        pos = ti.Vector([i, j]) * dx
        inside = (pos[0] >= to_box[None][0] and pos[0] <= from_box[None][0]
               and pos[1] >= to_box[None][1] and pos[1] <= from_box[None][1])
        grid_A[i, j] = 1 if inside else 0

    # Clear grid fields
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0

    # P2G: scatter particles to grid
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline weights
        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ]
        # Update deformation gradient
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Compute stress (compressible neo-Hookean)
        h = 0.5
        mu, la = mu_0 * h, lambda_0 * h
        U, sigma, V = ti.svd(F[p])
        J = sigma[0, 0] * sigma[1, 1]
        P = (2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose()
             + ti.Matrix.identity(float, 2) * la * J * (J - 1))
        stress = -dt * p_vol * 4 * inv_dx * inv_dx * P
        affine = stress + p_mass * C[p]

        # Scatter mass & momentum
        for ii, jj in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([ii, jj])
            dpos = (offs.cast(float) - fx) * dx
            weight = w[ii][0] * w[jj][1]
            grid_v[base + offs] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offs] += weight * p_mass

    # Inject rigid-box velocity into grid nodes inside it
    for i, j in grid_m:
        if grid_A[i, j] == 1 and grid_m[i, j] > 0:
            grid_v[i, j] = r_vel[None] * grid_m[i, j]

    # Grid update: momentum → velocity, apply gravity & BCs
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_v[i, j] / grid_m[i, j]
            grid_v[i, j][1] -= dt * gravity
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0

    # G2P: update particles from grid
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for ii, jj in ti.static(ti.ndrange(3, 3)):
            g_v = grid_v[base + ti.Vector([ii, jj])]
            if grid_A[base[0] + ii, base[1] + jj] == 1:
                g_v = r_vel[None]
            dpos = ti.Vector([ii, jj]).cast(float) - fx
            weight = w[ii][0] * w[jj][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]

# ----- Setup & main loop -----
initialize()

gui = ti.GUI("MPM with Rigid Box", res=800, background_color=0x112F41)
frame = 0
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    for _ in range(int(5e-3 // dt)):
        substep()

    # Draw particles
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    # Draw rigid box with topleft/bottomright
    p1 = to_box[None].to_numpy()
    p2 = from_box[None].to_numpy()
    gui.rect(topleft=p1, bottomright=p2, color=0xFF0000)

    gui.show()
    frame += 1