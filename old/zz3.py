import taichi as ti
import numpy as np

# Initialize Taichi
ti.init(arch=ti.gpu)

# ─── Simulation parameters ────────────────────────────────────────────────
dim         = 2
n_particles = 20000
n_grid      = 128
dx          = 1 / n_grid
inv_dx      = float(n_grid)
dt          = 2e-4  # time step

# Particle properties
p_rho  = 1.0
p_vol  = (dx * 0.5)**2
p_mass = p_vol * p_rho

# Neo‐Hookean material constants
E       = 5e3
nu      = 0.2
mu_0    = E / (2*(1 + nu))
lambda_0 = E * nu / ((1 + nu)*(1 - 2*nu))

# Roller (end‐effector) geometry & motion
roller_radius = 0.05
roller_center = ti.Vector.field(dim, dtype=ti.f32, shape=())

# Roller state: 0 = descending, 1 = rolling
state          = ti.field(ti.i32, shape=())
contact_height = ti.field(ti.f32, shape=())

# Accumulated contact force (N)
contact_force = ti.field(ti.f32, shape=())

# Speeds (units per second)
v_desc = 0.5
v_roll = 0.2

# ─── Taichi fields ────────────────────────────────────────────────────────
x      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # positions
v      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)  # velocities
F      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
C      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
J      = ti.field(dtype=ti.f32, shape=n_particles)             # det(F)

grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

# ─── Utility: Neo‐Hookean stress ───────────────────────────────────────────
@ti.func
def neo_hookean_stress(F_i):
    J_det  = F_i.determinant()
    FinvT  = F_i.inverse().transpose()
    return mu_0 * (F_i - FinvT) + lambda_0 * ti.log(J_det) * FinvT

# ─── Initialization ────────────────────────────────────────────────────────
@ti.kernel
def init_particles():
    for p in range(n_particles):
        # initialize a block of particles on the “floor”
        x[p] = [0.3 + ti.random()*0.4, 0.0 + ti.random()*0.2]
        v[p] = [0.0, 0.0]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        J[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)

    # roller starts above
    state[None]          = 0
    contact_height[None] = 0.0
    roller_center[None]  = ti.Vector([0.5, 0.50])

    # reset contact force
    contact_force[None]  = 0.0
    
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
        # quadratic B‐spline weights
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]

        # compute stress and affine momentum
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * neo_hookean_stress(F[p])
        affine = stress + p_mass * C[p]

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos   = (offset.cast(ti.f32) - fx) * dx
                weight = w[i].x * w[j].y
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

# ─── Detect first contact of roller with soft body ─────────────────────────
@ti.kernel
def detect_contact():
    for I in ti.grouped(grid_m):
        if state[None] == 0:
            pos = I.cast(ti.f32) * dx
            if (pos - roller_center[None]).norm() < 0.5*roller_radius and grid_m[I] > 0:
                state[None] = 1
                contact_height[None] = roller_center[None].y

# ─── Update roller position (Python side) ─────────────────────────────────
def update_roller_position():
    center = roller_center[None]
    if state[None] == 0:
        # Phase 0: descend vertically
        center.y -= v_desc * dt
    else:
        # Phase 1: roll horizontally at fixed height
        center.x += v_roll * dt
        center.y = contact_height[None]
    roller_center[None] = center

# ─── Apply grid‐based forces & roller contact ──────────────────────────────
@ti.kernel
def apply_grid_forces():
    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 0:
            v_I = grid_v[I] / m
            # gravity
            v_I += dt * ti.Vector([0.0, -9.8])
            # roller contact: clamp node velocity to roller motion
            pos = I.cast(ti.f32) * dx
            if (pos - roller_center[None]).norm() < roller_radius:
                if state[None] == 0:
                    v_I = ti.Vector([0.0, -v_desc])
                else:
                    v_I = ti.Vector([v_roll, 0.0])
            # simple boundary conditions
            if pos.x < dx or pos.x > 1 - dx: v_I.x = 0
            if pos.y < dx:                    v_I.y = 0
            grid_v[I] = v_I * m

# ─── Grid→Particle (G2P) ──────────────────────────────────────────────────
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
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos   = (offset.cast(ti.f32) - fx) * dx
                weight = w[i].x * w[j].y
                g_v    = grid_v[base + offset] / grid_m[base + offset]
                new_v  += weight * g_v
                new_C  += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]
        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]
        J[p] = F[p].determinant()

# ─── Main simulation loop ──────────────────────────────────────────────────
init_particles()
gui = ti.GUI('MPM Massage', res=(512, 512))

while gui.running:
    for _ in range(20):  # sub‐steps for stability
        p2g()
        detect_contact()
        update_roller_position()
        apply_grid_forces()
        g2p()

    # render soft body particles
    gui.circles(x.to_numpy(), radius=1.5, color=0x66CCFF)
    # render roller
    gui.circle(roller_center[None].to_numpy(),
               radius=int(roller_radius * 512),
               color=0xFF0000)
    gui.show()
