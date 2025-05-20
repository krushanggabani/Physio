import taichi as ti
import numpy as np

# ─── Init Taichi ──────────────────────────────────────────────────────────
# Ensure Vulkan or CPU backend
ti.init(arch=ti.vulkan)

# ─── Constants ─────────────────────────────────────────────────────────────
PI = 3.141592653589793

# ─── Simulation parameters ────────────────────────────────────────────────
dim         = 2
n_particles = 20000
n_grid      = 128
dx          = 1 / n_grid
inv_dx      = float(n_grid)
dt          = 2e-4

# ─── Particle properties ──────────────────────────────────────────────────
p_rho  = 1.0
p_vol  = (dx * 0.5)**2
p_mass = p_vol * p_rho

# ─── Material (Neo-Hookean) ──────────────────────────────────────────────
E        = 5e3
nu       = 0.2
mu_0     = E / (2*(1 + nu))
lambda_0 = E * nu / ((1 + nu)*(1 - 2*nu))

# ─── Roller (end-effector) ────────────────────────────────────────────────
roller_radius = 0.05
roller_center = ti.Vector.field(dim, dtype=ti.f32, shape=())

# ─── Roller phases & force detection ───────────────────────────────────────
state          = ti.field(ti.i32, shape=())   # 0=descend, 1=roll
contact_height = ti.field(ti.f32, shape=())
contact_force  = ti.field(ti.f32, shape=())

# ─── Speeds ───────────────────────────────────────────────────────────────
v_desc = 0.5  # vertical descent m/s
v_roll = 0.2  # horizontal roll m/s

# ─── Floor parameters ────────────────────────────────────────────────────
floor_level    = 0.0
floor_friction = 0.4

# ─── Semi-circle soft body parameters ─────────────────────────────────────
half_radius   = 0.2
soft_center_x = 0.5

# ─── Taichi fields ────────────────────────────────────────────────────────
x      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
F      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
C      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
J      = ti.field(dtype=ti.f32, shape=n_particles)

grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

# ─── Neo-Hookean stress ───────────────────────────────────────────────────
@ti.func
def neo_hookean_stress(F_i):
    J_det = F_i.determinant()
    FinvT = F_i.inverse().transpose()
    return mu_0 * (F_i - FinvT) + lambda_0 * ti.log(J_det) * FinvT

# ─── Initialize semi-circular soft-body & roller ───────────────────────────
@ti.kernel
def init_particles():
    for p in range(n_particles):
        u     = ti.random()
        r     = half_radius * ti.sqrt(u)
        theta = ti.random() * PI
        x[p]  = ti.Vector([soft_center_x + r * ti.cos(theta),
                           floor_level    + r * ti.sin(theta)])
        v[p] = ti.Vector([0.0, 0.0])
        F[p] = ti.Matrix.identity(ti.f32, dim)
        J[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)
    state[None]          = 0
    contact_height[None] = 0.0
    contact_force[None]  = 0.0
    roller_center[None]  = ti.Vector([soft_center_x, 0.5])

# ─── Particle to Grid (P2G) ────────────────────────────────────────────────
@ti.kernel
def p2g():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, dim)
        grid_m[I] = 0.0
    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx)**2,
             0.75 - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * neo_hookean_stress(F[p])
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos   = (offset.cast(ti.f32) - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

# ─── Grid forces, roller (normal-only), floor, walls, force detect ─────────
@ti.kernel
def apply_grid_forces_and_detect():
    contact_force[None] = 0.0
    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 0:
            v_old = grid_v[I] / m
            v_new = v_old + dt * ti.Vector([0.0, -9.8])
            pos   = I.cast(ti.f32) * dx

            # Roller contact (normal only)
            if (pos - roller_center[None]).norm() < roller_radius:
                v_target = ti.Vector.zero(ti.f32, dim)
                if state[None] == 0:
                    v_target = ti.Vector([0.0, -v_desc])
                else:
                    v_target = ti.Vector([v_roll, 0.0])
                n       = (pos - roller_center[None]).normalized()
                v_norm  = n * (n.dot(v_target))
                v_tang  = v_old - n * (n.dot(v_old))
                v_new   = v_tang + v_norm
                f_imp   = m * (v_new - v_old) / dt
                contact_force[None] += f_imp.norm()

            # Floor contact
            if pos.y < floor_level + dx:
                if v_new.y < 0: v_new.y = 0
                v_new.x *= (1 - floor_friction)

            # Walls on grid
            if pos.x < dx:      v_new.x = 0
            if pos.x > 1-dx:    v_new.x = 0

            grid_v[I] = v_new * m

    if state[None] == 0 and contact_force[None] >= 2.0:
        state[None] = 1
        contact_height[None] = roller_center[None].y

# ─── Update roller position ────────────────────────────────────────────────
def update_roller_position():
    c = roller_center[None]
    if state[None] == 0:
        c.y -= v_desc * dt
    else:
        c.x += v_roll * dt
        c.y = contact_height[None]
    roller_center[None] = c

# ─── Grid to Particle (G2P) with boundaries ────────────────────────────────
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
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos   = (offset.cast(ti.f32) - fx) * dx
            weight = w[i].x * w[j].y
            g_v    = grid_v[base + offset] / grid_m[base + offset]
            new_v  += weight * g_v
            new_C  += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v

        # Particle-level boundaries
        if x[p].y < floor_level:
            x[p].y = floor_level; v[p].y = 0; v[p].x *= (1 - floor_friction)
        if x[p].x < dx:
            x[p].x = dx; v[p].x = 0
        if x[p].x > 1 - dx:
            x[p].x = 1 - dx; v[p].x = 0
        if x[p].y > 1 - dx:
            x[p].y = 1 - dx; v[p].y = 0

        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * new_C) @ F[p]
        J[p] = F[p].determinant()

# ─── Main Loop ────────────────────────────────────────────────────────────
init_particles()
gui = ti.GUI('MPM Massage w/ Half-Circle', res=(512, 512))

while gui.running:
    for _ in range(20):
        p2g()
        apply_grid_forces_and_detect()
        update_roller_position()
        g2p()
    gui.circles(x.to_numpy(), radius=1.5, color=0x66CCFF)
    gui.circle(roller_center[None].to_numpy(),
               radius=int(roller_radius * 512), color=0xFF0000)
    gui.text(f'Phase: {"Rolling" if state[None] else "Descending"}  '
             f'Force: {contact_force[None]:.2f} N',
             pos=(0.02, 0.95), color=0xFFFFFF)
    gui.show()
