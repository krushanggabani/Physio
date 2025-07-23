import taichi as ti
import numpy as np


# approved by professor.
# ─── Taichi init ───────────────────────────────────────────────────────────
ti.init(arch=ti.vulkan)

# ─── Simulation parameters ─────────────────────────────────────────────────
dim         = 2
n_particles = 20000
n_grid      = 128
dx          = 1.0 / n_grid
inv_dx      = float(n_grid)
dt          = 2e-4

# ─── Material (Neo‐Hookean) ────────────────────────────────────────────────
p_rho   = 1.0
p_vol   = (dx * 0.5)**2
p_mass  = p_rho * p_vol
E       = 5.65e4
nu      = 0.185
mu_0    = E / (2 * (1 + nu))
lambda_0= E * nu / ((1 + nu) * (1 - 2 * nu))

eta = 0.5



# ─── Floor & domain boundaries ─────────────────────────────────────────────
floor_level    = 0.0
floor_friction = 0.4

# ─── Single‐hand 2‐link arm params ─────────────────────────────────────────
L1, L2       = 0.12, 0.10
theta1       = np.array([0.0], dtype=np.float32)
theta2       = np.array([0.0],     dtype=np.float32)
dtheta1      = np.zeros(1, dtype=np.float32)
dtheta2      = np.zeros(1, dtype=np.float32)
theta1_rest  = theta1.copy()
theta2_rest  = theta2.copy()

k1, k2       = 10,10   # compliant springs
b1, b2       = 0.2, 0.25   # damping
I1           = L1**2 / 12.0
I2           = L2**2 / 12.0

# ─── Moving base (x fixed at 0.5, y sinusoidal) ────────────────────────────
base_x       = 0.5
y0           = 0.4
A            = 0.1
ω            = 0.5
base_y       = y0
time_t       = 0.0

# ─── Roller & contact fields (single hand) ─────────────────────────────────
roller_radius     = 0.025
roller_center     = ti.Vector.field(dim, dtype=ti.f32, shape=1)
roller_velocity   = ti.Vector.field(dim, dtype=ti.f32, shape=1)
contact_force_vec = ti.Vector.field(dim, dtype=ti.f32, shape=1)

# ─── MPM fields ────────────────────────────────────────────────────────────
x      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
F      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
C      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
J      = ti.field(dtype=ti.f32, shape=n_particles)
grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))


# ─── Neo‐Hookean stress ────────────────────────────────────────────────────
@ti.func
def neo_hookean_stress(F_i):
    J_det = F_i.determinant()
    FinvT = F_i.inverse().transpose()
    return mu_0 * (F_i - FinvT) + lambda_0 * ti.log(J_det) * FinvT


PI = 3.141592653589793
half_radius   = 0.2
soft_center_x = 0.5


# ─── Initialize particles + place roller via FK ────────────────────────────
@ti.kernel
def init_mpm():
    for p in range(n_particles):
        # x[p] = [0.3 + ti.random() * 0.4, ti.random() * 0.2]
        u     = ti.random()
        r     = half_radius * ti.sqrt(u)
        theta = ti.random() * PI
        x[p]  = ti.Vector([soft_center_x + r * ti.cos(theta),
                           floor_level    + r * ti.sin(theta)])
        v[p] = [0.0, 0.0]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        J[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)
    # place roller at hand’s FK position
    base = ti.Vector([base_x, base_y])
    j2   = base + ti.Vector([ti.sin(theta1[0]), -ti.cos(theta1[0])]) * L1
    ee   = j2   + ti.Vector([ti.sin(theta1[0] + theta2[0]),
                             -ti.cos(theta1[0] + theta2[0])]) * L2
    roller_center[0]   = ee
    roller_velocity[0] = ti.Vector.zero(ti.f32, dim)
    contact_force_vec[0] = ti.Vector.zero(ti.f32, dim)


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
        w    = [0.5 * (1.5 - fx)**2,
                0.75 - (fx - 1.0)**2,
                0.5 * (fx - 0.5)**2]
        

        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]
        J = F[p].determinant()
        r, s = ti.polar_decompose(F[p])

        # Neo-Hookean elastic stress
        sigma_elastic = 2 * mu_0 * (F[p] - r) @ F[p].transpose() + \
                        ti.Matrix.identity(ti.f32, dim) * lambda_0 * J * (J - 1)

        # Kelvin–Voigt viscous stress (strain-rate from C)
        sigma_viscous = eta * (C[p] + C[p].transpose())

        # Total stress
        stress = (sigma_elastic + sigma_viscous)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]


        # stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * neo_hookean_stress(F[p])
        # affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([i, j])
            dpos = (offs.cast(ti.f32) - fx) * dx
            wt   = w[i].x * w[j].y
            grid_v[base + offs] += wt * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offs] += wt * p_mass


# ─── Grid forces & contact detect (normal‐only sliding) ────────────────────
@ti.kernel
def apply_grid_forces_and_detect():
    contact_force_vec[0] = ti.Vector.zero(ti.f32, dim)
    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 0:
            v_old = grid_v[I] / m
            v_new = v_old + dt * ti.Vector([0.0, -9.8])  # gravity
            pos   = I.cast(ti.f32) * dx

            # Roller: enforce only normal component
            rel = pos - roller_center[0]
            if rel.norm() < roller_radius:
                rv     = roller_velocity[0]
                n      = rel.normalized()
                v_norm = n * n.dot(rv)
                v_tan  = v_old - n * (n.dot(v_old))
                v_new  = v_tan + v_norm
                delta_v= v_new - v_old
                f_imp  = m * delta_v / dt
                contact_force_vec[0] += f_imp

            # Floor: clamp vertical AND horizontal for nodes touching floor
            if pos.y < floor_level + dx:
                if v_new.y < 0:
                    v_new.y = 0
                v_new.x = 0

            # Walls: clamp horizontal
            if pos.x < dx:     v_new.x = 0
            if pos.x > 1 - dx: v_new.x = 0

            grid_v[I] = v_new * m


# ─── Grid→Particle (G2P) w/ floor & wall clamp ────────────────────────────
@ti.kernel
def g2p():
    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx   = Xp - base.cast(ti.f32)
        w    = [0.5 * (1.5 - fx)**2,
                0.75 - (fx - 1.0)**2,
                0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offs   = ti.Vector([i, j])
            dpos   = (offs.cast(ti.f32) - fx) * dx
            wt     = w[i].x * w[j].y
            gv     = grid_v[base + offs] / grid_m[base + offs]
            new_v += wt * gv
            new_C += 4 * inv_dx * wt * gv.outer_product(dpos)

        # Update particle velocity and position
        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v

        # Floor: if particle touches floor, clamp vertical AND horizontal
        if x[p].y < floor_level:
            x[p].y = floor_level
            v[p].y = 0
            v[p].x = 0

        # Walls: clamp horizontal
        if x[p].x < dx:
            x[p].x = dx
            v[p].x = 0
        if x[p].x > 1 - dx:
            x[p].x = 1 - dx
            v[p].x = 0
        if x[p].y > 1 - dx:
            x[p].y = 1 - dx
            v[p].y = 0

        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * new_C) @ F[p]
        J[p] = F[p].determinant()


# ─── Update moving base & single passive arm dynamics ───────────────────────
def update_base_and_arm():
    global time_t, base_y, theta1, theta2, dtheta1, dtheta2

    # 1) Advance time and update vertical base_y
    time_t += dt * 15
    base_y = y0 + A * np.cos(ω * time_t)

    # 2) Read contact force (2D) from MPM kernel
    Fc   = contact_force_vec[0].to_numpy()
    base = np.array([base_x, base_y], dtype=np.float32)

    # 3) Forward kinematics (0 rad = downward)
    j2     = base + np.array([np.sin(theta1[0]), -np.cos(theta1[0])]) * L1
    ee_old = roller_center[0].to_numpy()
    ee_new = j2   + np.array([np.sin(theta1[0] + theta2[0]),
                               -np.cos(theta1[0] + theta2[0])]) * L2
    rv     = (ee_new - ee_old) / dt

    # 4) Update roller’s position & velocity
    roller_center[0]   = ee_new.tolist()
    roller_velocity[0] = rv.tolist()

    # 5) Compute torques via 2D cross‐product + passive spring‐damper
    r1     = ee_new - base       # lever arm from shoulder
    r2     = ee_new - j2         # lever arm from elbow
    tau1_c = r1[1] * Fc[0] - r1[0] * Fc[1]
    # Flip sign so elbow opens outward rather than inward:
    tau2_c = r2[1] * Fc[0] - r2[0] * Fc[1]

    tau1   = tau1_c - k1 * (theta1[0] - theta1_rest[0]) - b1 * dtheta1[0]
    tau2   = tau2_c - k2 * (theta2[0] - theta2_rest[0]) - b2 * dtheta2[0]

    # 6) Integrate joint accelerations (semi‐implicit Euler)
    alpha1      = tau1 / I1
    alpha2      = tau2 / I2
    dtheta1[0] += alpha1 * dt
    theta1[0]  += dtheta1[0] * dt
    dtheta2[0] += alpha2 * dt
    theta2[0]  += dtheta2[0] * dt


# ─── Main loop + GUI ──────────────────────────────────────────────────────
init_mpm()
gui = ti.GUI('MPM + Single Passive Arm (Attached & Outward)', res=(512, 512))

while gui.running:
    for _ in range(15):
        p2g()
        apply_grid_forces_and_detect()
        g2p()
        update_base_and_arm()

    # Draw soft body particles
    gui.circles(x.to_numpy(), radius=1.5, color=0x66CCFF)

    # Draw the single arm & roller
    base_pt = np.array([base_x, base_y], dtype=np.float32)
    j2      = base_pt + np.array([np.sin(theta1[0]), -np.cos(theta1[0])]) * L1
    ee      = roller_center[0].to_numpy()

    gui.line(begin=base_pt, end=j2, radius=2, color=0x000050)
    gui.line(begin=j2,     end=ee, radius=2, color=0x000050)
    gui.circle(base_pt, radius=4, color=0xFF0000)
    gui.circle(j2,      radius=4, color=0xFF0000)
    gui.circle(ee,      radius=int(roller_radius * 512), color=0xFF0000)
    gui.text(f'Time: {time_t}   '
             f'Force: {contact_force_vec[0]} N',
             pos=(0.02, 0.95), color=0xFFFFFF)
    gui.show()
