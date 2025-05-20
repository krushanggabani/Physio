import taichi as ti
import numpy as np

# ─── Taichi init ───────────────────────────────────────────────────────────
ti.init(arch=ti.vulkan)

# ─── Simulation parameters ─────────────────────────────────────────────────
dim         = 2
n_particles = 20000
n_grid      = 128
dx          = 1.0 / n_grid
inv_dx      = float(n_grid)
dt          = 2e-4

# ─── Material ───────────────────────────────────────────────────────────────
p_rho   = 1.0
p_vol   = (dx * 0.5)**2
p_mass  = p_rho * p_vol
E       = 5e3
nu      = 0.2
mu_0    = E / (2 * (1 + nu))
lambda_0= E * nu / ((1 + nu) * (1 - 2 * nu))

# ─── Floor & domain boundaries ─────────────────────────────────────────────
floor_level    = 0.0
floor_friction = 0.4


# Speeds
v_desc = 0.5  # vertical descent m/s
v_roll = 0.2  # horizontal roll m/s



# ─── Passive‐joint 2‐link arm params (for both arms) ────────────────────────
L1, L2     = 0.12, 0.1
theta1     = np.array([-np.pi/6, np.pi/6], dtype=np.float32)
theta2     = np.array([-np.pi/6*0, np.pi/6*0], dtype=np.float32)
dtheta1    = np.zeros(2, dtype=np.float32)
dtheta2    = np.zeros(2, dtype=np.float32)
theta1_rest = theta1.copy()
theta2_rest = theta2.copy()
k1, k2     = 50.0, 50.0   # spring k
b1, b2     = 1.0, 1.0     # damping
I1         = L1**2 / 3.0  # inertia
I2         = L2**2 / 3.0

# ─── Roller / end‐effector ─────────────────────────────────────────────────
roller_radius     = 0.025
roller_center     = ti.Vector.field(dim, dtype=ti.f32, shape=2)
roller_velocity   = ti.Vector.field(dim, dtype=ti.f32, shape=2)

state             = ti.Vector.field(1,dtype=ti.f32, shape=2) # 0=descend, 1=roll
contact_height    = ti.Vector.field(1,dtype=ti.f32, shape=2)
contact_force_vec = ti.Vector.field(dim, dtype=ti.f32, shape=2)

# ─── MPM fields ────────────────────────────────────────────────────────────
x      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v      = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
F      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
C      = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
J      = ti.field(dtype=ti.f32, shape=n_particles)

grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))


# two base‐joint anchors (must be visible in Python)
base_positions = np.array([[0.5, 0.8]], dtype=np.float32)


# ─── Base positions encoded as a Taichi func ───────────────────────────────
# Replace your existing get_base with this:
@ti.func
def get_base(i):
    # i is 0 or 1
    # base_x = 0.3 when i=0, 0.7 when i=1
    base_x = 0.3*(1 - i) + 0.7*i
    return ti.Vector([base_x, 1.0])


# ─── Neo‐Hookean stress ────────────────────────────────────────────────────
@ti.func
def neo_hookean_stress(F_i):
    J_det = F_i.determinant()
    FinvT = F_i.inverse().transpose()
    return mu_0 * (F_i - FinvT) + lambda_0 * ti.log(J_det) * FinvT

# ─── Initialize particles + rollers ────────────────────────────────────────
@ti.kernel
def init():
    for p in range(n_particles):
        x[p] = [0.3 + ti.random() * 0.4, ti.random() * 0.2]
        v[p] = [0.0, 0.0]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        J[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)
    # place each roller at its arm’s FK position
    for i in ti.static(range(2)):
        base = get_base(i)
        # j2   = base + ti.Vector([ti.cos(theta1[i]), ti.sin(theta1[i])]) * L1
        # ee   = j2   + ti.Vector([ti.cos(theta1[i] + theta2[i]),
        #                          ti.sin(theta1[i] + theta2[i])]) * L2
        
        j2 = base + ti.Vector([ti.sin(theta1[i]), -ti.cos(theta1[i])]) * L1
        ee = j2 + ti.Vector([ti.sin(theta1[i] + theta2[i]),
                         -ti.cos(theta1[i] + theta2[i])]) * L2
        roller_center[i]   = ee
        roller_velocity[i] = ti.Vector.zero(ti.f32, dim)
        contact_force_vec[i] = ti.Vector.zero(ti.f32, dim)

        state[i]          = 0
        contact_height[i] = 0.0

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
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * neo_hookean_stress(F[p])
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([i, j])
            dpos = (offs.cast(ti.f32) - fx) * dx
            wt   = w[i].x * w[j].y
            grid_v[base + offs] += wt * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offs] += wt * p_mass

# ─── Grid forces + two‐roller contact & floor & walls ────────────────────
@ti.kernel
def apply_grid_forces_and_detect():
    for i in ti.static(range(2)):
        contact_force_vec[i] = ti.Vector.zero(ti.f32, dim)

    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 0:
            v_old = grid_v[I] / m
            v_new = v_old + dt * ti.Vector([0.0, -9.8])
            pos   = I.cast(ti.f32) * dx

            # each roller: enforce only normal component
            for i in ti.static(range(2)):

                rel = pos - roller_center[i]
                if rel.norm() < roller_radius:

                    rv    = roller_velocity[i]
                    n     = rel.normalized()
                    v_norm  = n * n.dot(rv)
                    v_tan   = v_old - n * (n.dot(v_old))
                    v_new   = v_tan + v_norm
                    delta_v = v_new - v_old
                    f_imp   = m * delta_v / dt
                    contact_force_vec[i] += f_imp

            # floor
            if pos.y < floor_level + dx:
                if v_new.y < 0: v_new.y = 0
                v_new.x *= (1 - floor_friction)

            # walls
            if pos.x < dx:      v_new.x = 0
            if pos.x > 1 - dx:  v_new.x = 0

            grid_v[I] = v_new * m

# ─── Grid→Particle (G2P) w/ particle‐level boundaries ────────────────────
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

        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v

        # floor
        if x[p].y < floor_level:
            x[p].y = floor_level
            v[p].y = 0
            v[p].x *= (1 - floor_friction)
        # walls
        if x[p].x < dx:       x[p].x = dx;       v[p].x = 0
        if x[p].x > 1 - dx:   x[p].x = 1 - dx;   v[p].x = 0
        if x[p].y > 1 - dx:   x[p].y = 1 - dx;   v[p].y = 0

        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * new_C) @ F[p]
        J[p] = F[p].determinant()

# ─── Python: update two passive arms ───────────────────────────────────────
def update_arms():
    global theta1, theta2, dtheta1, dtheta2
    for i in range(2):
        Fc    = contact_force_vec[i].to_numpy()
        base  = base_positions[0]
        # FK
        # j2     = base + np.array([np.cos(theta1[i]), np.sin(theta1[i])]) * L1
        j2       = base + np.array([np.sin(theta1[i]), -np.cos(theta1[i])]) * L1
        ee_old = roller_center[i].to_numpy()
        # ee_new = j2 + np.array([np.cos(theta1[i]+theta2[i]),
        #                         np.sin(theta1[i]+theta2[i])]) * L2
        ee_new = j2 + np.array([np.sin(theta1[i]+theta2[i]),
                                -np.cos(theta1[i]+theta2[i])]) * L2
        rv     = (ee_new - ee_old) / dt

        # lever arms & torques
        r1     = ee_new - base
        r2     = ee_new - j2
        tau1_c = r1[0]*Fc[1] - r1[1]*Fc[0]
        tau2_c = r2[0]*Fc[1] - r2[1]*Fc[0]
        tau1   = tau1_c - k1*(theta1[i]-theta1_rest[i]) - b1*dtheta1[i]
        tau2   = tau2_c - k2*(theta2[i]-theta2_rest[i]) - b2*dtheta2[i]

        alpha1 = tau1 / I1
        alpha2 = tau2 / I2
        dtheta1[i] += alpha1 * dt
        theta1[i]  += dtheta1[i] * dt
        dtheta2[i] += alpha2 * dt
        theta2[i]  += dtheta2[i] * dt

        roller_center[i]   = ee_new.tolist()
        roller_velocity[i] = rv.tolist()

# ─── Main loop + GUI ──────────────────────────────────────────────────────
init()
gui = ti.GUI('MPM + 2-Arm Passive Robot', res=(512, 512))

while gui.running:
    for _ in range(15):
        p2g()
        apply_grid_forces_and_detect()
        g2p()
        update_arms()

    # draw soft body
    gui.circles(x.to_numpy(), radius=1.5, color=0x66CCFF)

    # # draw base as a rectangle: x, y, w, h
    # gui.rect((0.3, 0.95), (0.4, 0.1), 0x777777)
    base  = base_positions[0]
    gui.rect(topleft=(base[0]-0.05,base[1]+0.01), bottomright=(base[0]+ 0.05,base[1]-0.01), color=0xFFFFFF)


    # draw arms & rollers
    for i in range(2):
         
        j2   = base + ti.Vector([ti.sin(theta1[i]), -ti.cos(theta1[i])]) * L1

        ee   = roller_center[i].to_numpy()
        gui.line(begin=base, end=j2, radius=2, color=0x66CCFF)
        gui.line(begin=j2,   end=ee, radius=2, color=0x66CCFF)
        gui.circle(base, radius=4, color=0x66CCFF)
        gui.circle(j2,   radius=4, color=0x66CCFF)
        gui.circle(ee,   radius=int(roller_radius*512),
                   color=0xFF0000)

    gui.show()
