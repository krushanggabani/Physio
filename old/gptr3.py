import taichi as ti
import numpy as np

# --------------------------------
# Simulation parameters
# --------------------------------
ti.init(arch=ti.vulkan)

# MPM grid
domain_size = 1.0
n_grid = 128
dx = domain_size / n_grid
inv_dx = float(n_grid)
dt = 2e-4

# Material (viscoelastic rubber)
rho = 1000.0
E = 5e4
nu = 0.3
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Viscoelastic relaxation
tau = 0.5
beta = dt / tau

# Soft body resolution
n_particles_side = 64
n_particles = n_particles_side * n_particles_side

# Robot & roller parameters
L1, L2 = 0.1,0.1
base_x = 0.5
# Spring-damper base
M_base, K_base, C_base = 5.0, 1e3, 1e2
base_rest = 0.5
# Joint 1
I1, K1, C1 = 0.1, 50.0, 5.0
# Joint 2 & roller
I2, K2, C2 = 0.05, 50.0, 5.0
I_r, dR, friction_mu = 0.02, 0.05, 1.0

# --------------------------------
# Taichi fields
# --------------------------------
# Particles
x = ti.Vector.field(2, ti.f32, n_particles)
v = ti.Vector.field(2, ti.f32, n_particles)
F = ti.Matrix.field(2, 2, ti.f32, n_particles)

# Grid
grid_v = ti.Vector.field(2, ti.f32, (n_grid, n_grid))
grid_m = ti.field(ti.f32, (n_grid, n_grid))

# Robot state
base_y    = ti.field(ti.f32, (),)
base_vy   = ti.field(ti.f32, (),)
theta1    = ti.field(ti.f32, (),)
omega1    = ti.field(ti.f32, (),)
theta2    = ti.field(ti.f32, (),)
omega2    = ti.field(ti.f32, (),)
omega_r   = ti.field(ti.f32, (),)
phi       = ti.field(ti.f32, (),)

# Contact accumulators
force_roller = ti.Vector.field(2, ti.f32,())
torque_roller= ti.field(ti.f32,())

# --------------------------------
# Initialization
# --------------------------------
@ti.kernel
def initialize():
    for i in range(n_particles):
        sx = i % n_particles_side
        sy = i // n_particles_side
        x[i] = ti.Vector([0.3, 0.0]) + 0.3 * ti.Vector([sx/(n_particles_side-1), sy/(n_particles_side-1)])
        v[i] = ti.Vector([0.0, 0.0])
        F[i] = ti.Matrix.identity(ti.f32, 2)
    base_y[None], base_vy[None] = 0.8, 0.0
    theta1[None], omega1[None] = -0.3, 0.0
    theta2[None], omega2[None] = 0.5, 0.0
    omega_r[None], phi[None]     = 0.0, 0.0

initialize()

# --------------------------------
# Simulation substep
# --------------------------------
@ti.kernel
def substep():
    # Reset grid
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector([0.0, 0.0])
        grid_m[I] = 0.0
    # Reset accumulators
    force_roller[None], torque_roller[None] = ti.Vector([0.0,0.0]), 0.0

    # P2G
    for p in range(n_particles):
        Xp = x[p] * inv_dx
        base = int(Xp - 0.5)
        fx = Xp - base
        # Quadratic kernel
        w = [0.5*(1.5-fx)**2, 0.75-(fx-1.0)**2, 0.5*(fx-0.5)**2]
        mass_p = rho*(dx*0.5)**2
        # stress
        Jp = F[p].determinant()
        stress = mu*(F[p] - F[p].inverse().transpose()) + lmbda*ti.log(Jp)*F[p].inverse().transpose()
        stress = - (dt*inv_dx*inv_dx) * stress * mass_p
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                idx = base + ti.Vector([i,j])
                if 0 <= idx.x < n_grid and 0 <= idx.y < n_grid:
                    weight = w[i].x * w[j].y
                    dpos = (idx.cast(ti.f32)*dx - x[p])
                    grid_v[idx] += weight*(mass_p*v[p] + stress@dpos)
                    grid_m[idx] += weight*mass_p

    # Grid update + collision on grid
    # Precompute roller state
    by = base_y[None]
    j1 = ti.Vector([base_x + L1*ti.sin(theta1[None]), by - L1*ti.cos(theta1[None])])
    cen = j1 + ti.Vector([L2*ti.sin(theta1[None]+theta2[None]), -L2*ti.cos(theta1[None]+theta2[None])])
    # Assume roller_vel = base_vy only vertical + link rotations neglected for grid collision
    roller_vel = ti.Vector([0.0, base_vy[None]])
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            # normalize
            grid_v[I] = grid_v[I]/grid_m[I]
            # gravity
            grid_v[I].y -= dt*9.8
            # boundary
            if I.x<3 or I.x>n_grid-4 or I.y<3:
                grid_v[I]=ti.Vector([0.0,0.0])
        # grid-level collision: override velocity inside roller
        pos = I.cast(ti.f32)*dx
        if (pos-cen).norm() < dR:
            # accumulate momentum change
            delta_v = roller_vel - grid_v[I]
            grid_v[I] = roller_vel
            # impulse = m * dv
            imp = grid_m[I]*delta_v
            force_roller[None] += imp / dt
            torque_roller[None] += (pos - cen).cross(imp/dt)

    # G2P + constitutive
    for p in range(n_particles):
        Xp = x[p]*inv_dx
        base = int(Xp-0.5)
        fx = Xp - base
        w = [0.5*(1.5-fx)**2, 0.75-(fx-1.0)**2, 0.5*(fx-0.5)**2]
        new_v = ti.Vector([0.0,0.0])
        new_C = ti.Matrix.zero(ti.f32,2,2)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                idx = base + ti.Vector([i,j])
                if 0<=idx.x<n_grid and 0<=idx.y<n_grid:
                    weight = w[i].x * w[j].y
                    g_v = grid_v[idx]
                    new_v += weight*g_v
                    new_C += 4*inv_dx*weight*g_v.outer_product(idx.cast(ti.f32)*dx - x[p])
        v[p] = new_v
        x[p] += dt*v[p]
        F[p] = (ti.Matrix.identity(ti.f32,2)+dt*new_C)@F[p]
        # visco relaxation
        U, s, Vt = ti.svd(F[p])
        for d in ti.static(range(2)):
            s[d,d] = ti.exp((1-beta)*ti.log(s[d,d]))
        F[p] = U@s@Vt

    # Robot dynamics
    # base
    Fb = force_roller[None].y - K_base*(base_y[None]-base_rest) - C_base*base_vy[None] - M_base*9.8
    base_vy[None] += dt*Fb/M_base
    base_y[None]  += dt*base_vy[None]
    # link2
    tau2 = torque_roller[None] - K2*theta2[None] - C2*omega2[None]
    omega2[None] += dt*tau2/I2
    theta2[None] += dt*omega2[None]
    # link1
    tau1 = -K1*theta1[None] - C1*omega1[None]
    omega1[None] += dt*tau1/I1
    theta1[None] += dt*omega1[None]
    # roller spin
    omega_r[None] += dt*torque_roller[None]/I_r
    phi[None]      += dt*omega_r[None]

# --------------------------------
# Visualization
# --------------------------------
gui = ti.GUI("MPM Robot Arm", res=(600,600))
while gui.running:
    for _ in range(5): substep()
    gui.clear(0x112F41)
    gui.circles(x.to_numpy(), radius=1.5, color=0xED553B)
    # draw robot
    by = base_y[None]
    bpos = np.array([base_x,by])
    j1 = bpos + np.array([np.sin(theta1[None]), -np.cos(theta1[None])])*L1
    j2 = j1 + np.array([np.sin(theta1[None]+theta2[None]), -np.cos(theta1[None]+theta2[None])])*L2
    gui.line(bpos, j1, radius=3, color=0x4ECDC4)
    gui.line(j1, j2, radius=3, color=0x4ECDC4)
    gui.circle(j2, radius=int(dR), color=0xC7F464)
    gui.show()
