import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ti.init(arch=ti.gpu)

# ─── Simulation Parameters ───────────────────────────────────────────────
dim = 2
n_particles = 4000
n_grid = 128
dx = 1 / n_grid
inv_dx = float(n_grid)
dt = 2e-4
p_vol = (dx * 0.5)**2
p_rho = 1
p_mass = p_vol * p_rho
E = 400
nu = 0.2
mu_0 = E / (2 * (1 + nu))
la_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# ─── Passive Robot Arm Model ─────────────────────────────────────────────
link_length = 0.2
base_pos = ti.Vector([0.6, 0.6])
rest_angles = [np.pi / 4, -np.pi / 3]
spring_k = 20.0
damping_b = 0.5

theta1 = ti.field(dtype=ti.f32, shape=())
theta2 = ti.field(dtype=ti.f32, shape=())
omega1 = ti.field(dtype=ti.f32, shape=())
omega2 = ti.field(dtype=ti.f32, shape=())

# ─── MPM Fields ──────────────────────────────────────────────────────────
x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
Jp = ti.field(dtype=ti.f32, shape=n_particles)
contact_force = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)

grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

@ti.kernel
def initialize():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.2 + 0.1]
        v[i] = [0, 0]
        F[i] = ti.Matrix.identity(ti.f32, dim)
        Jp[i] = 1
    theta1[None] = rest_angles[0]
    theta2[None] = rest_angles[1]

@ti.func
def get_arm_joints():
    joint1 = base_pos
    joint2 = joint1 + link_length * ti.Vector([ti.cos(theta1[None]), ti.sin(theta1[None])])
    tip = joint2 + link_length * ti.Vector([ti.cos(theta1[None] + theta2[None]), ti.sin(theta1[None] + theta2[None])])
    return joint1, joint2, tip

@ti.kernel
def update_arm():
    torque1 = -spring_k * (theta1[None] - rest_angles[0]) - damping_b * omega1[None]
    torque2 = -spring_k * (theta2[None] - rest_angles[1]) - damping_b * omega2[None]
    omega1[None] += torque1 * dt
    omega2[None] += torque2 * dt
    theta1[None] += omega1[None] * dt
    theta2[None] += omega2[None] * dt

@ti.func
def robot_arm_force(pos):
    joint1, joint2, tip = get_arm_joints()
    f = ti.Vector([0.0, 0.0])
    for joint in [joint1, joint2, tip]:
        r = pos - joint
        dist = r.norm()
        if dist < 0.05:
            normal = r.normalized()
            penetration = 0.05 - dist
            f += 3000 * penetration * normal - 10 * normal
    return f

@ti.kernel
def substep(t: ti.f32):
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0

    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]
        h = ti.exp(10 * (1.0 - Jp[p]))
        mu, la = mu_0 * h, la_0 * h

        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(dim)):
            J *= sig[d, d]

        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + \
                 ti.Matrix.identity(ti.f32, dim) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_v[i, j] / grid_m[i, j]
            grid_v[i, j][1] -= dt * 9.8
            if i < 3 or i > n_grid - 3 or j < 3:
                grid_v[i, j] = [0, 0]

    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]

        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                g_v = grid_v[base + offset]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        frc = robot_arm_force(x[p])
        contact_force[p] = frc
        new_v += dt * frc / p_mass

        v[p] = new_v
        x[p] += dt * v[p]
        C[p] = new_C

initialize()
positions = []
forces = []

for frame in range(300):
    update_arm()
    substep(frame * dt)
    if frame % 5 == 0:
        positions.append(x.to_numpy())
        force_mags = np.linalg.norm(contact_force.to_numpy(), axis=1)
        forces.append(force_mags)

fig, ax = plt.subplots()
sc = ax.scatter([], [], c=[], cmap='hot', s=1, vmin=0, vmax=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
cb = plt.colorbar(sc, ax=ax, label='Contact Force (N)')

def update(f):
    sc.set_offsets(positions[f])
    sc.set_array(forces[f])
    return sc,

ani = animation.FuncAnimation(fig, update, frames=len(positions), interval=50)
plt.title("Human-Robot Massage Simulation (Force Visualization)")
plt.show()