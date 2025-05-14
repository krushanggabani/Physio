import taichi as ti
import numpy as np
import imageio

# ----------------------------------
# Configuration & Initialization
# ----------------------------------
ti.init(arch=ti.gpu)

# Simulation parameters (tunable)
quality = 1
n_particles = 50000 * quality ** 2
n_grid = 128 * quality * 2

dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 1e-4 / quality

gravity = 10.0  # gravitational acceleration

# Material parameters (SLS model)
E_e = 1e3 * 0.8     # elastic modulus
E_v = 1e3 * 0.2     # viscous modulus
tau_relax = 0.5     # relaxation time

# Particle properties
p_vol = (dx * 0.5)**2
p_mass = p_vol * 1.0  # density = 1

# Rigid-box & motion and roller
amplitude = 0.075
omega = 10.0

t = ti.field(float, shape=())  # global simulation time
r_vel = ti.Vector.field(2, float, shape=())
to_box = ti.Vector.field(2, float, shape=())
from_box = ti.Vector.field(2, float, shape=())

eps= 1e-6
roller_center = ti.Vector.field(2, float, shape=())
roller_radius = ti.field(float, shape=())
roller_angle  = ti.field(float, shape=())
roller_angvel = ti.field(float, shape=())


# Fourier surface coefficients
coeff_a = ti.field(float, shape=9)
coeff_b = ti.field(float, shape=9)
w0 = ti.field(float, shape=())

# Particle grid & state fields
x = ti.Vector.field(2, float, n_particles)   # position
v = ti.Vector.field(2, float, n_particles)   # velocity
C = ti.Matrix.field(2, 2, float, n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, float, n_particles)  # deformation gradient
S_v = ti.Matrix.field(2, 2, float, n_particles)  # viscous stress history

# Grid fields
grid_v = ti.Vector.field(2, float, (n_grid, n_grid))  # node momentum/velocity
grid_m = ti.field(float, (n_grid, n_grid))            # node mass
grid_A = ti.field(int, (n_grid, n_grid))              # active under rigid box
grid_R = ti.field(int,   (n_grid, n_grid))            # roller mask
col = ti.field(int, shape=()) 

# Initialize Fourier coefficients and rigid box
coeffs_a = [0.0619, 0.0444, -0.0099, 0.0042, -0.0026, 0.0009, 0.0002, -0.0003, 0.001]
coeffs_b = [0.0,    0.007,  -0.004, 0.0004, -0.0019, -0.0012, -0.0002, -0.0011, 0.0001]
for i in range(9):
    coeff_a[i] = coeffs_a[i]
    coeff_b[i] = coeffs_b[i]
w0[None] = 21.689


@ti.kernel
def initialize():
     # Set initial time
    t[None] = 0.0
    # Randomly sample particles under a Fourier-defined surface
    for i in range(n_particles):
        x_val = -0.135 + ti.random() * (0.14 + 0.135)
        y_surf = coeff_a[0]
        # Compute Fourier series y = a0 + sum (a_k cos + b_k sin)
        for k in range(1, 9):
            y_surf += coeff_a[k] * ti.cos(k * w0[None] * x_val)
            y_surf += coeff_b[k] * ti.sin(k * w0[None] * x_val)
        x[i] = ti.Vector([0.5 + 2 * x_val, 2 * ti.random() * y_surf])
        v[i] = [0.0, 0.0]
        F[i] = ti.Matrix.identity(float, 2)
        C[i] = ti.Matrix.zero(float, 2, 2)
        S_v[i] = ti.Matrix.zero(float, 2, 2)
    # Rigid box initial position & velocity
    r_vel[None] = [0.0, 0.0]
    to_box[None] = [0.48, 0.3]
    from_box[None] = [0.52, 0.5]

    # Attach roller just below the box
    roller_radius[None] = 0.025
    box_mid_x = 0.5 * (to_box[None][0] + from_box[None][0])
    roller_center[None] = ti.Vector([box_mid_x, to_box[None][1]])
    roller_angle[None]  = 0.0
    roller_angvel[None] = 0.0
    col[None] = 0

h_speed = 0.8

@ti.kernel
def substep():
    # Advance rigid-box position
    t[None] += dt

    if roller_center[None][1] >= 0.20:
        v_y = -amplitude*omega *(ti.sin(omega * t[None]))
        r_vel[None]   = ti.Vector([0.0, v_y])
        to_box[None] += r_vel[None] * dt
        from_box[None] += r_vel[None] * dt

    else:
        # v_y = -amplitude*omega *(ti.sin(omega * t[None]))
        # r_vel[None]   = ti.Vector([0.0, 0.0])
        # to_box[None] += r_vel[None] * dt
        # from_box[None] += r_vel[None] * dt

        old_to_y = to_box[None][1]
        to_box[None][0]   += h_speed * dt
        from_box[None][0] += h_speed * dt

        mid_x = 0.5 * (to_box[None][0] + from_box[None][0])
        real_x = (mid_x - 0.5) * 0.5
        y_s = coeff_a[0]
        for k in ti.static(range(1, 9)):
            y_s += coeff_a[k] * ti.cos(k * w0[None] * real_x)
            y_s += coeff_b[k] * ti.sin(k * w0[None] * real_x)
        y_world = y_s * 2.0

        box_h = from_box[None][1] - to_box[None][1]
        to_box[None][1]   = y_world
        from_box[None][1] = y_world + box_h
        
        r_vel[None] = ti.Vector([h_speed,(to_box[None][1] - old_to_y) / dt ])

    roller_center[None] += r_vel[None] * dt  # roller moves with box
    roller_angvel[None] = r_vel[None][0] / (roller_radius[None] + eps)
    roller_angle[None] += roller_angvel[None] * dt

    # Reset grid
    for I in ti.grouped(grid_m):
        grid_v[I] = [0.0, 0.0]
        grid_m[I] = 0.0
        # grid_A[I] = 0
    

    for i,j in grid_A:
        # Compute world‐space position of this node
        pos = ti.Vector([i, j]) * dx

        # Inside axis‐aligned box?
        inside_box = (pos[0] >= to_box[None][0] and pos[0] <= from_box[None][0]
               and pos[1] >= to_box[None][1] and pos[1] <= from_box[None][1])
        # Inside circular roller?
        inside_roller = (pos - roller_center[None]).norm() <= roller_radius[None]

        # Mark active if either condition holds
        active = inside_box | inside_roller
        grid_A[i,j] = 1 if inside_box else 0






    # P2G: scatter particles to grid
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)
        base[0] = ti.max(0, ti.min(base[0], n_grid - 3))
        base[1] = ti.max(0, ti.min(base[1], n_grid - 3))
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline weights
        w = [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ]
        # Update deformation gradient
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Compute stress (sls model)
        U, sigma, V = ti.svd(F[p])
        J = sigma[0, 0] * sigma[1, 1]
        P_el = 2 * (E_e/2) * (F[p] - U @ V.transpose()) @ F[p].transpose()
        P_visc = 2 * (E_v/2) * (F[p] - U @ V.transpose()) @ F[p].transpose()  # : viscous branch
        S_v[p] += dt * (P_visc - S_v[p]) / tau_relax  # : SLS evolution
        # Total stress combines elastic + viscous
        P_total = P_el + S_v[p]
        stress = -dt * p_vol * 4 * inv_dx * inv_dx * P_total
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
        m = grid_m[i,j]
        if m > 0:
            # Box collision: override momentum
            if grid_A[i,j] == 1:
                grid_v[i,j] = r_vel[None] * m*0.0005
            else:
                # Floor collision at y=0 (j=0)
                pos_y = j * dx
                if pos_y <= 0.0:
                    # Zero out downward motion: static floor
                    grid_v[i,j] = [0.0, 0.0]
    # Grid update: momentum → velocity, apply gravity & BCs




    for i, j in grid_m:
        m = grid_m[i,j]
        if m > 0:
            grid_v[i, j] = grid_v[i, j] / m
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
        base[0] = ti.max(0, ti.min(base[0], n_grid - 3))
        base[1] = ti.max(0, ti.min(base[1], n_grid - 3))
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
# Storage for recording frames
frames = []
gui = ti.GUI("MPM with Rigid Box", res=800, background_color=0x112F41)

while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    for _ in range(int(5e-3 // dt)):
        substep()
    # Draw particles
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    # Draw rigid box with topleft/bottomright
    p1 = to_box[None].to_numpy()
    p2 = from_box[None].to_numpy()
    gui.rect(topleft=p1, bottomright=p2, color=0xFF0000)
    gui.circle(pos=roller_center[None].to_numpy(), radius=roller_radius[None]*800, color=0xEEEE00)

    lb = roller_center[None].to_numpy()
    angle = roller_angle[None]
    dir_vec = np.array([np.cos(angle), np.sin(angle)]) * roller_radius[None] * 1.5
    le = roller_center[None].to_numpy() + dir_vec
    gui.line(lb, le, radius=1, color=0x068587)


    frame = gui.get_image()
    frame = (np.array(frame) * 255).astype(np.uint8)
    frame = np.rot90(frame, k=1)  # Rotate 90 degrees counterclockwise
    frames.append(np.array(frame))
    gui.show()

# Save animation
imageio.mimsave("Roller_staright.gif", frames, fps=30)
print("Animation saved as 01.gif")