import taichi as ti
import numpy as np
import imageio
# Initialize Taichi GPU
ti.init(arch=ti.gpu)
# Simulation resolution
quality = 1
n_particles = 50000 * quality ** 2
n_grid = 128 * quality *2
dx = 1 / n_grid
inv_dx = float(n_grid)
dt = 1e-4 / quality
# Particle properties
p_vol = (dx * 0.5) ** 2
p_rho = 1
p_mass = p_vol * p_rho
# Material parameters
E_e          = 1e3 * 0.8           # Elastic spring modulus (SLS)  #
E_v          = 1e3 * 0.8           # Viscous dashpot modulus      #
tau_relax    = 0.5                 # Relaxation time constant     #
# k_penalty = 1e5
# c_penalty = 1e2
# Fields: particles
x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
F = ti.Matrix.field(2, 2, float, n_particles)
S_v    = ti.Matrix.field(2, 2, float, n_particles)  # Viscous stress internal variable  #
# Fields: grid
grid_v = ti.Vector.field(2, float, (n_grid, n_grid))  # node momentum/velocity
grid_m = ti.field(float, (n_grid, n_grid))            # node mass
grid_A = ti.field(int, (n_grid, n_grid))              # active under rigid box
grid_force  = ti.Vector.field(2, float, shape=(n_grid, n_grid)) # : collision force
# Gravity and rigid-box parameters
gravity = 10
t = ti.field(float, shape=())                 # simulation time
r_v = 1  # rigid speed (units per second)
# Rigid-box velocity vector (downwards)
r_vel = ti.Vector.field(2, float, shape=())
# Rigid-box bounds: min and max corners
to_box = ti.Vector.field(2, float, shape=())  # lower-left corner
from_box = ti.Vector.field(2, float, shape=()) # upper-right corner
# ----------------
# Fourier Surface
# ----------------
coeff_a = ti.field(float, shape=9)
coeff_b = ti.field(float, shape=9)
w0      = ti.field(float, shape=())
# Assign arrays in Python then copy
a_list = [0.0619, 0.0444, -0.0099, 0.0042, -0.0026, 0.0009, 0.0002, -0.0003, 0.001]
b_list = [0.0,    0.007,  -0.004,  0.0004, -0.0019, -0.0012, -0.0002, -0.0011, 0.0001]
for i in ti.static(range(9)):
    coeff_a[i] = a_list[i]
    coeff_b[i] = b_list[i]
w0[None] = 21.689
amplitude = 0.2
omega     = 10
# Semicircle initialization parameters
center_sc  = ti.Vector.field(2, float, shape=())  # semicircle center
radius_sc  = ti.field(float, shape=())           # semicircle radius
center_sc[None] = [0.5, 0.0]
radius_sc[None] = 0.25
@ti.kernel
def initialize():
    t[None] = 0.0
    # Initialize particles in a blob
    for i in range(n_particles):
        # x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.2]
        # x[i]   = [ti.random()*0.4+0.3, ti.random()*0.2]
        # theta = ti.random() * np.pi            # angle in [0, π]
        # r = radius_sc[None] * ti.sqrt(ti.random())
        # x[i] = center_sc[None] + ti.Vector([r * ti.cos(theta), r * ti.sin(theta)])
        x_val = -0.135 + ti.random() * (0.14 + 0.135)
        # compute Fourier surface y = a0 + sum_{k=1}^8 (a_k cos(k w x) + b_k sin(k w x))
        y_surf = coeff_a[0]
        for k in ti.static(range(1, 9)):
            y_surf += coeff_a[k] * ti.cos(k * w0[None] * x_val)
            y_surf += coeff_b[k] * ti.sin(k * w0[None] * x_val)
        # sample y uniformly in [0, y_surf]
        y_val = ti.random() * y_surf
        x[i]  = ti.Vector([0.5+x_val*2, y_val*2])
        v[i] = [0.0, 0.0]
        F[i] = ti.Matrix.identity(float, 2)
        C[i] = ti.Matrix.zero(float, 2, 2)
        S_v[i] = ti.Matrix.zero(float, 2, 2)
    # Rigid-box initial velocity (downwards)
    r_vel[None] = [0.0, 0.0]
    # Rigid-box initial corners (axis-aligned)
    to_box[None] = [0.48, 0.5]   # lower-left corner
    from_box[None] = [0.52, 0.7] # upper-right corner


@ti.kernel
def substep():
    # Advance rigid-box position
    t[None] += dt


    to_box[None][0]   += 0.1 * dt
    from_box[None][0] += 0.1 * dt

    y_s = coeff_a[0]
    mid_x = 0.5 * (to_box[None][0] + from_box[None][0])
    real_x = (mid_x - 0.5) * 0.5 

    for k in ti.static(range(1, 9)):
        y_s += coeff_a[k] * ti.cos(k * w0[None] * real_x)
        y_s += coeff_b[k] * ti.sin(k * w0[None] * real_x)
    y_world = y_s * 2.0

    box_h = from_box[None][1] - to_box[None][1]
    to_box[None][1]   = y_world
    from_box[None][1] = y_world + box_h
    
    v_y = -amplitude*omega *(ti.sin(omega * t[None]))
    r_vel[None]   = ti.Vector([0.0, v_y])
    # to_box[None] += r_vel[None] * dt
    # from_box[None] += r_vel[None] * dt
    
    
    
    
    
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
        if grid_m[i,j] > 0:
            # Box collision: override momentum
            if grid_A[i,j] == 1:
                # pos = ti.Vector([i,j])*dx
                # d = from_box[None][1] - pos[1]
                # if d>0:
                #     n = ti.Vector([0.0,1.0])
                #     v_rel = (grid_v[i,j]/grid_m[i,j] - r_vel[None]).dot(n)
                #     F_pen = k_penalty*d*n - c_penalty*v_rel*n
                #     grid_v[i,j] += F_pen*dt
                grid_v[i,j] = r_vel[None] * grid_m[i,j]*0.01
            else:
                # Floor collision at y=0 (j=0)
                pos_y = j * dx
                if pos_y <= 0.0:
                    # Zero out downward motion: static floor
                    grid_v[i,j] = [0.0, 0.0]
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
    frame = gui.get_image()
    frame = (np.array(frame) * 255).astype(np.uint8)
    frame = np.rot90(frame, k=1)  # Rotate 90 degrees counterclockwise
    frames.append(np.array(frame))
    gui.show()
# Save animation
imageio.mimsave("01.gif", frames, fps=30)
print("Animation saved as 01.gif")