import taichi as ti
import numpy as np
import imageio

# Initialize Taichi GPU
ti.init(arch=ti.gpu)

# Simulation resolution
quality     = 1
n_particles = 50000 * quality ** 2
n_grid      = 128 * quality * 2

dx     = 1 / n_grid
inv_dx = float(n_grid)
dt     = 1e-4 / quality

# Particle properties
p_vol = (dx * 0.5)**2
p_rho = 1
p_mass = p_vol * p_rho

# Material parameters
eps       = 1e-6
E_e       = 1e3 * 0.8
E_v       = 1e3 * 0.2
tau_relax = 0.5



# Fields: particles
x   = ti.Vector.field(2, float, n_particles)
v   = ti.Vector.field(2, float, n_particles)
C   = ti.Matrix.field(2, 2, float, n_particles)
F   = ti.Matrix.field(2, 2, float, n_particles)
S_v = ti.Matrix.field(2, 2, float, n_particles)

# Fields: grid
grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))
grid_A = ti.field(int, (n_grid, n_grid))
grid_R = ti.field(int, (n_grid, n_grid))
grid_f = ti.Vector.field(2, float, (n_grid, n_grid))

# Gravity
gravity = 10.0

# Rigid-box parameters
t        = ti.field(float, shape=())
r_vel    = ti.Vector.field(2, float, shape=())
to_box   = ti.Vector.field(2, float, shape=())
from_box = ti.Vector.field(2, float, shape=())

# Fourier surface coefficients
coeff_a = ti.field(float, shape=9)
coeff_b = ti.field(float, shape=9)
w0      = ti.field(float, shape=())
a_list  = [0.0619, 0.0444, -0.0099, 0.0042, -0.0026, 0.0009, 0.0002, -0.0003, 0.001]
b_list  = [0.0,    0.007,  -0.004,  0.0004, -0.0019, -0.0012, -0.0002, -0.0011, 0.0001]
for i in ti.static(range(9)):
    coeff_a[i] = a_list[i]
    coeff_b[i] = b_list[i]
w0[None] = 21.689

# Roller parameters
roller_center = ti.Vector.field(2, float, shape=())
roller_radius = ti.field(float, shape=())
roller_angle  = ti.field(float, shape=())
roller_angvel = ti.field(float, shape=())

# New: horizontal scan speed
h_speed = 0.5  # units/sec
# Rigid-box & motion and roller
amplitude = 0.075
omega = 10.0

col = ti.field(int, shape=()) 

@ti.kernel
def initialize():
    t[None] = 0.0
    # Initialize particles on Fourier blob
    for i in range(n_particles):
        x_val = -0.135 + ti.random() * (0.14 + 0.135)
        y_s = coeff_a[0]
        for k in ti.static(range(1, 9)):
            y_s += coeff_a[k] * ti.cos(k * w0[None] * x_val)
            y_s += coeff_b[k] * ti.sin(k * w0[None] * x_val)
        y_val = ti.random() * y_s
        x[i] = ti.Vector([0.5 + x_val * 2, y_val * 2])
        v[i] = [0.0, 0.0]
        F[i] = ti.Matrix.identity(float, 2)
        C[i] = ti.Matrix.zero(float, 2, 2)
        S_v[i] = ti.Matrix.zero(float, 2, 2)

    # Initial box position & velocity
    r_vel[None] = [0.0, 0.0]
    to_box[None]   = [0.48, 0.3]
    from_box[None] = [0.52, 0.5]

    # Attach roller just below box
    roller_radius[None] = 0.025
    box_mid_x = 0.5 * (to_box[None][0] + from_box[None][0])
    roller_center[None] = ti.Vector([box_mid_x, to_box[None][1]])
    roller_angle[None]  = 0.0
    roller_angvel[None] = 0.0
    col[None] = 0



@ti.kernel
def substep():
    # Advance time
    t[None] += dt

    if col[None] ==0 :
        v_y = -amplitude*omega *(ti.sin(omega * t[None]))
        r_vel[None]   = ti.Vector([0.0, v_y])
        to_box[None] += r_vel[None] * dt
        from_box[None] += r_vel[None] * dt
    else:
        v_y = -amplitude*omega *(ti.sin(omega * t[None]))
        r_vel[None]   = ti.Vector([0.0, 0.0])
        to_box[None] += r_vel[None] * dt
        from_box[None] += r_vel[None] * dt

        # old_to_y = to_box[None][1]
        # to_box[None][0]   += h_speed * dt
        # from_box[None][0] += h_speed * dt

        # mid_x = 0.5 * (to_box[None][0] + from_box[None][0])
        # real_x = (mid_x - 0.5) * 0.5
        # y_s = coeff_a[0]
        # for k in ti.static(range(1, 9)):
        #     y_s += coeff_a[k] * ti.cos(k * w0[None] * real_x)
        #     y_s += coeff_b[k] * ti.sin(k * w0[None] * real_x)
        # y_world = y_s * 2.0

        # box_h = from_box[None][1] - to_box[None][1]
        # to_box[None][1]   = y_world
        # from_box[None][1] = y_world + box_h
        
        # r_vel[None] = ti.Vector([h_speed,(to_box[None][1] - old_to_y) / dt ])

    # 5) Move roller with box
    roller_center[None] += r_vel[None] * dt
    roller_angvel[None]  = r_vel[None][0] / (roller_radius[None] + eps)
    roller_angle[None]  += roller_angvel[None] * dt

    # Build masks & clear grids
    for i, j in grid_A:
        pos = ti.Vector([i, j]) * dx
        inside = (pos[0] >= to_box[None][0] and pos[0] <= from_box[None][0]
               and pos[1] >= to_box[None][1] and pos[1] <= from_box[None][1])
        
        inside_roller = (roller_center[None]-pos).norm() <= roller_radius[None]
        print((roller_center[None]-pos))
        grid_A[i, j] = 1 if inside | inside_roller else 0
        col[None] = col[None] | grid_A[i, j]
    



    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0
        grid_f[i, j] = [0.0, 0.0]

    # P2G
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)
        if base[0] < 0 | base[1] < 0 | base[0] > n_grid-3 | base[1] >n_grid-3:  
            print(base)
        base[0] = ti.max(0, ti.min(base[0], n_grid - 3))
        base[1] = ti.max(0, ti.min(base[1], n_grid - 3))
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2,
             0.75     - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        U, sigma, V = ti.svd(F[p])
        P_el   = 2 * (E_e/2) * (F[p] - U @ V.transpose()) @ F[p].transpose()
        P_visc = 2 * (E_v/2) * (F[p] - U @ V.transpose()) @ F[p].transpose()
        S_v[p] += dt * (P_visc - S_v[p]) / tau_relax
        P_tot   = P_el + S_v[p]
        stress  = -dt * p_vol * 4 * inv_dx * inv_dx * P_tot
        affine  = stress + p_mass * C[p]
        for ii, jj in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([ii, jj])
            dpos = (offs.cast(float) - fx) * dx
            weight = w[ii][0] * w[jj][1]
            grid_v[base + offs] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offs] += weight * p_mass

    # Inject velocities & collisions
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            if grid_A[i, j] == 1 :
                grid_v[i, j] = r_vel[None] * grid_m[i, j] * 0.001
            
            else:
                if j * dx <= 0.0:
                    grid_v[i, j] = [0.0, 0.0]
        # grid_v[i,j] *= (1 - 100 * dt)

    # Grid update + BCs
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_v[i, j] / grid_m[i, j]
            grid_v[i, j][1] -= dt * gravity
            if i < 3 and grid_v[i, j][0] < 0: grid_v[i, j][0] = 0
            if i > n_grid-3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
            if j > n_grid-3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

    # G2P
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)

        if base[0] < 0 | base[1] < 0 | base[0] > n_grid-3 | base[1] >n_grid-3:  
            print(base)
        base[0] = ti.max(0, ti.min(base[0], n_grid - 3))
        base[1] = ti.max(0, ti.min(base[1], n_grid - 3))
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2,
             0.75     - (fx - 1.0)**2,
             0.5 * (fx - 0.5)**2]
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

        disp = dt * new_v
        if disp.norm() > 0.2:
            disp = disp.normalized() * 0.1
        x[p] += disp
        # optionally update v[p] to reflect the clamped move
        v[p] = disp / dt

# ----- Setup & main loop -----
initialize()
gui = ti.GUI("MPM: Box Following Soft-Body Surface", res=800, background_color=0x112F41)

while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    for _ in range(int(5e-3 // dt)):
        substep()
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    gui.rect(topleft=to_box[None].to_numpy(),
             bottomright=from_box[None].to_numpy(),
             color=0xFF0000)
    gui.circle(pos=roller_center[None].to_numpy(),
               radius=roller_radius[None] * 800,
               color=0xEEEE00)
    # Draw roller orientation
    lb = roller_center[None].to_numpy()
    angle  = roller_angle[None]
    dir_v  = np.array([np.cos(angle), np.sin(angle)]) * roller_radius[None] * 1.5
    le     = lb + dir_v
    gui.line(lb, le, radius=1, color=0x068587)

    gui.show()
