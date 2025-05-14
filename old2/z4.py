import taichi as ti
import numpy as np
import imageio

# Initialize Taichi GPU
ti.init(arch=ti.gpu)

# Simulation resolution
quality    = 1
n_particles = 50000 * quality ** 2
n_grid     = 128 * quality * 2
dim        = 2
dtype      = float

# Grid spacing and time step
dx     = 1.0 / n_grid
inv_dx = float(n_grid)
dt     = 1e-4 / quality

# Particle properties
p_vol = (dx * 0.5) ** 2
p_rho = 1.0
p_mass = p_vol * p_rho

# Material parameters (SLS model)
E_e       = 1e3 * 0.8   # elastic modulus (spring)
E_v       = 1e3 * 0.2   # viscous modulus (dashpot)
tau_relax = 0.5         # relaxation time constant

# Collision parameters
threshold = 5e-3  # penetration threshold for impulse
k_penalty = 1e5   # stiffness for penalty force
friction  = 0.2   # Coulomb friction coefficient

# Fields: particles
x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
F = ti.Matrix.field(2, 2, float, n_particles)
S_v    = ti.Matrix.field(2, 2, float, n_particles)  # Viscous stress internal variable  # 

# Fields: grid
grid_v = ti.Vector.field(2, float, (n_grid, n_grid))  # node momentum/velocity
grid_m = ti.field(float, (n_grid, n_grid))            # node mass

# Rigid-box state
to_box   = ti.Vector.field(2, float, shape=())  # lower-left corner
to_box[None]   = [0.48, 0.5]
from_box = ti.Vector.field(2, float, shape=())  # upper-right corner
from_box[None] = [0.52, 0.7]
r_vel    = ti.Vector.field(2, float, shape=())  # box velocity

# Time tracking
t = ti.field(float, shape=())

# Fourier surface coefficients (for initialization)
coeff_a = ti.field(float, shape=9)
coeff_b = ti.field(float, shape=9)
w0      = ti.field(float, shape=())

# Precompute Fourier arrays in Python scope
a_list = [0.0619, 0.0444, -0.0099, 0.0042, -0.0026, 0.0009, 0.0002, -0.0003, 0.001]
b_list = [0.0,    0.007,  -0.004,  0.0004, -0.0019, -0.0012, -0.0002, -0.0011, 0.0001]
for i in ti.static(range(9)):
    coeff_a[i] = a_list[i]
    coeff_b[i] = b_list[i]
w0[None] = 21.689

# Oscillation parameters
amplitude = 0.2
omega     = 10.0


@ti.kernel
def initialize():
    # Reset time
    t[None] = 0.0
    # Initialize particles on a Fourier-based blob
    for p in range(n_particles):
        x_val = -0.135 + ti.random() * (0.14 + 0.135)
        y_surf = coeff_a[0]
        for k in ti.static(range(1, 9)):
            y_surf += coeff_a[k] * ti.cos(k * w0[None] * x_val)
            y_surf += coeff_b[k] * ti.sin(k * w0[None] * x_val)
        y_val = ti.random() * y_surf
        x[p] = ti.Vector([0.5 + x_val * 2, y_val * 2])
        v[p] = [0.0, 0.0]
        F[p] = ti.Matrix.identity(float, 2)
        C[p] = ti.Matrix.zero(float, 2, 2)
        S_v[p] = ti.Matrix.zero(float, 2, 2)
    # Box initial velocity
    r_vel[None] = [0.0, 0.0]


@ti.func
def sdf_box(p):
    # Signed distance to axis-aligned box (positive outside)
    ll = to_box[None]
    ur = from_box[None]
    dxl = ll.x - p.x
    dxr = p.x - ur.x
    dyl = ll.y - p.y
    dyr = p.y - ur.y
    outside = ti.max(ti.max(dxl, dxr), ti.max(dyl, dyr))
    inside  = ti.min(ti.min(-dxl, -dxr), ti.min(-dyl, -dyr))
    return outside if outside > 0.0 else inside

@ti.func
def normal_box(p):
    # Outward normal of the penetrated box face
    ll = to_box[None]
    ur = from_box[None]
    dxl = ll.x - p.x
    dxr = p.x - ur.x
    dyl = ll.y - p.y
    dyr = p.y - ur.y
    m = ti.max(ti.max(-dxl, -dxr), ti.max(-dyl, -dyr))
    return ti.select(m == -dxl, ti.Vector([-1.0,  0.0]),
           ti.select(m == -dxr, ti.Vector([ 1.0,  0.0]),
           ti.select(m == -dyl, ti.Vector([ 0.0, -1.0]),
                                           ti.Vector([ 0.0,  1.0]))))

@ti.func
def collide_impulse(p_i):
    pos = x[p_i]
    dist = sdf_box(pos)
    c = dist - threshold

    p_f = ti.Vector.zero(float, 2)

    if c < 0.0:
        D = normal_box(pos)
        v_rel = v[p_i] - r_vel[None]
        v_n   = v_rel.dot(D)
        v_t   = v_rel - v_n * D
        # penalty spring force
        f1 = -D * (c * k_penalty)
        # Coulomb friction force
        tn = ti.sqrt(v_t.dot(v_t) + 1e-8)
        f2 = -v_t / tn * ti.abs(v_n) * friction

        p_f = (f1+f2)*1.0

    return p_f * dt


@ti.kernel
    def clear_grid():
        zero = ti.Vector.zero(dtype, dim)
        for I in ti.grouped(grid_m):
            grid_v_in[I] = zero
            grid_v_out[I] = zero
            grid_m[I] = 0

            grid_v_in.grad[I] = zero
            grid_v_out.grad[I] = zero
            grid_m.grad[I] = 0

            if ti.static(collision_type == CONTACT_MIXED):
                grid_v_mixed[I] = zero
                grid_v_mixed.grad[I] = zero

        for p in range(0, n_particles):
            if ti.static(collision_type == CONTACT_MIXED):
                v_tmp[p] = zero
                v_tmp.grad[p] = zero
                v_tgt[p] = zero
                v_tgt.grad[p] = zero

@ti.kernel
def substep():

    clear_grid()
    compute_F_tmp(s)
    svd()
    p2g(s)

    forward_kinematics(s,dt)
      
    grid_op(s)
    g2p(s)



    # # Advance rigid-box position
    # t[None] += dt
    # v_y = -amplitude*omega *(ti.sin(omega * t[None]))
    # r_vel[None]   = ti.Vector([0.0, v_y])
    # to_box[None] += r_vel[None] * dt
    # from_box[None] += r_vel[None] * dt

    # # Clear grid fields
    # for i, j in grid_m:
    #     grid_v[i, j] = [0.0, 0.0]
    #     grid_m[i, j] = 0.0

    # # P2G: scatter particles to grid
    # for p in range(n_particles):

    #     collision_impulse = ti.Vector.zero(float, 2)
    #     collision_impulse = collide_impulse(p)

    #     base = (x[p] * inv_dx - 0.5).cast(int)
    #     fx = x[p] * inv_dx - base.cast(float)
    #     # Quadratic B-spline weights
    #     w = [
    #         0.5 * (1.5 - fx) ** 2,
    #         0.75 - (fx - 1.0) ** 2,
    #         0.5 * (fx - 0.5) ** 2,
    #     ]
    #     # Update deformation gradient
    #     F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
    #     # Compute stress (sls model)

    #     U, sigma, V = ti.svd(F[p])
    #     J = sigma[0, 0] * sigma[1, 1]
    #     P_el = 2 * (E_e/2) * (F[p] - U @ V.transpose()) @ F[p].transpose()  
    #     P_visc = 2 * (E_v/2) * (F[p] - U @ V.transpose()) @ F[p].transpose()  # : viscous branch
    #     S_v[p] += dt * (P_visc - S_v[p]) / tau_relax  # : SLS evolution
    #     # Total stress combines elastic + viscous
    #     P_total = P_el + S_v[p] 
               
        
    #     stress = -dt * p_vol * 4 * inv_dx * inv_dx * P_total
    #     affine = stress + p_mass * C[p]

    #     # Scatter mass & momentum
    #     for ii, jj in ti.static(ti.ndrange(3, 3)):
    #         offs = ti.Vector([ii, jj])
    #         dpos = (offs.cast(float) - fx) * dx
    #         weight = w[ii][0] * w[jj][1]
    #         grid_v[base + offs] += weight * (p_mass * v[p] + affine @ dpos + collision_impulse)
    #         grid_m[base + offs] += weight * p_mass

    # # Inject rigid-box velocity into grid nodes inside it
    # for i, j in grid_m:
    #     if grid_m[i,j] > 0:
    #         # Floor collision at y=0 (j=0)
    #         pos_y = j * dx
    #         if pos_y <= 0.0:
    #             # Zero out downward motion: static floor
    #             grid_v[i,j] = [0.0, 0.0]




    # # Grid update: momentum → velocity, apply gravity & BCs
    # for i, j in grid_m:
    #     if grid_m[i, j] > 0:
    #         grid_v[i, j] = grid_v[i, j] / grid_m[i, j]
    #         grid_v[i, j][1] -= dt * 10
    #         if i < 3 and grid_v[i, j][0] < 0:
    #             grid_v[i, j][0] = 0
    #         if i > n_grid - 3 and grid_v[i, j][0] > 0:
    #             grid_v[i, j][0] = 0
    #         if j < 3 and grid_v[i, j][1] < 0:
    #             grid_v[i, j][1] = 0
    #         if j > n_grid - 3 and grid_v[i, j][1] > 0:
    #             grid_v[i, j][1] = 0

    # # G2P: update particles from grid
    # for p in range(n_particles):
    #     base = (x[p] * inv_dx - 0.5).cast(int)
    #     fx = x[p] * inv_dx - base.cast(float)
    #     w = [
    #         0.5 * (1.5 - fx) ** 2,
    #         0.75 - (fx - 1.0) ** 2,
    #         0.5 * (fx - 0.5) ** 2,
    #     ]
    #     new_v = ti.Vector.zero(float, 2)
    #     new_C = ti.Matrix.zero(float, 2, 2)
    #     for ii, jj in ti.static(ti.ndrange(3, 3)):
    #         g_v = grid_v[base + ti.Vector([ii, jj])]

    #         dpos = ti.Vector([ii, jj]).cast(float) - fx
    #         weight = w[ii][0] * w[jj][1]
    #         new_v += weight * g_v
    #         new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
    #     v[p], C[p] = new_v, new_C
    #     x[p] += dt * v[p]

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
# imageio.mimsave("01.gif", frames, fps=30)
# print("Animation saved as 01.gif")


