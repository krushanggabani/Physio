import taichi as ti
import numpy as np

# ---------------------------------------------
# Taichi MPM Softbody Simulation with Rigid Box
# ---------------------------------------------
# - Uses Material Point Method (MPM) to simulate
#   deformable soft body behavior.
# - A rigid, axis-aligned rectangular box oscillates
#   vertically (up/down) and collides with the soft body.
# - Grid nodes under the box are forced to follow
#   the box velocity, imparting deformation via MPM.

# Initialize Taichi to run on GPU for performance
ti.init(arch=ti.gpu)

# ----------------------
# Simulation Parameters
# ----------------------
quality = 1                      # Resolution multiplier
n_particles = 10000 * quality**2 # Total number of material points
n_grid = 128 * quality          # Grid resolution (n_grid x n_grid nodes)

dx = 1.0 / n_grid                # Grid cell size
inv_dx = float(n_grid)           # 1/dx, used for coordinate conversions
dt = 1e-4 / quality              # Timestep size

# ----------------------
# Particle Properties
# ----------------------
p_vol = (dx * 0.5)**2            # Initial particle volume
p_rho = 1.0                      # Material density (mass per volume)
p_mass = p_vol * p_rho           # Particle mass

# --------------------------------
# Neo-Hookean Material Parameters
# --------------------------------
E, nu = 1e3, 0.2                 # Young's modulus and Poisson's ratio
mu_0 = E / (2 * (1 + nu))        # First Lamé parameter (shear modulus)
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))  # Second Lamé param

# ----------------
# Grid Fields
# ----------------
# grid_v: node momentum/velocity
# grid_m: node mass
# grid_A: active flag = 1 if node under rigid box, else 0
grid_v = ti.Vector.field(2, float, shape=(n_grid, n_grid))
grid_m = ti.field(float, shape=(n_grid, n_grid))
grid_A = ti.field(int,   shape=(n_grid, n_grid))

# ------------------
# Particle Fields
# ------------------
# x:  position, v:  velocity
# C:  affine velocity field (for MLS-MPM)
# F:  deformation gradient tensor
x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
F = ti.Matrix.field(2, 2, float, n_particles)

# -----------------------------------
# Rigid Box (Up-Down Oscillation)
# -----------------------------------
gravity = 10.0                   # Gravity magnitude (downwards)
amplitude = 0.2                  # Vertical oscillation amplitude
omega = 10     # Angular frequency

t = ti.field(float, shape=())                 # Simulation time accumulator
mid_box   = ti.Vector.field(2, float, shape=())  # Box center position
half_size = ti.Vector.field(2, float, shape=())  # Half-extents of box (width/2, height/2)
r_vel     = ti.Vector.field(2, float, shape=())  # Instantaneous box velocity

to_box    = ti.Vector.field(2, float, shape=())  # Box lower-left corner
from_box  = ti.Vector.field(2, float, shape=())  # Box upper-right corner

@ti.kernel
def initialize():
    """
    Initialize particle states and rigid box geometry.
    """
    # Create a random blob of particles in [0.3,0.7] x [0,0.2]
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.3, ti.random() * 0.2]
        v[i] = [0.0, 0.0]
        # Start with identity deformation (no pre-strain)
        F[i] = ti.Matrix.identity(float, 2)
        # No initial affine velocity
        C[i] = ti.Matrix.zero(float, 2, 2)

    # Initialize time
    t[None] = 0.0
    # Define initial box corners
    to_box[None]   = [0.45, 0.5]   # lower-left
    from_box[None] = [0.55, 0.7]   # upper-right
    # Compute center and half-size
    mid_box[None]   = (to_box[None] + from_box[None]) * 0.5
    half_size[None] = (from_box[None] - to_box[None]) * 0.5
    # Initial rigid-box velocity
    r_vel[None] = [0.0, 0.0]

@ti.kernel
def substep():
    """
    Single MPM substep:
      1) Advance rigid box
      2) Build mask for collision
      3) P2G (particle→grid)
      4) Inject rigid velocity
      5) Grid update (apply gravity, BC)
      6) G2P (grid→particle)
    """
    # 1) Advance simulation time and box motion
    t[None] += dt

    # Velocity is derivative: v_y = A ω sin(ωt)
    v_y = -amplitude * omega * ti.sin(omega * t[None])

    # Update center and velocity
    r_vel[None]   = ti.Vector([0.0, v_y])
    # Recompute corners from center + half-size
    to_box[None] += r_vel[None] * dt
    from_box[None] += r_vel[None] * dt


    # 2) Mark grid nodes under the rigid box for collision
    for i, j in grid_A:
        pos = ti.Vector([i, j]) * dx  # world-space node position
        inside = (
            pos[0] >= to_box[None][0] and pos[0] <= from_box[None][0] and
            pos[1] >= to_box[None][1] and pos[1] <= from_box[None][1]
        )
        grid_A[i, j] = 1 if inside else 0

    # Clear grid velocity & mass from previous step
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0

    # 3) Particle-to-Grid (P2G)
    for p in range(n_particles):
        # Determine base node index and relative position fx
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx   = x[p] * inv_dx - base.cast(float)
        # Quadratic B-spline weights for interpolation
        w = [
            0.5 * (1.5 - fx)**2,
            0.75 - (fx - 1.0)**2,
            0.5 * (fx - 0.5)**2,
        ]
        # 3.1) Update deformation gradient F
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # 3.2) Compute PK1 stress via compressible neo-Hookean
        h = 0.5
        mu, la = mu_0 * h, lambda_0 * h
        U, sigma, V = ti.svd(F[p])
        J = sigma[0, 0] * sigma[1, 1]  # determinant
        P = (
            2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() +
            ti.Matrix.identity(float, 2) * la * J * (J - 1)
        )
        # 3.3) Convert stress to grid forces
        stress = -dt * p_vol * 4 * inv_dx * inv_dx * P
        affine = stress + p_mass * C[p]
        # 3.4) Scatter mass & momentum to grid
        for ii, jj in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([ii, jj])
            dpos = (offs.cast(float) - fx) * dx
            weight = w[ii][0] * w[jj][1]
            grid_v[base + offs] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offs] += weight * p_mass

    # 4) Inject rigid-box velocity into marked grid nodes
    for i, j in grid_m:
        if grid_A[i, j] == 1 and grid_m[i, j] > 0:
            # Overwrite node momentum to match rigid motion
            grid_v[i, j] = r_vel[None] * grid_m[i, j]

    # 5) Grid update: momentum→velocity, gravity, BCs
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            # Normalize to get velocity
            grid_v[i, j] = grid_v[i, j] / grid_m[i, j]
            # Apply gravity force
            grid_v[i, j][1] -= dt * gravity
            # Simple boundary conditions: zero out-of-bounds velocity
            if i < 3 and grid_v[i, j][0] < 0: grid_v[i, j][0] = 0
            if i > n_grid-3 and grid_v[i, j][0] > 0: grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: grid_v[i, j][1] = 0
            if j > n_grid-3 and grid_v[i, j][1] > 0: grid_v[i, j][1] = 0

    # 6) Grid-to-Particle (G2P)
    for p in range(n_particles):
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx   = x[p] * inv_dx - base.cast(float)
        w = [
            0.5 * (1.5 - fx)**2,
            0.75 - (fx - 1.0)**2,
            0.5 * (fx - 0.5)**2,
        ]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for ii, jj in ti.static(ti.ndrange(3, 3)):
            node_idx = base + ti.Vector([ii, jj])
            gv = grid_v[node_idx]
            # If node under rigid, use rigid velocity directly
            if grid_A[node_idx] == 1:
                gv = r_vel[None]
            dpos = ti.Vector([ii, jj]).cast(float) - fx
            weight = w[ii][0] * w[jj][1]
            new_v += weight * gv
            new_C += 4 * inv_dx * weight * gv.outer_product(dpos)
        # Update particle velocity, affine field, and advect
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]

# ----------------
# Main Entry Point
# ----------------
initialize()  # Set up particles and box

gui = ti.GUI("MPM with Cyclic Rigid Box", res=800, background_color=0x112F41)
frame = 0
while gui.running:
    # Advance simulation by ~5e-3 seconds per frame
    for _ in range(int(5e-3 // dt)):
        substep()
    # Draw soft-body particles
    gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
    # Draw rigid box as white rectangle
    p1 = to_box[None].to_numpy()
    p2 = from_box[None].to_numpy()
    gui.rect(topleft=p1, bottomright=p2, color=0xFFFFFF)
    gui.show()
    frame += 1