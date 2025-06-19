import taichi as ti

# --- Parameters ---
ti.init(arch=ti.cpu)

gravity = 9.8
show_stress = True    # toggle stress heatmap
output_force = True   # toggle numeric force display

# Viscoelastic SLS model parameters
E1, E2, eta = 200.0, 300.0, 100.0
nu = 0.3

# Lamé constants for Maxwell branch (E1) and parallel spring (E2)
lam1 = E1 * nu / (1 - nu**2)
mu1  = E1 / (2 * (1 + nu))
lam2 = E2 * nu / (1 - nu**2)
mu2  = E2 / (2 * (1 + nu))

# Domain and grid
domain_size = 1.0
Ngrid = 128
dx = domain_size / Ngrid
inv_dx = float(Ngrid)
dt = 2e-4

# Soft body rectangle
body_w, body_h = 0.5, 0.2
body_x0, body_y0 = 0.25, 0.0

# Particle grid
spacing = dx * 0.5
Nx = int(body_w / spacing)
Ny = int(body_h / spacing)
N_particles = Nx * Ny

# Fields
p_pos    = ti.Vector.field(2, float, N_particles)
p_vel    = ti.Vector.field(2, float, N_particles)
p_mass   = ti.field(float, N_particles)
p_vol0   = ti.field(float, N_particles)
p_eps    = ti.Matrix.field(2, 2, float, N_particles)
p_sigma_m = ti.Matrix.field(2, 2, float, N_particles)
p_sigma_e = ti.Matrix.field(2, 2, float, N_particles)

grid_v = ti.Vector.field(2, float, (Ngrid, Ngrid))
grid_m = ti.field(float, (Ngrid, Ngrid))

# Roller state
roller_radius = 0.05
roller_pos = ti.Vector.field(2, float, ())
roller_vel = ti.Vector.field(2, float, ())
roller_mass = 5.0
spring_k, damping_c = 100.0, 30.0
anchor_y = 0.5
slide_speed = 2.0
t_slide_delay = 0.2
start_contact = False
contact_time = 0.0

# Initialization
@ti.kernel
def init_particles():
    for i in range(N_particles):
        xi = i // Ny
        yi = i % Ny
        p_pos[i] = [body_x0 + (xi + 0.5) / Nx * body_w,
                    body_y0 + (yi + 0.5) / Ny * body_h]
        p_vel[i] = [0.0, 0.0]
        p_vol0[i] = spacing**2
        density = 1000.0
        p_mass[i] = density * p_vol0[i]
        p_eps[i] = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        p_sigma_m[i] = ti.Matrix.zero(float, 2, 2)
        p_sigma_e[i] = ti.Matrix.zero(float, 2, 2)
    roller_pos[None] = [body_x0 + roller_radius, anchor_y]
    roller_vel[None] = [0.0, 0.0]

# B-spline weights and derivatives
@ti.func
def bspline_weights(x):
    w0 = 0.5 * (1.5 - x) ** 2
    w1 = 0.75 - (x - 1.0) ** 2
    w2 = 0.5 * (x - 0.5) ** 2
    dw0 = -(1.5 - x)
    dw1 = -2 * (x - 1.0)
    dw2 = (x - 0.5)
    return ti.Vector([w0, w1, w2]), ti.Vector([dw0, dw1, dw2])

# P2G
@ti.kernel
def particle_to_grid():
    for I in ti.grouped(grid_m):
        grid_m[I] = 0
        grid_v[I] = [0, 0]
    for p in range(N_particles):
        base = (p_pos[p] * inv_dx).cast(int)
        fx = p_pos[p].x * inv_dx - base.x
        fy = p_pos[p].y * inv_dx - base.y
        wx, dwx = bspline_weights(fx)
        wy, dwy = bspline_weights(fy)
        sigma = p_sigma_e[p] + p_sigma_m[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            node = base + ti.Vector([i, j])
            if 0 <= node.x < Ngrid and 0 <= node.y < Ngrid:
                w = wx[i] * wy[j]
                grad_N = ti.Vector([dwx[i] * wy[j], wx[i] * dwy[j]]) * inv_dx
                grid_m[node] += w * p_mass[p]
                grid_v[node] += w * p_mass[p] * p_vel[p]
                f = -p_vol0[p] * (sigma @ grad_N)
                grid_v[node] += f * dt / (grid_m[node] + 1e-12)

# G2P and stress update
@ti.kernel
def grid_to_particle():
    # normalize grid velocities & BC
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] = grid_v[I] / grid_m[I]
        if I.y < 2:
            grid_v[I] = [0, 0]
    # update particles
    for p in range(N_particles):
        base = (p_pos[p] * inv_dx).cast(int)
        fx = p_pos[p].x * inv_dx - base.x
        fy = p_pos[p].y * inv_dx - base.y
        wx, _ = bspline_weights(fx)
        wy, _ = bspline_weights(fy)
        v_new = ti.Vector([0.0, 0.0])
        for i, j in ti.static(ti.ndrange(3, 3)):
            node = base + ti.Vector([i, j])
            if 0 <= node.x < Ngrid and 0 <= node.y < Ngrid:
                w = wx[i] * wy[j]
                v_new += w * grid_v[node]
        p_vel[p] = v_new
        p_pos[p] += dt * p_vel[p]
        # compute gradient via central difference
        i0 = ti.min(ti.max(base.x, 1), Ngrid - 2)
        j0 = ti.min(ti.max(base.y, 1), Ngrid - 2)
        dvx_dx = (grid_v[i0+1, j0].x - grid_v[i0-1, j0].x) / (2*dx)
        dvx_dy = (grid_v[i0, j0+1].x - grid_v[i0, j0-1].x) / (2*dx)
        dvy_dx = (grid_v[i0+1, j0].y - grid_v[i0-1, j0].y) / (2*dx)
        dvy_dy = (grid_v[i0, j0+1].y - grid_v[i0, j0-1].y) / (2*dx)
        grad_v = ti.Matrix([[dvx_dx, dvx_dy], [dvy_dx, dvy_dy]])
        strain_rate = 0.5 * (grad_v + grad_v.transpose())
        d_eps = strain_rate * dt
        p_eps[p] += d_eps
        I_mat = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        # Maxwell branch
        d_sig_m = lam1 * d_eps.trace() * I_mat + 2 * mu1 * d_eps
        p_sigma_m[p] = p_sigma_m[p] + d_sig_m - (E1/eta) * p_sigma_m[p] * dt
        # parallel spring
        p_sigma_e[p] += lam2 * d_eps.trace() * I_mat + 2 * mu2 * d_eps

# collision & roller force
@ti.kernel
def handle_collisions() -> float:
    imp = 0.0
    r_c = roller_pos[None]
    r_v = roller_vel[None]
    for p in range(N_particles):
        vec = p_pos[p] - r_c
        dist = vec.norm()
        if dist < roller_radius:
            if dist < 1e-6:
                vec = ti.Vector([0.0, 1.0]); dist = 1e-6
            n = vec / dist
            p_pos[p] = r_c + n * roller_radius
            v_rel = p_vel[p] - r_v
            vn = v_rel.dot(n)
            if vn < 0:
                p_vel[p] -= vn * n
                imp += p_mass[p] * (-vn)
    return imp

# GUI
gui = ti.GUI("MPM Massage", res=(600, 600))
init_particles()
time = 0.0
while gui.running:
    # detect contact & start slide
    if not start_contact and roller_pos[None].y - roller_radius <= body_y0 + body_h:
        start_contact = True; contact_time = time
    if start_contact and time - contact_time >= t_slide_delay:
        roller_vel[None].x = slide_speed
    # simulation steps
    particle_to_grid()
    grid_to_particle()
    imp = handle_collisions()
    # roller dynamics
    Fg = -roller_mass * gravity
    ext = anchor_y - roller_pos[None].y
    Fs = spring_k * ext; Fd = -damping_c * roller_vel[None].y
    Fc = imp / dt
    Fnet = Fg + Fs + Fd + Fc
    acc = Fnet / roller_mass
    roller_vel[None].y += acc * dt
    roller_pos[None].y += roller_vel[None].y * dt
    roller_pos[None].x += roller_vel[None].x * dt
    # draw
    gui.clear(0x112F41)
    pos_np = p_pos.to_numpy()
    if show_stress:
        sigma = p_sigma_e.to_numpy() + p_sigma_m.to_numpy()
        vm = [((s[0,0]**2 - s[0,0]*s[1,1] + s[1,1]**2) + 3*s[0,1]**2)**0.5 for s in sigma]
        max_vm = max(vm)
        colors = []
        for v in vm:
            t = v / (max_vm+1e-6)
            R = int(t*255); B = int((1-t)*255)
            colors.append((R<<16) | B)
        gui.circles(pos=pos_np, color=0xAAAAFF, radius=2)
    else:
        gui.circles(pos=pos_np, color=0x4FB99F, radius=2)
    gui.circle(pos=roller_pos.to_numpy(), color=0xAAAAFF, radius=roller_radius*600)
    if output_force:
        gui.text(f"Force: {imp/dt:.2f} N", pos=(0.02,0.95), color=0xFFFFFF)
    gui.text(f"Time: {time:.2f}s", pos=(0.8,0.95), color=0xFFFFFF)
    gui.show()
    time += dt
