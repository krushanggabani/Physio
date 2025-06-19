import taichi as ti

ti.init(arch=ti.cpu)  # or ti.gpu

# -----------------------
# Simulation parameters
# -----------------------
dt = 1e-4
num_panda = 1000          # panda particles
num_balls = 1             # one ball cluster
particles_per_ball = 100
num_particles = num_panda + num_balls * particles_per_ball

grid_size = 64            # 64x64 grid

domain_size = 1.0
dx = domain_size / grid_size
gravity = ti.Vector([0.0, -9.8])

# -----------------------
# Material parameters
# -----------------------
# Viscoelastic panda
d_mu_e_p, d_lam_e_p = 50.0, 50.0
d_mu_n_p, d_lam_n_p = 50.0, 50.0
d_visc_p = 5.0
# Elastic ball
d_mu_e_b, d_lam_e_b = 200.0, 200.0
d_mu_n_b, d_lam_n_b = 0.0, 0.0
d_visc_b = 1e-8

# -----------------------
# Taichi fields
# -----------------------
x    = ti.Vector.field(2, ti.f32, shape=num_particles)
vel  = ti.Vector.field(2, ti.f32, shape=num_particles)
F_N  = ti.Matrix.field(2, 2, ti.f32, shape=num_particles)
F_V  = ti.Matrix.field(2, 2, ti.f32, shape=num_particles)
# material params per particle
type_id = ti.field(ti.i32, shape=num_particles)
mu_e    = ti.field(ti.f32, shape=num_particles)
lam_e   = ti.field(ti.f32, shape=num_particles)
mu_n    = ti.field(ti.f32, shape=num_particles)
lam_n   = ti.field(ti.f32, shape=num_particles)
visc    = ti.field(ti.f32, shape=num_particles)
# grid fields
grid_v = ti.Vector.field(2, ti.f32, shape=(grid_size, grid_size))
grid_m = ti.field(ti.f32, shape=(grid_size, grid_size))

# -----------------------
# Initialization
# -----------------------
@ti.kernel
def initialize():
    # Panda
    for i in range(num_panda):
        x[i] = [0.3 + ti.random() * 0.4, ti.random() * 0.2]
        vel[i] = ti.Vector([0.0, 0.0])
        mu_e[i], lam_e[i] = d_mu_e_p, d_lam_e_p
        mu_n[i], lam_n[i] = d_mu_n_p, d_lam_n_p
        visc[i] = d_visc_p
        F_N[i] = ti.Matrix.identity(ti.f32, 2)
        F_V[i] = ti.Matrix.identity(ti.f32, 2)
        type_id[i] = 0
    # Ball cluster
    for b in range(num_balls):
        base = num_panda + b * particles_per_ball
        center = ti.Vector([0.5, 0.8])
        for j in range(particles_per_ball):
            idx = base + j
            r = 0.03 * ti.sqrt(ti.random())
            theta = 2 * 3.1415926 * ti.random()
            x[idx] = center + ti.Vector([ti.cos(theta), ti.sin(theta)]) * r
            vel[idx] = ti.Vector([0.0, 0.0])
            mu_e[idx], lam_e[idx] = d_mu_e_b, d_lam_e_b
            mu_n[idx], lam_n[idx] = d_mu_n_b, d_lam_n_b
            visc[idx] = d_visc_b
            F_N[idx] = ti.Matrix.identity(ti.f32, 2)
            F_V[idx] = ti.Matrix.identity(ti.f32, 2)
            type_id[idx] = 1

# -----------------------
# Particle to Grid (P2G)
# -----------------------
@ti.kernel
def p2g():
    for I in ti.grouped(grid_m):
        grid_m[I] = 0.0
        grid_v[I] = ti.Vector([0.0, 0.0])
    for p in range(num_particles):
        base = (x[p] / dx - 0.5).cast(int)
        # total deformation\        
        F_tot = F_N[p] @ F_V[p]
        mu = mu_e[p] + mu_n[p]
        lam = lam_e[p] + lam_n[p]
        # linear elastic stress\        
        eps = 0.5 * (F_tot + F_tot.transpose()) - ti.Matrix.identity(ti.f32, 2)
        tr = eps[0,0] + eps[1,1]
        sigma = ti.Matrix([[lam * tr + 2 * mu * (eps[0,0] - tr * 0.5), 2 * mu * eps[0,1]],
                           [2 * mu * eps[1,0], lam * tr + 2 * mu * (eps[1,1] - tr * 0.5)]])
        vol = dx * dx * 0.5
        for i,j in ti.static(ti.ndrange(2,2)):
            node = base + ti.Vector([i,j])
            # guard bounds
            if 0 <= node[0] < grid_size and 0 <= node[1] < grid_size:
                dpos = x[p] / dx - node.cast(ti.f32)
                w = (1 - abs(dpos[0])) * (1 - abs(dpos[1]))
                grid_m[node] += w
                grid_v[node] += w * vel[p]
                grad_w = ti.Vector([(-1 if dpos[0]<0 else 1)/dx, (-1 if dpos[1]<0 else 1)/dx])
                grid_v[node] -= dt * vol * (sigma @ grad_w)

# -----------------------
# Grid operations
# -----------------------
@ti.kernel
def grid_op():
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] = grid_v[I] / grid_m[I] + gravity * dt
        # ground collision
        if I[1] < 1:
            grid_v[I] = ti.Vector([0.0,0.0])

# -----------------------
# Grid to Particle (G2P)
# -----------------------
@ti.kernel
def g2p():
    for p in range(num_particles):
        base = (x[p] / dx - 0.5).cast(int)
        new_v = ti.Vector([0.0,0.0])
        grad_v = ti.Matrix([[0.0,0.0],[0.0,0.0]])
        for i,j in ti.static(ti.ndrange(2,2)):
            node = base + ti.Vector([i,j])
            if 0 <= node[0] < grid_size and 0 <= node[1] < grid_size:
                dpos = x[p]/dx - node.cast(ti.f32)
                w = (1 - abs(dpos[0]))*(1-abs(dpos[1]))
                new_v += w * grid_v[node]
                grad_w = ti.Vector([(-1 if dpos[0]<0 else 1)/dx,(-1 if dpos[1]<0 else 1)/dx])
                # accumulate grad_v
                for u in ti.static(range(2)):
                    for v_ in ti.static(range(2)):
                        grad_v[u,v_] += grid_v[node][u] * grad_w[v_]
        vel[p] = new_v
        x[p] += new_v * dt
        # clamp
        x_val = x[p]
        x_val[0] = min(max(x_val[0], 0.0), domain_size)
        x_val[1] = min(max(x_val[1], 0.0), domain_size)
        x[p] = x_val
        # viscoelastic update
        F_old = F_N[p] @ F_V[p]
        F_N_tr = (ti.Matrix.identity(ti.f32,2) + dt*grad_v) @ F_N[p]
        F_tot = (ti.Matrix.identity(ti.f32,2) + dt*grad_v) @ F_old
        U,sig,V = ti.svd(F_N_tr)
        eps = ti.Vector([ti.log(sig[0,0]), ti.log(sig[1,1])])
        factor = 1.0/(1.0 + dt*(mu_n[p]/visc[p]))
        eps2 = eps * factor
        sig2 = ti.Matrix([[ti.exp(eps2[0]),0.0],[0.0,ti.exp(eps2[1])]])
        F_N[p] = U @ sig2 @ V.transpose()
        F_V[p] = F_N[p].inverse() @ F_tot

# -----------------------
# Main Loop
# -----------------------
initialize()
import numpy as np

gui = ti.GUI('Viscoelastic Panda MPM', res=(512,512))
while gui.running:
    for _ in range(5): p2g(); grid_op(); g2p()
    gui.clear(0x112F41)
    pts = x.to_numpy() / domain_size
    gui.circles(pts, radius=1.5, color=0xED553B)
    gui.show()
