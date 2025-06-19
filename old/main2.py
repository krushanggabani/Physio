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

# ─── Material (Purely Hyper‐Viscoelastic) ──────────────────────────────────
p_rho   = 1.0
p_vol   = (dx * 0.5)**2
p_mass  = p_rho * p_vol
E       = 5e4
nu      = 0.2
bulk_modulus  = E / (3 * (1 - 2 * nu))        # κ
shear_modulus = E / (2 * (1 + nu))            # μ
viscosity     = 0.0                           # no extra Newtonian viscosity

bulk_modulus = 10000
shear_modulus = 20000
viscosity = 10

print(bulk_modulus)
print(shear_modulus)

@ti.data_oriented
class HyperelasticModel:
    def __init__(self, κ, μ, η):
        self.kappa = κ
        self.mu    = μ
        self.eta   = η

    @ti.func
    def compute_stress(self, F_e, grad_v, J_e):
        # Kirchhoff stress = volumetric + deviatoric + viscous
        I = ti.Matrix.identity(ti.f32, 2)
        # volumetric part
        tau_v = 0.5 * self.kappa * (J_e * J_e - 1.0) * I
        # deviatoric part
        A = F_e * (J_e ** (-0.5))
        AAT = A @ A.transpose()
        trace_A = AAT[0,0] + AAT[1,1]
        tau_s = self.mu * (AAT - 0.5 * trace_A * I)
        # Newtonian viscous part
        tau_N = self.eta * 0.5 * (grad_v + grad_v.transpose())
        return tau_v + tau_s + tau_N

model = HyperelasticModel(bulk_modulus, shear_modulus, viscosity)

# ─── Boundaries & Roller Params ────────────────────────────────────────────
floor_level = 0.0
L1, L2      = 0.12, 0.10
theta1      = np.array([np.pi/15], dtype=np.float32)
theta2      = np.array([0.0],     dtype=np.float32)
dtheta1     = np.zeros(1, dtype=np.float32)
dtheta2     = np.zeros(1, dtype=np.float32)
theta1_rest = theta1.copy()
theta2_rest = theta2.copy()
k1, k2      = 5, 5
b1, b2      = 2.0, 2.0
I1          = L1**2 / 12.0
I2          = L2**2 / 12.0
base_x      = 0.4
y0          = 0.4
A           = 0.1
ω           = 0.5
base_y      = y0
time_t      = 0.0
roller_radius   = 0.025
roller_center   = ti.Vector.field(dim, ti.f32, 1)
roller_velocity = ti.Vector.field(dim, ti.f32, 1)
contact_force   = ti.Vector.field(dim, ti.f32, 1)

# ─── MPM Fields ────────────────────────────────────────────────────────────
x      = ti.Vector.field(dim, ti.f32, n_particles)
v      = ti.Vector.field(dim, ti.f32, n_particles)
F      = ti.Matrix.field(dim, dim, ti.f32, n_particles)
C      = ti.Matrix.field(dim, dim, ti.f32, n_particles)  # velocity gradient
J      = ti.field(ti.f32, n_particles)
grid_v = ti.Vector.field(dim, ti.f32, (n_grid, n_grid))
grid_m = ti.field(ti.f32, (n_grid, n_grid))

@ti.kernel
def init_mpm():
    for p in range(n_particles):
        x[p] = [0.3 + ti.random() * 0.4, ti.random() * 0.2]
        v[p] = [0.0, 0.0]
        F[p] = ti.Matrix.identity(ti.f32, dim)
        J[p] = 1.0
        C[p] = ti.Matrix.zero(ti.f32, dim, dim)
    base = ti.Vector([base_x, base_y])
    j2   = base + ti.Vector([ti.sin(theta1[0]), -ti.cos(theta1[0])]) * L1
    ee   = j2 + ti.Vector([ti.sin(theta1[0]+theta2[0]), -ti.cos(theta1[0]+theta2[0])]) * L2
    roller_center[0]   = ee
    roller_velocity[0] = ti.Vector.zero(ti.f32, dim)
    contact_force[0]   = ti.Vector.zero(ti.f32, dim)

@ti.kernel
def p2g():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, dim)
        grid_m[I] = 0.0
    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx   = Xp - base.cast(ti.f32)
        w    = [0.5 * (1.5 - fx)**2,
                0.75 - (fx - 1.0)**2,
                0.5 * (fx - 0.5)**2]
        # compute stress (no plastic!)
        τ = model.compute_stress(F[p], C[p], J[p])
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * τ
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([i, j])
            dpos = (offs.cast(ti.f32) - fx) * dx
            wt   = w[i].x * w[j].y
            grid_v[base + offs] += wt * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offs] += wt * p_mass

@ti.kernel
def apply_grid_forces_and_detect():
    contact_force[0] = ti.Vector.zero(ti.f32, dim)
    for I in ti.grouped(grid_m):
        m = grid_m[I]
        if m > 0:
            v_old = grid_v[I] / m
            v_new = v_old + dt * ti.Vector([0.0, -9.8])
            pos   = I.cast(ti.f32) * dx
            rel   = pos - roller_center[0]
            if rel.norm() < roller_radius:
                n      = rel.normalized()
                rv     = roller_velocity[0]
                v_norm = n * n.dot(rv)
                v_tan  = v_old - n * n.dot(v_old)
                v_new  = v_tan + v_norm
                delta_v = v_new - v_old
                contact_force[0] += m * delta_v / dt
            if pos.y < floor_level + dx:
                if v_new.y < 0: v_new.y = 0
                v_new.x = 0
            if pos.x < dx or pos.x > 1 - dx:
                v_new.x = 0
            grid_v[I] = v_new * m

@ti.kernel
def g2p():
    for p in range(n_particles):
        Xp   = x[p] * inv_dx
        base = (Xp - 0.5).cast(int)
        fx   = Xp - base.cast(ti.f32)
        w    = [0.5 * (1.5 - fx)**2,
                0.75 - (fx - 1.0)**2,
                0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offs = ti.Vector([i, j])
            dpos = (offs.cast(ti.f32) - fx) * dx
            wt   = w[i].x * w[j].y
            gv   = grid_v[base + offs] / grid_m[base + offs]
            new_v += wt * gv
            new_C += 4 * inv_dx * wt * gv.outer_product(dpos)
        v[p] = new_v
        C[p] = new_C
        x[p] += dt * new_v
        if x[p].y < floor_level:
            x[p].y = floor_level
            v[p] = ti.Vector.zero(ti.f32, dim)
        if x[p].x < dx: x[p].x, v[p].x = dx, 0
        if x[p].x > 1-dx: x[p].x, v[p].x = 1-dx, 0
        if x[p].y > 1-dx: x[p].y, v[p].y = 1-dx, 0
        F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * new_C) @ F[p]
        J[p] = F[p].determinant()

def update_base_and_arm():
    global time_t, base_y
    time_t += dt * 15
    base_y = y0 + A * np.cos(ω * time_t)
    Fc   = contact_force[0].to_numpy()
    base = np.array([base_x, base_y], np.float32)
    j2   = base + np.array([np.sin(theta1[0]), -np.cos(theta1[0])]) * L1
    ee_old = roller_center[0].to_numpy()
    ee_new = j2 + np.array([np.sin(theta1[0]+theta2[0]), -np.cos(theta1[0]+theta2[0])]) * L2
    rv     = (ee_new - ee_old) / dt
    roller_center[0]   = ee_new.tolist()
    roller_velocity[0] = rv.tolist()
    r1    = ee_new - base
    r2    = ee_new - j2
    tau1_c= r1[1]*Fc[0] - r1[0]*Fc[1]
    tau2_c= r2[1]*Fc[0] - r2[0]*Fc[1]
    tau1  = 2*tau1_c - k1*(theta1[0]-theta1_rest[0]) - b1*dtheta1[0]
    tau2  = 5*tau2_c - k2*(theta2[0]-theta2_rest[0]) - b2*dtheta2[0]
    dtheta1[0] += (tau1/I1) * dt
    theta1[0]  += dtheta1[0] * dt
    dtheta2[0] += (tau2/I2) * dt
    theta2[0]  += dtheta2[0] * dt

# ─── Run ────────────────────────────────────────────────────────────────────
init_mpm()
gui = ti.GUI('MPM + Passive Arm (Elastic Only)', res=(512, 512))
while gui.running:
    for _ in range(15):
        p2g()
        apply_grid_forces_and_detect()
        g2p()
        update_base_and_arm()
    gui.circles(x.to_numpy(), radius=1.5, color=0x66CCFF)
    base_pt = np.array([base_x, base_y], np.float32)
    j2      = base_pt + np.array([np.sin(theta1[0]), -np.cos(theta1[0])]) * L1
    ee      = roller_center[0].to_numpy()
    gui.line(begin=base_pt, end=j2, radius=2, color=0x000050)
    gui.line(begin=j2, end=ee,   radius=2, color=0x000050)
    gui.circle(base_pt, radius=4, color=0xFF0000)
    gui.circle(j2,      radius=4, color=0xFF0000)
    gui.circle(ee,      radius=int(roller_radius*512), color=0xFF0000)
    gui.text(f'Force: {contact_force[0]} N', pos=(0.02,0.95), color=0xFFFFFF)
    gui.show()
