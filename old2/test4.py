import taichi as ti

ti.init(arch=ti.cpu)

# — window —
window_size = 800
gui = ti.GUI("Viscoelastic MPM", (window_size, window_size))
dt         = 2e-4
dim        = 3
dtype      = ti.f32
max_steps  = 1000
default_gravity = (0,-9.81,0)

# — particle patch —
center       = ti.Vector([0.5, 0.05,0.0], ti.f32)
width, height, spacing = 0.5, 0.25, 0.1
nx = max(1, int(width  / spacing) + 1)
ny = max(1, int(height / spacing) + 1)
n_particles = nx * ny
print(n_particles)

ground_friction = 20

# — MPM grid parameters —
n_grid     = 128
dx         = 1.0 / n_grid
inv_dx     = float(n_grid)


substeps = int(2e-3 / dt)

# — fields —
x = ti.Vector.field(dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # position
v = ti.Vector.field(dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # velocity
C = ti.Matrix.field(dim, dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # affine velocity field
F = ti.Matrix.field(dim, dim, dtype=dtype, shape=(max_steps, n_particles), needs_grad=True)  # deformation gradient

F_tmp = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles), needs_grad=True)  # deformation gradient
U = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)
V = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)
sig = ti.Matrix.field(dim, dim, dtype=dtype, shape=(n_particles,), needs_grad=True)

res = res = (n_grid, n_grid) if dim == 2 else (n_grid, n_grid, n_grid)
grid_v_in = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)  # grid node momentum/velocity
grid_m = ti.field(dtype=dtype, shape=res, needs_grad=True)  # grid node mass
grid_v_out = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)  # grid node momentum/velocity



grid_v_mixed = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)
v_tmp = ti.Vector.field(dim, dtype=dtype, shape=(n_particles), needs_grad=True)
v_tgt = ti.Vector.field(dim, dtype=dtype, shape=(n_particles), needs_grad=True)

gravity = ti.Vector.field(dim, dtype=dtype, shape=())
mu = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)
lam = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)
yield_stress = ti.field(dtype=dtype, shape=(n_particles,), needs_grad=False)


gravity[None] = default_gravity
yield_stress.fill(30)
E,nu = 3e3,0.2
_mu, _lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
mu.fill(_mu)
lam.fill(_lam)

p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho

@ti.kernel
def seed():
    I = ti.Matrix.identity(ti.f32,2)
    for p in range(n_particles):
        i = p // ny
        j = p % ny
        x[1,p]       = center + ti.Vector([-width*0.5 + i*spacing, j*spacing,0])
        v[1,p]       = ti.Vector([0.0, 0.0,0.0])
        
        # F[p]       = I
        # C[p]       = ti.Matrix.zero(ti.f32, 2, 2)
        # sigma_v[p] = ti.Matrix.zero(ti.f32, 2, 2)



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

    
        grid_v_mixed[I] = zero
        grid_v_mixed.grad[I] = zero

    for p in range(0, n_particles):
        
        v_tmp[p] = zero
        v_tmp.grad[p] = zero
        v_tgt[p] = zero
        v_tgt.grad[p] = zero



@ti.kernel
def compute_F_tmp(f: ti.i32):
    for p in range(0, n_particles):  # Particle state update and scatter to grid (P2G)
        F_tmp[p] = (ti.Matrix.identity(dtype, dim) + dt * C[f, p]) @ F[f, p]
    


@ti.kernel
def svd():
    for p in range(0, n_particles):
        U[p], sig[p], V[p] = ti.svd(F_tmp[p])

@ti.func
def stencil_range():
    return ti.ndrange(*((3, ) * dim))

@ti.kernel
def p2g(f: ti.i32):
    for p in range(0, n_particles):
        # particle collision
        collision_impulse = ti.Vector.zero(dtype, 3)
        # if ti.static(collision_type == CONTACT_PARTICLE and n_primitive > 0):
        #     for i in ti.static(range(n_primitive)):
        #         if primitives_contact[i]:
        #             collision_impulse += primitives[i].collide_particle(f, x[f, p], v[f, p], dt)

        # control signal
        control_impulse = ti.Vector.zero(dtype, 3)
        # if ti.static(n_control > 0):
        #     control_idx = control_idx[p]
        #     if control_idx >= 0:
        #         control_impulse += 6e-4 * action[control_idx] * dt

        base = (x[f, p] * inv_dx - 0.5).cast(int)
        fx = x[f, p] * inv_dx - base.cast(dtype)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        # stress
        stress = ti.Matrix.zero(dtype, dim, dim)
        new_F = F_tmp[p]
        J = new_F.determinant()
      
        stress = mu[p] * (new_F @ new_F.transpose()) + \
            ti.Matrix.identity(dtype, dim) * (lam[p] * ti.log(J) - mu[p])
        
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[f, p]

        F[f + 1, p] = new_F
        
        # update grid
        for offset in ti.static(ti.grouped(stencil_range())):
            dpos = (offset.cast(dtype) - fx) * dx
            weight = ti.cast(1.0, dtype)
            for d in ti.static(range(dim)):
                weight *= w[offset[d]][d]

            x = base + offset

            grid_v_in[base + offset] += weight * (p_mass * v[f, p] + affine @ dpos + collision_impulse + control_impulse)
            grid_m[base + offset] += weight * p_mass

    
@ti.kernel
def g2p( f: ti.i32):
    for p in range(0, n_particles):  # grid to particle (G2P)
        base = (x[f, p] * inv_dx - 0.5).cast(int)
        fx = x[f, p] * inv_dx - base.cast(dtype)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(dtype, dim)
        new_C = ti.Matrix.zero(dtype, dim, dim)
        for offset in ti.static(ti.grouped(stencil_range())):
            dpos = offset.cast(dtype) - fx
            g_v = grid_v_out[base + offset]
            weight = ti.cast(1.0, dtype)
            for d in ti.static(range(dim)):
                weight *= w[offset[d]][d]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[f + 1, p], C[f + 1, p] = new_v, new_C

        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]  # advection


@ti.kernel
def grid_op( f: ti.i32):
    for I in ti.grouped(grid_m):
        if grid_m[I] > 1e-10:  # No need for epsilon here, 1e-10 is to prevent potential numerical problems ..
            v_out = (1 / grid_m[I]) * grid_v_in[I]    # Momentum to velocity
            v_out += dt * gravity[None]               # gravity

            v_out = boundary_condition(I, v_out)
            grid_v_out[I] = v_out

@ti.func
def boundary_condition( I, v_out):
    bound = 3
    v_in2 = v_out
    for d in ti.static(range(dim)):
        if I[d] < bound and v_out[d] < 0:
            v_out[d] = 0
        if I[d] > n_grid - bound and v_out[d] > 0: 
            v_out[d] = 0

        if d == 1 and I[d] < bound and ti.static(ground_friction >= 10.):
            v_out[0] = v_out[1] 
        
    return v_out

# def grid_op_mixed( f):
#         grid_op_mixed1(f)
#         grid_op_mixed2(f)
#         grid_op_mixed3(f)
#         grid_op_mixed4(f)


# @ti.kernel
# def grid_op_mixed1(f: ti.int32):
#     for I in ti.grouped(grid_m):
#         if grid_m[I] > 1e-10:
#             v_out = (1 / grid_m[I]) * grid_v_in[I]  # Momentum to velocity
#             v_out += dt * gravity[None]  # gravity
#             v_out = boundary_condition(I, v_out)
#             grid_v_mixed[I] = v_out
#             grid_v_out[I] += grid_v_mixed[I]

# @ti.kernel
# def grid_op_mixed2( f: ti.int32):
#     for p in range(n_particles):
#         base = (x[f, p] * inv_dx - 0.5).cast(int)
#         fx = x[f, p] * inv_dx - base.cast(float)
#         w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
#         new_v = ti.Vector.zero(dtype, dim)
#         for offset in ti.static(ti.grouped(stencil_range())):
#             g_v = grid_v_mixed[base + offset]
#             weight = ti.cast(1.0, dtype)
#             for d in ti.static(range(dim)):
#                 weight *= w[offset[d]][d]
#             new_v += weight * g_v
#         v_tmp[p] = new_v

# @ti.kernel
# def grid_op_mixed3( f: ti.int32):
#     for p in range(n_particles):
#         v_tgt = v_tmp[p]
#         life = 1 / (substeps - f % substeps)
#         for i in ti.static(range(n_primitive)):
#             if primitives_contact[i]:
#                 v_tgt = primitives[i].collide_mixed(f, x[f, p], v_tgt, p_mass, dt, life)
#         v_tgt[p] = v_tgt

# @ti.kernel
# def grid_op_mixed4( f: ti.int32):
#     for p in range(n_particles):
#         base = (x[f, p] * inv_dx - 0.5).cast(int)
#         fx = x[f, p] * inv_dx - base.cast(float)
#         w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
#         alpha = 2.0
#         for offset in ti.static(ti.grouped(stencil_range())):
#             weight = ti.cast(1.0, dtype)
#             for d in ti.static(range(dim)):
#                 weight *= w[offset[d]][d]
#             if grid_m[base + offset] > 1e-10:
#                 grid_v_out[base + offset] -= alpha * weight * (v_tmp[p] - v_tgt[p])

def substep(s):
    clear_grid()
    compute_F_tmp(s)

    print(F_tmp)
    svd()
    p2g(s)

    # update_rigid_body(s,dt)
    # grid_op_mixed(s)
    grid_op(s)
    g2p(s)


# — main —
seed()



for s in range(max_steps):
    substep(s)

    gui.clear(0x112F41)
    # print(x)
    # gui.circles(x.to_numpy(), radius=2, color=0xED553B)
    gui.show()
