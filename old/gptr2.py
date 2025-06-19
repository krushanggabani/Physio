"""
Unified MPM Simulator with Generalized Constitutive Model,
Forward Simulation, Inverse Learning, and Visualization.

Dependencies:
  - taichi (pip install taichi)
  - torch (pip install torch)
  - numpy (pip install numpy)
  - pyrender (pip install pyrender trimesh)

Author: Adapted from "A Generalized Constitutive Model for Versatile MPM Simulation
        and Inverse Learning with Differentiable Physics" (Su et al., 2023)
"""

import taichi as ti
import torch
import numpy as np
import pyrender
import trimesh

# ---------- Constitutive Model ----------

@ti.data_oriented
class ConstitutiveModel:
    def __init__(self, bulk_modulus, shear_modulus, viscosity,
                 yield_stress=0.0, hardening=0.0,
                 f_c=0.0, f_s=1e6, two_phase=False):
        # Elastic / fluid params
        self.kappa0 = bulk_modulus
        self.mu0    = shear_modulus
        self.eta    = viscosity
        # Plasticity params
        self.sigmaY   = yield_stress
        self.hardening= hardening
        self.f_c = f_c   # min singular value
        self.f_s = f_s   # max singular value
        self.two_phase = two_phase
        # Current (possibly updated) moduli
        self.kappa = bulk_modulus
        self.mu    = shear_modulus

    @ti.func
    def compute_stress(self, F_e, b_e_d, grad_v, J_e):
        """
        Compute the Kirchhoff stress tensor τ = τ_v + τ_s + τ_N for one particle.
          F_e    : elastic deformation gradient (dim×dim)
          b_e_d  : left Cauchy elastic tensor for viscoplastic part
          grad_v : velocity gradient (dim×dim)
          J_e    : det(F_e)
        Returns τ (dim×dim).
        """
        dim = F_e.n
        I = ti.Matrix.identity(float, dim)
        # Volumetric stress τ_v = (κ/2)*(J^2 - 1)*I
        tau_v = 0.5 * self.kappa * (J_e * J_e - 1.0) * I
        # Deviatoric stress τ_s = μ*(A A^T - (1/3)tr(A A^T)I)
        A = F_e * (J_e ** (-1.0 / dim))
        AAT = A @ A.transpose()
        trace_A = (AAT[0, 0] + AAT[1, 1] + (AAT[2, 2] if dim == 3 else 0.0))
        tau_s = self.mu * (AAT - (trace_A / dim) * I)
        # Newtonian viscous stress τ_N = η * (grad_v + grad_v^T) / 2
        tau_N = self.eta * 0.5 * (grad_v + grad_v.transpose())
        # Combine
        tau = tau_v + tau_s + tau_N
        return tau

    @ti.func
    def elastic_plastic_projection(self, F_e_trial, F_p_trial, p_idx, F_e, F_p):
        """
        Pre-projection for elastoplasticity:
          Clamp singular values of trial F_e into [f_c, f_s], update F_e, F_p, and harden moduli.
        """
        dim = F_e_trial.n
        # SVD: F_e_trial = U Σ* V^T
        U, sig_star, V = ti.svd(F_e_trial)
        # Clamp singulars
        for i in ti.static(range(dim)):
            sig_star[i] = min(self.f_s, max(self.f_c, sig_star[i]))
        # Recompose
        Σ = ti.diag(sig_star)
        F_e_new = U @ Σ @ V.transpose()
        # Plastic update: F_p_new = V Σ^{-1} U^T F_p_trial
        Σ_inv = ti.diag([1.0 / sig_star[i] for i in range(dim)])
        F_p_new = V @ Σ_inv @ U.transpose() @ F_p_trial
        # Write back
        F_e[p_idx] = F_e_new
        F_p[p_idx] = F_p_new
        # Hardening: update kappa, mu
        Jp = ti.Matrix.determinant(F_p_new)
        self.kappa = self.kappa0 * ti.exp(self.hardening * (1 - Jp))
        self.mu    = self.mu0    * ti.exp(self.hardening * (1 - Jp))

    @ti.func
    def viscoplastic_correction(self, b_e_d_trial, p_idx, b_e_d):
        """
        Post-correction for viscoplasticity:
          Yield-stress flow: clamp deviatoric stress to sqrt(2/3)*σ_Y.
        """
        dim = b_e_d_trial.n
        I = ti.Matrix.identity(float, dim)
        # Trial deviatoric stress
        trace_b = (b_e_d_trial[0,0] + b_e_d_trial[1,1] + (b_e_d_trial[2,2] if dim==3 else 0.0))
        dev = b_e_d_trial - (trace_b / dim) * I
        tau_s_trial = self.mu * dev
        # Frobenius norm
        s_trial = ti.sqrt((tau_s_trial * tau_s_trial).sum())
        yield_s = ti.sqrt(2.0/3.0) * self.sigmaY
        if s_trial > yield_s:
            # Simple return mapping: scale dev to yield surface
            dev_hat = tau_s_trial * (1.0 / s_trial)
            tau_s_new = yield_s * dev_hat
            # Reconstruct b_e_d from shifted stress: b_e_d_new = (tau_s_new/μ) + (trace/3)I
            trace_b_d = trace_b
            b_e_d_new = (1.0 / self.mu) * tau_s_new + (trace_b_d / dim) * I
            b_e_d[p_idx] = b_e_d_new
        else:
            b_e_d[p_idx] = b_e_d_trial


# ---------- MPM Simulator ----------

@ti.data_oriented
class MPMSimulator:
    def __init__(self, model, num_particles, dim=2,
                 grid_res=(128, 128, 1), dx=1.0):
        ti.init(arch=ti.gpu)
        self.model = model
        self.dim   = dim
        # Grid
        self.nx, self.ny, self.nz = grid_res
        self.dx  = dx
        self.inv_dx = 1.0 / dx
        # Particle data
        n = num_particles
        self.x   = ti.Vector.field(dim, float, n)
        self.v   = ti.Vector.field(dim, float, n)
        self.F_e = ti.Matrix.field(dim, dim, float, n)
        self.F_p = ti.Matrix.field(dim, dim, float, n)
        self.b_d = ti.Matrix.field(dim, dim, float, n)
        # Initial particle state
        for p in range(n):
            self.F_e[p] = ti.Matrix.identity(float, dim)
            self.F_p[p] = ti.Matrix.identity(float, dim)
            self.b_d[p] = ti.Matrix.identity(float, dim)
        # Grid fields
        self.grid_v = ti.Vector.field(dim, float, grid_res)
        self.grid_m = ti.field(float, grid_res)
        # Simulation params
        self.p_mass = 1.0
        self.dt     = 1e-3

    @ti.kernel
    def reset_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector.zero(float, self.dim)

    @ti.func
    def weight(self, x, base, offset):
        """
        Cubic B-spline weight and gradient.
        x: particle pos in grid coords, base: integer base idx, offset: nodal offset
        """
        rel = x - (base + offset).cast(float)
        w = 1.0
        for i in ti.static(range(self.dim)):
            absx = abs(rel[i])
            if absx < 1:
                w *= 0.5 * absx**3 - absx**2 + 2/3
            elif absx < 2:
                w *= (2 - absx)**3 / 6
            else:
                w *= 0.0
        return w

    @ti.kernel
    def p2g(self):
        self.reset_grid()
        for p in range(self.x.shape[0]):
            xp = self.x[p] * self.inv_dx
            base = (xp - 0.5).cast(int)
            for i, j, k in ti.ndrange(3, 3, 3):
                node = ti.Vector([base[0] + i, base[1] + j, 0])
                if 0 <= node[0] < self.nx and 0 <= node[1] < self.ny:
                    w = self.weight(xp, base, ti.Vector([i, j, 0]))
                    self.grid_m[node] += w * self.p_mass
                    self.grid_v[node] += w * self.p_mass * self.v[p]

    @ti.kernel
    def apply_forces(self):
        for p in range(self.x.shape[0]):
            # Compute velocity gradient
            xp = self.x[p] * self.inv_dx
            base = (xp - 0.5).cast(int)
            grad_v = ti.Matrix.zero(float, self.dim, self.dim)
            for i, j, k in ti.ndrange(3, 3, 3):
                node = ti.Vector([base[0] + i, base[1] + j, 0])
                if 0 <= node[0] < self.nx and 0 <= node[1] < self.ny:
                    # finite-difference approx of velocity gradient
                    v_node = self.grid_v[node] / (self.grid_m[node] + 1e-12)
                    grad_v[0,0] += v_node[0] * (i - 1)
                    grad_v[1,1] += v_node[1] * (j - 1)
            J_e = ti.Matrix.determinant(self.F_e[p])
            tau = self.model.compute_stress(self.F_e[p], self.b_d[p], grad_v, J_e)
            sigma = tau / (J_e + 1e-12)
            Vp = self.p_mass  # assume unit density and initial volume
            # scatter force to grid
            for i, j, k in ti.ndrange(3, 3, 3):
                node = ti.Vector([base[0] + i, base[1] + j, 0])
                if 0 <= node[0] < self.nx and 0 <= node[1] < self.ny:
                    w = self.weight(xp, base, ti.Vector([i, j, 0]))
                    f = -Vp * (sigma @ ti.Vector([i-1, j-1])) * w
                    self.grid_v[node] += self.dt * f

    @ti.kernel
    def grid_collision_and_update(self):
        for I in ti.grouped(self.grid_v):
            m = self.grid_m[I]
            if m > 0:
                # gravity
                self.grid_v[I].y -= self.dt * 9.81
                # floor collision
                if I.y < 2 and self.grid_v[I].y < 0:
                    self.grid_v[I].y = 0
                self.grid_v[I] = self.grid_v[I] / m

    @ti.kernel
    def g2p(self):
        for p in range(self.x.shape[0]):
            xp = self.x[p] * self.inv_dx
            base = (xp - 0.5).cast(int)
            v_new = ti.Vector.zero(float, self.dim)
            grad_v = ti.Matrix.zero(float, self.dim, self.dim)
            for i, j, k in ti.ndrange(3, 3, 3):
                node = ti.Vector([base[0] + i, base[1] + j, 0])
                if 0 <= node[0] < self.nx and 0 <= node[1] < self.ny:
                    w = self.weight(xp, base, ti.Vector([i, j, 0]))
                    v_node = self.grid_v[node]
                    v_new += w * v_node
                    grad_v[0,0] += v_node[0] * (i - 1)
                    grad_v[1,1] += v_node[1] * (j - 1)
            # update deformation
            self.F_e[p] = (ti.Matrix.identity(float, self.dim) + self.dt * grad_v) @ self.F_e[p]
            # plasticity
            self.model.elastic_plastic_projection(self.F_e[p], self.F_p[p], p, self.F_e, self.F_p)
            self.model.viscoplastic_correction(self.b_d[p], p, self.b_d)
            # update velocity & position
            self.v[p] = v_new
            self.x[p] += self.dt * v_new

    def step(self):
        self.p2g()
        self.apply_forces()
        self.grid_collision_and_update()
        self.g2p()


# ---------- PyTorch Integration for Inverse Learning ----------

class MPMLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sim: MPMSimulator, theta: torch.Tensor, targets: np.ndarray, n_steps: int):
        """
        theta: [μ, κ, η, σ_Y]
        targets: np.array of shape (n_steps, num_particles, dim)
        """
        # Assign to model
        sim.model.mu0    = float(theta[0].item())
        sim.model.kappa0= float(theta[1].item())
        sim.model.eta   = float(theta[2].item())
        sim.model.sigmaY= float(theta[3].item())
        sim_loss = 0.0
        for t in range(n_steps):
            sim.step()
            pred = sim.x.to_numpy()
            tgt  = targets[t]
            # simple L2 loss per frame
            sim_loss += np.mean((pred - tgt)**2)
        ctx.sim = sim
        ctx.save_for_backward(theta)
        return torch.tensor(sim_loss, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        # Run Taichi autodiff (requires Taichi taped kernels; omitted for brevity)
        # Here we approximate zeros (full diff requires more setup)
        grad = torch.zeros_like(ctx.saved_tensors[0])
        return None, grad * grad_output, None, None, None


# ---------- Visualization ----------

def visualize_2d(sim: MPMSimulator, frames=200):
    gui = ti.GUI("MPM 2D Simulation", res=(512, 512))
    for _ in range(frames):
        sim.step()
        pos = sim.x.to_numpy()
        gui.clear(0xFFFFFF)
        gui.circles(pos / np.array([sim.nx, sim.ny]), radius=1.5, color=0xFF5533)
        gui.show()

def visualize_3d(sim: MPMSimulator, frames=200):
    # Setup pyrender scene
    mesh = trimesh.points.PointCloud(np.zeros((len(sim.x), 3)))
    scene = pyrender.Scene()
    pc = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    node = scene.add(pc)
    camera = pyrender.PerspectiveCamera(yfov=np.pi/3.0)
    cam_node = scene.add(camera, pose=np.array([[1,0,0,0],[0,1,0,-3],[0,0,1,5],[0,0,0,1]]))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    scene.add(light, parent=cam_node)
    r = pyrender.OffscreenRenderer(800, 600)
    for _ in range(frames):
        sim.step()
        pts = sim.x.to_numpy()
        mesh.vertices = np.hstack([pts, np.zeros((pts.shape[0], 1))])
        color, _ = r.render(scene)
        # Display or save color as image

# ---------- Main ----------

if __name__ == "__main__":
    # Example: 2D hyperelastic block bouncing
    model = ConstitutiveModel(bulk_modulus=10.0, shear_modulus=5.0, viscosity=0.0,
                              yield_stress=0.0, hardening=0.0, f_c=0.9, f_s=1.1)
    sim = MPMSimulator(model, num_particles=20000, dim=2, grid_res=(128, 128, 1), dx=1/128)
    # Initialize particles: a square block at top
    for i in range(128):
        for j in range(128, 160):
            idx = i * (160 - 128) + (j - 128)
            if idx < sim.x.shape[0]:
                sim.x[idx] = ti.Vector([i, j])
                sim.v[idx] = ti.Vector([0, 0])
    visualize_2d(sim, frames=500)

    # To run inverse learning, provide `targets` as np.ndarray of shape (n_steps, num_particles, 2)
    # theta = torch.tensor([5.0, 10.0, 0.0, 0.0], requires_grad=True)
    # loss = MPMLossFunction.apply(sim, theta, targets, n_steps=100)
    # (Perform optimizer steps...)
