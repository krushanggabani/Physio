import taichi as ti
import numpy as np


@ti.data_oriented
class ConstitutiveModel:
    def __init__(self, bulk_modulus, shear_modulus, viscosity,yield_stress=0.0, hardening=0.0,
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
        dim = 2
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