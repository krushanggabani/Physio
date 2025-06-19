import torch
import numpy as np

# Import your simulation wrapper: this should expose a function
# run_simulation(E, nu, cycles) -> (deformations: np.ndarray, forces: np.ndarray)
# You may need to refactor your Taichi/MPM script into a module for this.
from simulation_wrapper import run_simulation

# --- Load real-world data  ---
# Replace these paths or arrays with your actual data source
deformation_data = np.loadtxt('deformation_data.csv', delimiter=',')  # shape (8,)
force_data       = np.loadtxt('force_data.csv', delimiter=',')        # shape (8,)


deformation_data = torch.tensor(deformation_data, dtype=torch.float32)
force_data       = torch.tensor(force_data, dtype=torch.float32)

# --- Learnable parameters (optimize in log-space for positivity) ---
log_E  = torch.nn.Parameter(torch.log(torch.tensor(5e4)))    # initial E = 5e4
log_nu = torch.nn.Parameter(torch.log(torch.tensor(0.2)))    # initial nu = 0.2

optimizer = torch.optim.Adam([log_E, log_nu], lr=1e-2)

def loss_fn(sim_def, sim_force):
    """
    Mean squared error between simulated and real data.
    sim_def, sim_force: torch.Tensor shape (8,)
    deformation_data, force_data: torch.Tensor shape (8,)
    """
    loss_d = torch.sum((sim_def - deformation_data)**2)
    loss_f = torch.sum((sim_force - force_data)**2)
    return loss_d + loss_f


def optimize(num_iters=1000, eps=1e-3):
    for it in range(num_iters):
        # Current parameter values
        E_val  = torch.exp(log_E).item()
        nu_val = torch.exp(log_nu).item()
        rho_val = 1.0
        mu_floor_val = 0.4
        # Run the physics simulation for 8 cycles
        sim_def_np, sim_force_np = run_simulation(E_val, nu_val, rho_val, mu_floor_val,cycles=1)
        sim_def   = torch.tensor(sim_def_np, dtype=torch.float32)
        sim_force = torch.tensor(sim_force_np, dtype=torch.float32)

        # Compute loss
        loss = loss_fn(sim_def, sim_force)

        # Finite-difference gradient estimate
        grads = {}
        for param, name in [(log_E, 'E'), (log_nu, 'nu')]:
            orig = param.item()
            # +eps
            param.data.fill_(orig + eps)
            def_p, frc_p = run_simulation(torch.exp(log_E).item(), torch.exp(log_nu).item(), rho_val, mu_floor_val,cycles=1)
            loss_p = loss_fn(torch.tensor(def_p), torch.tensor(frc_p))
            # -eps
            param.data.fill_(orig - eps)
            def_m, frc_m = run_simulation(torch.exp(log_E).item(), torch.exp(log_nu).item(),rho_val, mu_floor_val, cycles=1)
            loss_m = loss_fn(torch.tensor(def_m), torch.tensor(frc_m))
            # restore
            param.data.fill_(orig)
            grads[name] = (loss_p - loss_m) / (2 * eps)

        # Assign gradients and step
        log_E.grad  = torch.tensor(grads['E'])
        log_nu.grad = torch.tensor(grads['nu'])
        optimizer.step()
        optimizer.zero_grad()

        if it % 10 == 0:
            print(f"Iter {it:3d}: E={E_val:.2e}, nu={nu_val:.3f}, loss={loss.item():.6f}")

    print("Optimization finished.")
    print(f"Optimized E: {torch.exp(log_E).item():.2e}")
    print(f"Optimized nu: {torch.exp(log_nu).item():.3f}")


if __name__ == '__main__':
    optimize(num_iters=200)  # adjust iterations as needed
