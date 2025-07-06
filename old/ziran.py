import taichi as ti
import numpy as np
import pyrender
import trimesh
import time
from OpenGL.GL import *


np.infty = np.inf


# Initialize Taichi
ti.init(arch=ti.gpu)

# Simulation parameters
dim = 3
n_grid = 64
dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 1e-4
p_vol = (dx * 0.5) ** 3
p_rho = 1000.0
p_mass = p_vol * p_rho

# Material properties
E = 5e3  # Young's modulus
nu = 0.3  # Poisson's ratio
mu_0 = E / (2 * (1 + nu))
lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

# Particle counts
n_particles_soft = 6000
n_particles_rigid = 1000
n_particles = n_particles_soft + n_particles_rigid

# Material identifiers
SOFT = 0
RIGID = 1

# Taichi fields
x = ti.Vector.field(dim, dtype=float, shape=n_particles)
v = ti.Vector.field(dim, dtype=float, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
Jp = ti.field(dtype=float, shape=n_particles)
material = ti.field(dtype=int, shape=n_particles)

# Grid fields
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))

@ti.func
def clamp(a, low, high):
    return max(low, min(a, high))

@ti.kernel
def clear_grid():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0.0, 0.0, 0.0]
        grid_m[i, j, k] = 0.0

@ti.kernel
def p2g():
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        
        # Quadratic kernels
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        
        # Deformation gradient update
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]
        
        # Compute stress
        stress = ti.Matrix.zero(float, 3, 3)
        
        if material[p] == SOFT:
            # Neo-Hookean model
            J = F[p].determinant()
            if J > 0.1:
                F_inv_T = F[p].inverse().transpose()
                stress = mu_0 * (F[p] @ F[p].transpose() - ti.Matrix.identity(float, 3)) + \
                        lambda_0 * ti.log(J) * F_inv_T
            Jp[p] = J
        else:
            # Rigid body - no deformation
            F[p] = ti.Matrix.identity(float, 3)
            Jp[p] = 1.0
        
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            
            grid_pos = base + offset
            if 0 <= grid_pos[0] < n_grid and 0 <= grid_pos[1] < n_grid and 0 <= grid_pos[2] < n_grid:
                grid_v[grid_pos] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[grid_pos] += weight * p_mass

@ti.kernel
def grid_op():
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = grid_v[i, j, k] / grid_m[i, j, k]
            grid_v[i, j, k] += dt * ti.Vector([0.0, -9.8, 0.0])  # gravity
            
            # Boundary conditions
            boundary = 3
            if i < boundary and grid_v[i, j, k][0] < 0:
                grid_v[i, j, k][0] = 0
            if i > n_grid - boundary and grid_v[i, j, k][0] > 0:
                grid_v[i, j, k][0] = 0
                
            if j < boundary and grid_v[i, j, k][1] < 0:
                grid_v[i, j, k][1] = 0
            if j > n_grid - boundary and grid_v[i, j, k][1] > 0:
                grid_v[i, j, k][1] = 0
                
            if k < boundary and grid_v[i, j, k][2] < 0:
                grid_v[i, j, k][2] = 0
            if k > n_grid - boundary and grid_v[i, j, k][2] > 0:
                grid_v[i, j, k][2] = 0

@ti.kernel
def g2p():
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset - fx) * dx
            grid_pos = base + offset
            
            if 0 <= grid_pos[0] < n_grid and 0 <= grid_pos[1] < n_grid and 0 <= grid_pos[2] < n_grid:
                weight = w[i][0] * w[j][1] * w[k][2]
                g_v = grid_v[grid_pos]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        
        v[p] = new_v
        x[p] += dt * v[p]
        C[p] = new_C

@ti.kernel
def init_particles():
    # Initialize soft sphere particles (resting on ground)
    center_soft = ti.Vector([0.4, 0.15, 0.5])
    radius_soft = 0.12
    
    for i in range(n_particles_soft):
        # Use spherical coordinates for better distribution
        theta = ti.random() * 2.0 * 3.14159
        phi = ti.acos(1.0 - 2.0 * ti.random())  # Uniform distribution on sphere
        r = radius_soft * (ti.random() ** (1.0/3.0))  # Uniform volume distribution
        
        # Only keep upper hemisphere
        if ti.cos(phi) >= -0.2:  # Slight extension below center
            pos = ti.Vector([
                r * ti.sin(phi) * ti.cos(theta),
                r * ti.cos(phi) * 0.7,  # Flatten slightly
                r * ti.sin(phi) * ti.sin(theta)
            ])
            
            x[i] = center_soft + pos
            v[i] = ti.Vector([0.0, 0.0, 0.0])
            F[i] = ti.Matrix.identity(float, 3)
            C[i] = ti.Matrix.zero(float, 3, 3)
            Jp[i] = 1.0
            material[i] = SOFT
    
    # Initialize rigid sphere particles (falling from above)
    center_rigid = ti.Vector([0.4, 0.7, 0.5])
    radius_rigid = 0.06
    
    for i in range(n_particles_rigid):
        theta = ti.random() * 2.0 * 3.14159
        phi = ti.acos(1.0 - 2.0 * ti.random())
        r = radius_rigid * (ti.random() ** (1.0/3.0))
        
        pos = ti.Vector([
            r * ti.sin(phi) * ti.cos(theta),
            r * ti.cos(phi),
            r * ti.sin(phi) * ti.sin(theta)
        ])
        
        idx = n_particles_soft + i
        x[idx] = center_rigid + pos
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        F[idx] = ti.Matrix.identity(float, 3)
        C[idx] = ti.Matrix.zero(float, 3, 3)
        Jp[idx] = 1.0
        material[idx] = RIGID

def substep():
    clear_grid()
    p2g()
    grid_op()
    g2p()

def create_particle_visualization(positions, color, radius=0.008):
    """Create particle visualization as point clouds"""
    if len(positions) == 0:
        return None
    
    # Subsample for performance
    step = max(1, len(positions) // 1000)
    sampled_positions = positions[::step]
    
    spheres = []
    for pos in sampled_positions:
        sphere = trimesh.creation.uv_sphere(radius=radius, count=[6, 6])
        sphere.apply_translation(pos)
        spheres.append(sphere)
    
    if spheres:
        combined_mesh = trimesh.util.concatenate(spheres)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.0,
            roughnessFactor=0.8
        )
        return pyrender.Mesh.from_trimesh(combined_mesh, material=material)
    return None

def create_floor():
    """Create floor plane"""
    floor = trimesh.creation.box(extents=[2.0, 0.02, 2.0])
    floor.apply_translation([0.5, -0.01, 0.5])
    
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.6, 0.6, 0.6, 1.0],
        metallicFactor=0.1,
        roughnessFactor=0.9
    )
    return pyrender.Mesh.from_trimesh(floor, material=material)

def get_particle_data():
    """Extract particle positions and materials"""
    pos_data = x.to_numpy()
    mat_data = material.to_numpy()
    
    soft_pos = pos_data[mat_data == SOFT]
    rigid_pos = pos_data[mat_data == RIGID]
    
    return soft_pos, rigid_pos

def main():
    print("Initializing MPM Physics Simulation...")
    print("Green spheres: Soft viscoelastic material")
    print("Red spheres: Rigid falling object")
    print("Press ESC or close window to exit")
    
    # Initialize simulation
    init_particles()
    
    # Setup rendering
    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])
    
    # Add floor
    floor_mesh = create_floor()
    scene.add(floor_mesh)
    
    # Add lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    light_pose = np.array([
        [0.707, -0.707, 0, 0],
        [0.5, 0.5, -0.707, 0],
        [0.5, 0.5, 0.707, 2],
        [0, 0, 0, 1]
    ])
    scene.add(light, pose=light_pose)
    
    # Setup camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    camera_pose = np.array([
        [0.866, -0.25, 0.433, 0.8],
        [0, 0.866, 0.5, 0.6],
        [-0.5, -0.433, 0.75, 1.0],
        [0, 0, 0, 1]
    ])
    scene.add(camera, pose=camera_pose)
    
    # Create viewer
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    
    # Simulation loop
    frame = 0
    soft_node = None
    rigid_node = None
    
    try:
        while viewer.is_active:
            # Run multiple substeps per frame for stability
            # for _ in range(8):
            #     substep()
            
            # Update visualization every few frames
            if frame % 2 == 0:
                soft_positions, rigid_positions = get_particle_data()
                
                # Remove old meshes
                if soft_node is not None:
                    scene.remove_node(soft_node)
                if rigid_node is not None:
                    scene.remove_node(rigid_node)
                
                # Create new particle meshes
                soft_mesh = create_particle_visualization(
                    soft_positions, 
                    [0.2, 0.8, 0.3, 1.0],  # Green
                    0.006
                )
                rigid_mesh = create_particle_visualization(
                    rigid_positions, 
                    [0.8, 0.2, 0.2, 1.0],  # Red
                    0.008
                )
                
                if soft_mesh is not None:
                    soft_node = scene.add(soft_mesh)
                if rigid_mesh is not None:
                    rigid_node = scene.add(rigid_mesh)
            
            frame += 1
            time.sleep(0.01)  # Target ~60 FPS
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        if viewer.is_active:
            viewer.close_external()
        print("Simulation ended")

if __name__ == "__main__":
    main()