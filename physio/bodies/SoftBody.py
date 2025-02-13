# super().__init__(name="Phidget Sensor")

# bodies/softbody.py
import numpy as np

class SoftBody:
    def __init__(self, nx, ny, width, height, mass_node, spring_params, floor_y,g,k_floor):
        """
        Initializes a soft body grid.
        spring_params: dict with keys 'h_k', 'v_k', 'd_k', 'd_struct'
        """
        self.type = 'soft'
        self.nx = nx
        self.ny = ny
        self.width = width
        self.height = height
        self.mass_node = mass_node
        self.floor_y = floor_y
        self.floor_x = [0.1, 0.9]

        self.g = g 
        self.k_floor = k_floor

        # Create grid nodes
        self.positions = []
        self.velocities = []
        x0 = 0.5 - width / 2
        y0 = floor_y
        dx = width / (nx - 1)
        dy = height / (ny - 1)
        for j in range(ny):
            for i in range(nx):
                self.positions.append(np.array([x0 + i * dx, y0 + j * dy], dtype=np.float64))
                self.velocities.append(np.array([0, 0], dtype=np.float64))
        self.positions = np.array(self.positions)
        self.velocities = np.array(self.velocities)
        self.fixed = np.zeros(len(self.positions), dtype=bool)
        
        N_nodes = nx * ny
        self.fixed = np.zeros(N_nodes, dtype=bool)
        for j in range(ny):
            left_index = j * nx + 0
            right_index = j * nx + (nx - 1)
            self.fixed[left_index] = True
            self.fixed[right_index] = True



        # Build spring connections: each spring is a dict with i, j, rest_length, k, damping.
        self.springs = []
        h_k = spring_params.get('h_k', 100)
        v_k = spring_params.get('v_k', 1000)
        d_k = spring_params.get('d_k', 1000)
        d_struct = spring_params.get('d_struct', 0.5)

        # Horizontal springs
        for j in range(ny):
            for i in range(nx - 1):
                idx1 = j * nx + i
                idx2 = idx1 + 1
                rest_length = np.linalg.norm(self.positions[idx2] - self.positions[idx1])
                self.springs.append({'i': idx1, 'j': idx2, 'rest_length': rest_length, 'k': h_k, 'damping': d_struct})
        # Vertical springs
        for j in range(ny - 1):
            for i in range(nx):
                idx1 = j * nx + i
                idx2 = idx1 + nx
                rest_length = np.linalg.norm(self.positions[idx2] - self.positions[idx1])
                self.springs.append({'i': idx1, 'j': idx2, 'rest_length': rest_length, 'k': v_k, 'damping': d_struct})
        # Diagonal springs (down-right)
        for j in range(ny - 1):
            for i in range(nx - 1):
                idx1 = j * nx + i
                idx2 = idx1 + nx + 1
                rest_length = np.linalg.norm(self.positions[idx2] - self.positions[idx1])
                self.springs.append({'i': idx1, 'j': idx2, 'rest_length': rest_length, 'k': d_k, 'damping': d_struct})
        # Diagonal springs (down-left)
        for j in range(ny - 1):
            for i in range(1, nx):
                idx1 = j * nx + i
                idx2 = idx1 + nx - 1
                rest_length = np.linalg.norm(self.positions[idx2] - self.positions[idx1])
                self.springs.append({'i': idx1, 'j': idx2, 'rest_length': rest_length, 'k': d_k, 'damping': d_struct})

    def apply_forces(self, dt, g, k_floor):
        """
        Compute forces on nodes due to gravity, springs, and floor collisions.
        """
        forces = np.zeros_like(self.positions)
        # Gravity
        forces[:, 1] -= self.mass_node * g

        # Spring forces
        for spring in self.springs:
            i = spring['i']
            j = spring['j']
            p1 = self.positions[i]
            p2 = self.positions[j]
            v1 = self.velocities[i]
            v2 = self.velocities[j]
            delta = p2 - p1
            dist = np.linalg.norm(delta)
            if dist == 0:
                continue
            direction = delta / dist
            springF = spring['k'] * (dist - spring['rest_length'])
            relVel = np.dot(v2 - v1, direction)
            dampF = spring['damping'] * relVel
            F = (springF + dampF) * direction
            forces[i] += F
            forces[j] -= F

        # Floor collision force
        for i in range(len(self.positions)):
            if self.positions[i, 1] < self.floor_y:
                penetration = self.floor_y - self.positions[i, 1]
                forces[i, 1] += k_floor * penetration

        for i in range(len(self.positions)):
            if self.positions[i, 0] < self.floor_x[0]:
                penetration = self.floor_x[0] - self.positions[i, 0]
                forces[i, 0] += k_floor * penetration
    
        for i in range(len(self.positions)):
            if self.positions[i, 0] > self.floor_x[1]:
                penetration = self.floor_x[1]- self.positions[i, 0]
                forces[i, 0] += k_floor * penetration


        return forces


    def update(self, dt):
        g  = self.g 
        k_floor = self.k_floor
        forces = self.apply_forces(dt, g, k_floor)
        acceleration = forces / self.mass_node
        self.velocities += acceleration * dt
        self.positions += self.velocities * dt

        self.velocities[self.fixed] = 0
