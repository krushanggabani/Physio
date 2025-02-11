# bodies/link.py
import numpy as np

class Link:
    def __init__(self, mass, length, origin, angle, angular_velocity=0, k_rot=0, theta_eq=0):
        self.mass = mass
        self.length = length
        self.origin = np.array(origin, dtype=np.float64)
        self.angle = angle
        self.angular_velocity = angular_velocity
        self.k_rot = k_rot
        self.theta_eq = theta_eq

    def get_end(self):
        """Compute the end position of the link from its origin and angle."""
        dx = self.length * np.sin(self.angle)
        dy = -self.length * np.cos(self.angle)
        return self.origin + np.array([dx, dy])

    def set_origin(self, new_origin):
        self.origin = np.array(new_origin, dtype=np.float64)
        return self

    def update(self, dt, external_torque, damping):
        """
        Update the angular state using a simple Euler integration.
        A rod’s moment of inertia about one end is approximated as I = m*l^2/3.
        """
        I = self.mass * self.length**2 / 3.0
        tau_rest = -self.k_rot * (self.angle - self.theta_eq)
        total_tau = tau_rest + external_torque - damping * self.angular_velocity
        angular_acc = total_tau / I
        self.angular_velocity += angular_acc * dt
        self.angle += self.angular_velocity * dt
        return self

    def __repr__(self):
        return f"Link(origin={self.origin}, angle={self.angle:.2f}, length={self.length})"
