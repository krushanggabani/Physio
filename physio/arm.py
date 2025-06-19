import numpy as np
import taichi as ti

@ti.data_oriented
class PassiveArm:
    def __init__(self,
                 L1=0.12,L2=0.10,
                 k = 15, b=0.5,
                 y0=0.4, A=0.1,w=0.5):
        # Link properties
        self.L1, self.L2 = L1, L2
        self.k1, self.k2 = k,k
        self.b1, self.b2 = b, b
        self.I1 = L1**2 / 12.0
        self.I2 = L2**2 / 12.0

        # State
        self.theta1 = np.array([np.pi/20], dtype=np.float32)
        self.theta2 = np.array([0.0],     dtype=np.float32)
        self.dtheta1 = np.zeros(1, dtype=np.float32)
        self.dtheta2 = np.zeros(1, dtype=np.float32)
        self.theta1_rest = self.theta1.copy()
        self.theta2_rest = self.theta2.copy()
        # Base motion
        self.base_x = 0.5
        self.y0, self.A, self.w = y0, A, w
        self.base_y = self.y0
        self.time_t = 0.0

        self.base, self.j2, self.ee = self.forward_kinematics()


    def forward_kinematics(self):

        base = ti.Vector([self.base_x, self.base_y])
        j2   = base + ti.Vector([ti.sin(self.theta1[0]), -ti.cos(self.theta1[0])]) * self.L1
        ee   = j2   + ti.Vector([ti.sin(self.theta1[0] + self.theta2[0]),
                                -ti.cos(self.theta1[0] + self.theta2[0])]) * self.L2
    

        return base, j2, ee

    def update(self, roller_center, roller_velocity, contact_force, dt):
        # Update base vertical motion
        self.time_t += dt * 15
        self.base_y = self.y0 + self.A * np.cos(self.w * self.time_t)
        # Compute FK and velocities
        base, j2, ee_old = self.forward_kinematics()
        ee_new = self.forward_kinematics()[2]
        rv = (ee_new - ee_old) / dt
        # Update roller state
        roller_center[0] = ee_new
        roller_velocity[0] = rv
        # Compute torques from contact
        Fc = contact_force[0]
        r1 = ee_new - base
        r2 = ee_new - j2
        tau1_c = r1[1] * Fc[0] - r1[0] * Fc[1]
        tau2_c = r2[1] * Fc[0] - r2[0] * Fc[1]
        # Spring-damper torques
        tau1 = tau1_c - self.k1 * (self.theta1[0] - self.theta1_rest[0]) - self.b1 * self.dtheta1[0]
        tau2 = tau2_c - self.k2 * (self.theta2[0] - self.theta2_rest[0]) - self.b2 * self.dtheta2[0]
        # Integrate dynamics (semi-implicit Euler)
        alpha1 = tau1 / self.I1
        alpha2 = tau2 / self.I2
        self.dtheta1[0] += alpha1 * dt
        self.theta1[0] += self.dtheta1[0] * dt
        self.dtheta2[0] += alpha2 * dt
        self.theta2[0] += self.dtheta2[0] * dt
