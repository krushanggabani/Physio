# bodies/robotic_arm.py
from physio.bodies.Link import *
import numpy as np

class RoboticArm:
    def __init__(self, base, link_params):
        """
        Initialize the arm with a base position and a list of link parameters.
        Each link_params element is a dict with keys: mass, length, angle, k_rot, theta_eq.
        """
        self.type = "robot_arm"
        self.links = []
        self.damping = 0.25
        self.external_torques = [0 ,0]
        origin = np.array(base, dtype=np.float64)
        for params in link_params:
            link = Link(params['mass'], params['length'], origin, params['angle'],
                        angular_velocity=0, k_rot=params.get('k_rot', 0),
                        theta_eq=params.get('theta_eq', 0))
            self.links.append(link)
            origin = link.get_end()


    def set_external_torques(self, external_torques):
        """
        Store external torques to be used in the update.
        external_torques: List or array of torques for each link.
        """
        self.external_torques = external_torques
        
    def update(self, dt):
        external_torques = self.external_torques
        damping = self.damping
        """Update each link with the corresponding external torque."""
        for i, link in enumerate(self.links):
            ext_tau = external_torques[i] if i < len(external_torques) else 0
            link.update(dt, ext_tau, damping)
            if i < len(self.links) - 1:
                self.links[i+1].set_origin(link.get_end())

    def set_base(self, new_base):
        """Update the base position of the arm and propagate changes along the links."""
        self.links[0].set_origin(new_base)
        for i in range(1, len(self.links)):
            self.links[i].set_origin(self.links[i-1].get_end())

    def get_link_segments(self):
        """Return a list of segments (as tuples of (start, end)) for each link."""
        segments = []
        for link in self.links:
            segments.append((link.origin, link.get_end()))
        return segments



    def __repr__(self):
        return "\n".join(str(link) for link in self.links)
