"""
Defines the RobotArm class with configurable number of links.
Provides forward kinematics to compute end effector pose.
"""
import numpy as np

class RobotArm:
    def __init__(self, link_lengths):
        """
        link_lengths: list of floats, one per link
        joint_angles: initialized to zeros
        """
        self.links = np.array(link_lengths, dtype=np.float32)
        self.n_links = len(self.links)
        self.joint_angles = np.zeros(self.n_links, dtype=np.float32)

    def forward_kinematics(self):
        """
        Computes the 3D end effector position assuming planar rotation about Z-axis.
        Returns a numpy array of length 3.
        """
        pos = np.zeros(3, dtype=np.float32)
        angle_accum = 0.0
        for i in range(self.n_links):
            angle_accum += self.joint_angles[i]
            dx = self.links[i] * np.cos(angle_accum)
            dy = self.links[i] * np.sin(angle_accum)
            pos[:2] += [dx, dy]
        return pos