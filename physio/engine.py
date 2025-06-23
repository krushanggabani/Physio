import numpy as np
import taichi as ti
import torch
import time

from physio.mpm_simulator import MPMSim
from physio.renderer import pyrender



ti.init(arch=ti.gpu, debug=False, fast_math=True, device_memory_GB=9)
@ti.data_oriented
class Env:

    def __init__(self, args):


        print(args)
        self.args   = args
        self.env_dt = args.Time_step 
        self.dim    = args.Dimension
        self.sim    = MPMSim(args)

        self.gui = ti.GUI('MPM + Single Passive Arm (Attached & Outward)', res=(512, 512))

    def reset(self):
        self.sim.init_mpm()


    def step(self):
        for _ in range(15):
            self.sim.step()
            

    def render(self):

        """
        Render particles, arm links, roller, and contact info.
        """
        # Draw soft body particles
        particle_positions = self.sim.x.to_numpy()
        self.gui.circles(
            particle_positions,
            radius=1.5,
            color=0x66CCFF
        )

        # Compute arm keypoints
        theta1 = self.sim.theta1[0]
        theta2 = self.sim.theta2[0]
        L1, L2 = self.sim.L1, self.sim.L2
        base_pt = np.array([self.sim.base_x, self.sim.base_y], dtype=np.float32)
        joint2 = base_pt + np.array([
            np.sin(theta1),
            -np.cos(theta1)
        ]) * L1
        end_effector = self.sim.roller_center[0].to_numpy()

        # Draw arm links
        self.gui.line(begin=base_pt, end=joint2, radius=2, color=0x000050)
        self.gui.line(begin=joint2, end=end_effector, radius=2, color=0x000050)

        # Draw joints and roller
        self.gui.circle(base_pt, radius=4, color=0xFF0000)
        self.gui.circle(joint2, radius=4, color=0xFF0000)
        self.gui.circle(
            end_effector,
            radius=int(self.sim.roller_radius * self.gui.res[0]),
            color=0xFF0000
        )

        self.gui.show()

        

        # time_t = self.sim.time_t
        # Display time and contact force
        force = self.sim.contact_force_vec[0].to_numpy()
        # self.gui.text(
        #     f"Time: {self.time_t:.2f}s  Force: [{force[0]:.2f}, {force[1]:.2f}] N",
        #     pos=(0.02, 0.95),
        #     color=0xFFFFFF
        # )

        # Render to screen
        