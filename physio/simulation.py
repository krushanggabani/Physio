import taichi as ti
import numpy as np
from .mpm import MPMSimulator
from .arm import PassiveArm


ti.init(arch=ti.gpu, debug=False, fast_math=True, device_memory_GB=9)
@ti.data_oriented
class Simulation:
    def __init__(self, res=(800, 800)):
        self.arm = PassiveArm()
        self.sim = MPMSimulator(arm=self.arm)
        self.gui = ti.GUI('MPM + Single Passive Arm', res=res)
        self.res = res

    def run(self):
        while self.gui.running:
            # update simulation
            for _ in range(15):
                self.sim.p2g()
                self.sim.apply_grid_forces_and_detect()
                self.sim.g2p()
                # pull data for arm update
                rc = self.sim.roller_center.to_numpy()
                rv = self.sim.roller_velocity.to_numpy()
                cf = self.sim.contact_force.to_numpy()
                self.arm.update(rc, rv, cf, self.sim.dt)

            # render soft particles
            self.gui.circles(self.sim.x.to_numpy(), radius=1.5, color=0x66CCFF)
            # draw arm and roller
            base = np.array([self.arm.base_x, self.arm.base_y], dtype=np.float32)
            j2 = base + np.array([np.sin(self.arm.theta1[0]), -np.cos(self.arm.theta1[0])]) * self.arm.L1
            ee = self.sim.roller_center.to_numpy()[0]
            # links
            self.gui.line(begin=base, end=j2, radius=2, color=0x000050)
            self.gui.line(begin=j2,   end=ee, radius=2, color=0x000050)
            # joints
            self.gui.circle(base, radius=4, color=0xFF0000)
            self.gui.circle(j2,   radius=4, color=0xFF0000)
            # roller
            rr = int(self.sim.roller_radius * self.res[0])
            self.gui.circle(ee, radius=rr, color=0xFF0000)
            # info
            self.gui.text(f'Time: {self.arm.time_t:.3f}   Force: {self.sim.contact_force[0]} N',
                          pos=(0.02, 0.95), color=0xFFFFFF)
            self.gui.show()