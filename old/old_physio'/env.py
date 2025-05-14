import numpy as np
import taichi as ti
import time

from physio.mpm import MPM
from physio.link import Link
from physio.render import Render

ti.init(arch=ti.gpu, debug=False, fast_math=True, device_memory_GB=9)

@ti.data_oriented
class Env:
    def __init__(self,config):
        
        self.cfg    = config
        self.env_dt = config.dt

        self.link   = Link(config)
        self.sim    = MPM (config)

        self.render = Render(config)

    def step(self):
        

        # for _ in range(int(5e-3 // self.dt)):
        #     self.sim.substep()

        x = self.sim.x
        points_np = x.to_numpy()
        colors = np.ones((self.sim.n_particles, 4), dtype=np.float32)
        colors[:, :3] = (points_np + 1.0) * 0.5


        self.render.view(points_np , colors)


        




