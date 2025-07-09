# --------------------------------------------------------------------------------
# Copyright (c) 2025 Krushang Gabani
# All rights reserved.
#
# High-level environment wrapper for the Material Point Method (MPM) simulator
# with PyRender-based visualization.
#
# Author: Krushang Gabani
# Date: July 7, 2025
# --------------------------------------------------------------------------------

import time
from typing import Dict, Any, Tuple

import numpy as np
import taichi as ti

from physio.config.base_config import Config
from physio.simulators.base_mpm import BASE_MPM
from physio.visualization.PyRenderer import PyRenderer3D


ti.init(arch=ti.gpu, kernel_profiler=True, debug=False, fast_math=True, device_memory_GB=9)


@ti.data_oriented
class PhysioEnv:
  
    def __init__(self,cfg):
        
        self.cfg = cfg
        self.env_dt = cfg.dt

        self.simulator = BASE_MPM(cfg)

        self._t = 0.0
        self._step = 0

        if cfg.render_model == "PyRender":
            self.renderer = PyRenderer3D(cfg)


    
    def reset(self):
        self.simulator.reset()


    def step(self,n_substeps=1):
        self.simulator.step(n_substeps)
        # ti.profiler.print_scoped_profiler_info()
        # ti.profiler.print_kernel_profiler_info('trace')
        # ti.profiler.clear_kernel_profiler_info()

    def render(self):
        pts = self.simulator.x.to_numpy()
        robot = self.simulator.roller_center.to_numpy()
        self.renderer.set_particles(pts)
        self.renderer.update_robot(robot)
        self.renderer.render()