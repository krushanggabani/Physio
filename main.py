import time
from yacs.config import CfgNode as CN
import taichi as ti
import numpy as np

from physio.env import Env

cfg             = CN()
cfg.dtype       = ti.f64 
cfg.dt          = 1e-3
cfg.gravity     = (0.0,0.0,-9.8)
cfg.dim         = 3
cfg.max_steps   = 2048

cfg.n_particles     = 50000
cfg.material_model  = 0              # 0 = "neo-hooken" and 1 = "sls"
cfg.E               = 3e3
cfg.nu              = 0.2

cfg.link         = CN()
cfg.link.size    = (0.02,0.02,0.1)
cfg.link.pos     = (0.5,0,0)




if __name__ == "__main__":
    
    env = Env(cfg)
    print(env)

    while True:
        env.step()
