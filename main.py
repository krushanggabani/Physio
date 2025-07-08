import time


from physio.config.base_config import Config
from physio.physioenv import PhysioEnv

cfg = Config()



env = PhysioEnv(cfg)


for _ in range(1000):

    env.step(n_substeps=25)
    env.render()
