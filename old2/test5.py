import time
from argparse import ArgumentParser

import taichi as ti
import numpy as np

from softmac.engine.taichi_env import TaichiEnv
from softmac.utils import  prepare



t = 0

def draw_link_with_thickness(gui, A, B, thickness, color):
    """
    Draw a thick link as a filled quadrilateral (using two triangles).
    
    Parameters:
      gui: Taichi GUI instance.
      A, B: Endpoints of the link (as numpy arrays).
      thickness: The half–thickness (offset) to apply.
      color: Color for the patch (e.g. 0x00FF00).
    """
    v = B - A
    norm_v = np.linalg.norm(v)


    if norm_v == 0:
        return
    # Compute the perpendicular unit vector.
    n = np.array([-v[1], v[0]]) / norm_v
    # Compute the four vertices of the quadrilateral.
    p1 = A + n * thickness
    p2 = B + n * thickness
    p3 = B - n * thickness
    p4 = A - n * thickness


    # Split the quadrilateral into two triangles:
    triangle1 = np.array([p1, p2, p3])
    triangle2 = np.array([p1, p3, p4])


    
    # Draw each triangle; alpha is set to 0.5 for transparency.
    gui.triangle(p1,p2,p3, color)
    gui.triangle(p1,p3,p4, color)




parser = ArgumentParser()

parser.add_argument("--config", type=str, default="softmac/config/demo_grip_config.py")

args = parser.parse_args()


ti.init(arch=ti.cpu)

# — window —
window_size = 800
gui = ti.GUI("Viscoelastic MPM", (window_size, window_size))



cfg = "softmac/config/demo_grip_config.py"

log_dir, cfg = prepare(args)
env = TaichiEnv(cfg)


env.simulator.primitives_contact = [False, True, True]
env.reset()


print("Starting Simulation")
while gui.running:
    # preparation
    tik = time.time()
    ti.ad.clear_all_gradients()
  
   



    # forward
    tik = time.time()

    # print(actions[i])
    for i in range(2):
        env.step()
    
    for idx in range(env.simulator.n_particles):
        env.simulator.y[idx] = env.simulator.x[i,idx]

    y2 = env.simulator.y.to_numpy()
    # print(y2.shape)
    # print(y2[:,:2])
    

    gui.circles(y2[1:1000,:2], radius=2, color=0xED553B)


    
    t = t + env.env_dt
    A = np.array([0.5,0.5-np.sin(5*t)])
    B = np.array([0.5,0.15- np.sin(5*t)])

    draw_link_with_thickness(gui, A, B, 0.01, color=0x00FF00)

    gui.show()














