# sim.py
import taichi as ti
import numpy as np
import time

from physio.visulization.Visualizer import *
from physio.visulization.recorder import *

class SimulationEnv:

    def __init__(self):

        self.env_dt         = 1e-3                 # Time step for the simulation
        self.time           = 0.0                  # Current simulation time
        self.gravity        = [0,-9.81]            # Gravity Vector
        self.bodies         = []                   # List to store all bodies in the simulation
        self.vis            = Visualizer()         #  initilize visulizing screen
        self.recorder       = Recorder(output_file='simulation2.gif', fps=30)
        self.gui_flag       = True
        self.record_flag    = True
        

    def add_body(self,body):

        self.bodies.append(body)


    def update(self):
        """Update all bodies in the simulation and advance time."""
        for body in self.bodies:
            body.update(self.dt)

        self.time += self.dt

        self.vis.render(self)

        self.gui_flag = self.vis.gui.running
        
        frame = self.vis.gui.get_image()
        frame = (np.array(frame) * 255).astype(np.uint8)  # Convert float32 to uint8
        frame = np.rot90(frame, k=1)  # Rotate 90 degrees counterclockwise
        self.recorder.add_frame(frame)

        # imageio.imwrite('frame.png', frame)

        self.vis.gui.show()
        