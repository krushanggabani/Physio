# sim.py
import taichi as ti
import numpy as np

from physio.visulization.Visualizer import *


class SimulationEngine:

    def __init__(self):

        self.dt      = 0.001                # Time step for the simulation
        self.time    = 0.0                  # Current simulation time
        self.gravity = [0,-9.81]            # Gravity Vector
        self.bodies  = []                   # List to store all bodies in the simulation
        self.vis     = Visualizer()         #  initilize visulizing screen
        self.gui_flag = True


    def add_body(self,body):

        self.bodies.append(body)


    def update(self):
        """Update all bodies in the simulation and advance time."""
        for body in self.bodies:
            body.update(self.dt)

        self.time += self.dt

        self.vis.render(self)

        self.gui_flag = self.vis.gui.running
