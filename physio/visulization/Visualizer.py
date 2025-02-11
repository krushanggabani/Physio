# visualization/Visualizer.py
import taichi as ti
import numpy as np

class Visualizer:
    def __init__(self, width=800, height=600):
        self.res = (width, height)
        self.gui = ti.GUI("Simulator", self.res)
        self.gui.clear(color=0x112F41)

    def render(self,sim):

        t = sim.time

        for body in sim.bodies:
            

            if body.type =="soft":
                positions = body.positions
                for spring in body.springs:
                    A = positions[spring['i']]
                    B = positions[spring['j']]
                    self.gui.line(begin=A, end=B, radius=1, color=0xFFFFFF)
                for pos in positions:
                    self.gui.circle(pos, radius=2, color=0xFF0000)


            if body.type == "robot_arm":

                 # Draw robotic arm links as thick lines
                for A, B in body.get_link_segments():
                    self.gui.line(begin=A, end=B, radius=3, color=0x00FF00)
        

        self.gui.text(f"Time: {t:.3f}s", (0.05, 0.95), font_size=24, color=0xFFFFFF)
        self.gui.show()
        

