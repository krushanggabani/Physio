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
                #     self.gui.line(begin=A, end=B, radius=3, color=0x00FF00)
                    draw_link_with_thickness(self.gui, A, B, 0.05, color=0x00FF00)
        

        self.gui.text(f"Time: {t:.3f}s", (0.05, 0.95), font_size=24, color=0xFFFFFF)
        self.gui.show()
        




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
