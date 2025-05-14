import taichi as ti
import numpy as np


class Link:

    def __init__(self,config):
        
        self.dim = config.dim
        dtype = self.dtype = config.dtype


        self.friction = ti.field(dtype, shape=())                   # friction coeff
        self.softness = ti.field(dtype, shape=())                   # softness coeff for contact modeling
        self.position = ti.Vector.field(3, dtype, needs_grad=True)  # positon of the primitive
        self.rotation = ti.Vector.field(4, dtype, needs_grad=True)  # quaternion for storing rotation

        self.v = ti.Vector.field(3, dtype, needs_grad=True)         # velocity
        self.w = ti.Vector.field(3, dtype, needs_grad=True)         # angular velocity
