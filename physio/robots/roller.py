# --------------------------------------------------------------------------------
# Copyright (c) 2025 Krushang Gabani
# All rights reserved.
#
# Roller: rigid spherical roller with dynamics and collision for MPM interaction.
# Implements forward kinematics (fk), rigid-body dynamics, and impulse-based
# collision response with particles.
#
# Author: Krushang Gabani
# Date: July 7, 2025
# --------------------------------------------------------------------------------

import taichi as ti
import numpy as np


@ti.data_oriented
class Roller:

    """
    Rigid spherical roller that can translate, rotate, and collide elastically
    with point particles. Uses impulse-based collision and simple Euler integration.
    """

    def __init__(self,cfg):

        self.cfg = cfg
        self.dim = dim = cfg.dim
        self.dtype = ti.f32 if cfg.dtype == 'float32' else ti.f64

       # physical properties
        self.radius = cfg.roller_radius      # roller radius
        self.mass = cfg.roller_mass          # roller mass
        # moment of inertia for a solid sphere: (2/5)m r^2
        self.inertia = 2.0 / 5.0 * self.mass * self.radius ** 2
        self.restitution = getattr(cfg, 'roller_restitution', 0.5)
        self.friction = getattr(cfg, 'roller_friction', 0.2)

        # state fields: rotation quaternion [w, x, y, z], angular velocity, position, linear velocity
        self.position = ti.Vector.field(self.dim, dtype=self.dtype, shape=())
        self.linear_velocity = ti.Vector.field(self.dim, dtype=self.dtype, shape=())
        self.rotation = ti.Vector.field(4, dtype=self.dtype, shape=())
        self.angular_velocity = ti.Vector.field(self.dim, dtype=self.dtype, shape=())

        # initialize state
        self._init_state()

    @ti.kernel
    def _init_state(self):
        # place roller at initial position
        pos = ti.Vector(list(self.cfg.roller_init_pos))
        self.position[None] = pos
        # zero velocities
        self.linear_velocity[None] = ti.Vector.zero(self.dtype, self.dim)
        # identity quaternion: (1,0,0,0)
        self.rotation[None] = ti.Vector([1.0, 0.0, 0.0, 0.0])
        self.angular_velocity[None] = ti.Vector.zero(self.dtype, self.dim)

    @staticmethod
    @ti.func
    def quat_mul(a, b):
        # quaternion multiplication a*b
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return ti.Vector([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    @ti.kernel
    def fk(self):
        # forward kinematics: update rotation quaternion based on angular velocity
        omega = self.angular_velocity[None]
        q = self.rotation[None]
        # quaternion derivative: 0.5 * omega_quat * q
        omega_quat = ti.Vector([0.0, omega[0], omega[1], omega[2]])
        dq = 0.5 * Roller.quat_mul(omega_quat, q)
        self.rotation[None] = q + dq * self.cfg.dt
        # normalize quaternion
        norm = ti.sqrt((self.rotation[None]**2).sum())
        self.rotation[None] = self.rotation[None] / norm

    @ti.kernel
    def dynamic(self):
        # integrate linear and angular velocity under gravity
        g = ti.Vector(list(self.cfg.gravity))
        # linear: v += g*dt, x += v*dt
        v = self.linear_velocity[None]
        v += g * self.cfg.dt
        self.linear_velocity[None] = v
        self.position[None] += v * self.cfg.dt
        # angular: no external torque for now
        # can add torque term: alpha = torque / inertia
        # self.angular_velocity[None] += alpha * dt

    @ti.func
    def collide_particle(self, p_pos, p_vel) -> ti.Vector:
        # impulse response of a single particle against the roller
        # returns new particle velocity
        # vector from center to particle
        rel = p_pos - self.position[None]
        dist = rel.norm()
        # if inside roller (penetration), generate impulse
        if dist < self.radius:
            # collision normal
            n = rel.normalized()
            # relative velocity at contact
            # linear + angular contribution: v_contact = v + w x r
            r_vec = n * self.radius
            w = self.angular_velocity[None]
            v_cont = self.linear_velocity[None] + w.cross(r_vec)
            rel_v = p_vel - v_cont
            vn = rel_v.dot(n)
            if vn < 0:
                # normal impulse magnitude
                jn = -(1 + self.restitution) * vn
                # apply normal impulse to particle only (roller is heavy)
                new_v = p_vel + jn * n
                # friction impulse
                vt = rel_v - vn * n
                jt = -vt
                jt_max = self.friction * jn
                if jt.norm() > jt_max:
                    jt = jt * (jt_max / jt.norm())
                new_v += jt
                return new_v
        return p_vel

    def step(self):
        # one timestep: move roller and optionally collide many particles
        self.dynamic()
        self.fk()


        
        


