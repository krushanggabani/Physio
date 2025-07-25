# this is test scenario with rigid rectangular body is free falling on softbody.
import taichi as ti
import numpy as np

from physio.SimulationEngine import *
from physio.bodies.SoftBody import *
from physio.bodies.Robot_Arm import *
from physio.collision.detect_collision import *
from physio.visulization.recorder import *


# Initialize Taichi (change arch if needed)
ti.init(arch=ti.cpu)


# record_setting = {'record_flag' : True,} 

engine = SimulationEngine()




nx, ny = 25, 5
width, height = 0.75, 0.25
mass_node = 0.01
spring_params = {'h_k': 100, 'v_k': 100, 'd_k': 1000, 'd_struct': 0.5}
soft_body = SoftBody(nx, ny, width, height, mass_node, spring_params,floor_y=0,g = 9.81, k_floor=10000)
engine.add_body(soft_body)



base = [0.7, 0.4]
link_params = [
    {'mass': 1, 'length': 0.1, 'angle': 0, 'k_rot': 2, 'theta_eq': 0*-np.pi/6},
    {'mass': 1, 'length': 0.15, 'angle': 0, 'k_rot': 2, 'theta_eq': 0*-np.pi/6}
]
damping_link = 0.25
arm = RoboticArm(base, link_params)
engine.add_body(arm)



while engine.gui_flag:

    t = engine.time

    newbase = np.array([0.5, 0.6 - 0.15*np.sin(5*t)])             #[0.5,0.5-0.5*t])      

    arm.set_base(newbase)

    external_torques = [0 for _ in arm.links]

    link_segments = arm.get_link_segments()
    k_arm = 10000    # collision stiffness for the arm
    link_radius = 0.05

    # Check each soft body node against every link segment.
    for i in range(len(soft_body.positions)):
        p = soft_body.positions[i]
        for seg_index, (A, B) in enumerate(link_segments):
            collision, penetration, normal, closest = detect_collision(p, A, B, link_radius)
            if collision:
                F_coll = k_arm * penetration * normal
                # Adjust the soft body node's velocity (as a simple collision response)
                soft_body.velocities[i] += (F_coll / mass_node) * engine.dt
                # Compute the torque reaction on the corresponding arm link.
                r = closest - A
                torque = r[0] * F_coll[1] - r[1] * F_coll[0]
                external_torques[seg_index] -= torque

    # Pass external torques to the arm so its update method can use them.
    arm.set_external_torques(external_torques)

    engine.update()



engine.recorder.save()