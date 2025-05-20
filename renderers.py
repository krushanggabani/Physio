"""
Defines abstract Renderer and concrete 2D/3D renderers.
"""
import time
import numpy as np
import taichi as ti
import pyrender
import trimesh

class Renderer:
    def render(self):
        raise NotImplementedError

class Taichi2DRenderer(Renderer):
    def __init__(self, sim):
        self.sim = sim
        self.gui = ti.GUI('MPM 2D', res=(512, 512))

    def render(self):
        while self.gui.running:
            self.sim.step()
            pts = self.sim.soft_body.x.to_numpy()
            self.gui.circles(pts, radius=1.5, color=0x66CCFF)
            self.gui.show()




class PyRender3DRenderer(Renderer):
    def __init__(self, sim):
        self.sim = sim
        self.viewer = None
        self._setup_scene()

    def _setup_scene(self):
        self.scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.3,0.3,0.3])
        # floor
        vs = np.array([[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]],dtype=np.float32)
        fs = np.array([[0,1,2],[0,2,3]],dtype=np.int32)
        floor = trimesh.Trimesh(vertices=vs, faces=fs)
        self.scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False))
        # point cloud placeholder
        self.pc = pyrender.Mesh.from_points(np.zeros((1,3)), colors=np.ones((1,4)))
        self.pc_node = self.scene.add(self.pc)
        # sphere as robot link placeholder
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.05)
        self.sphere = pyrender.Mesh.from_trimesh(sphere, smooth=True)
        self.sphere_node = self.scene.add(self.sphere)
        # camera
        cam = pyrender.PerspectiveCamera(yfov=np.pi/6, aspectRatio=1.0)
        cam_pose = np.eye(4)
        cam_pose[:3,3] = [0, -2.7, 2]
        angle = np.deg2rad(60)
        cam_pose[:3,:3] = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]],dtype=np.float32)
        self.scene.add(cam, pose=cam_pose)
        # light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        light_pose = np.eye(4)
        angle_l = np.deg2rad(-30)
        light_pose[:3,:3] = np.array([[1,0,0],[0,np.cos(angle_l),-np.sin(angle_l)],[0,np.sin(angle_l),np.cos(angle_l)]],dtype=np.float32)
        self.scene.add(light, pose=light_pose)

    def render(self):
        if self.viewer is None:
            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, run_in_thread=True)
        while True:
            self.sim.step()
            pts = self.sim.soft_body.x.to_numpy()
            # update point cloud
            with self.viewer.render_lock:
                self.pc.primitives[0].positions = pts
                # update end effector sphere position
                ee = self.sim.robot.forward_kinematics()
                mat = np.eye(4)
                mat[:3,3] = ee
                self.scene.set_pose(self.sphere_node, pose=mat)
            time.sleep(1/60)