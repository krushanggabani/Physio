# --------------------------------------------------------------------------------
# Copyright (c) 2025 Krushang Gabani
# All rights reserved.
#
# PyRenderer3D: Object-oriented 3D renderer for MPM simulations using PyRender.
# Includes utilities for camera setup, lighting, floor mesh, particle clouds,
# and robot link primitives (cylinders & rollers).
#
# Author: Krushang Gabani
# Date: July 7, 2025
# --------------------------------------------------------------------------------

import os
import math
import time

import numpy as np
import trimesh
import pyrender


np.infty = np.inf

# Ensure headless rendering uses EGL
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# Utility: build combined yaw (around Y-axis) and pitch (around X-axis) rotation matrix
def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    yaw_mat = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ], dtype=np.float32)
    pitch_mat = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch),  math.cos(pitch)]
    ], dtype=np.float32)
    return yaw_mat @ pitch_mat



class PyRenderer3D:
    """
    3D renderer for Taichi-based MPM simulations.
    """

    def __init__(self,cfg):

        # store config and mode
        self.cfg = cfg
        self.solid_mode = cfg.solid_mode

        # Camera setup
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi/6, aspectRatio=1.0)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.camera_pose[:3, 3] = np.array(cfg.camera_pose, dtype=np.float32)
        yaw, pitch = cfg.camera_rotation
        self.camera_pose[:3, :3] = rotation_matrix(yaw, pitch)

        # Light setup
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self.light_pose = np.eye(4, dtype=np.float32)
        self.light_pose[:3, 3] = np.array(cfg.light_pose, dtype=np.float32)
        lyaw, lpitch = cfg.light_rotation
        self.light_pose[:3, :3] = rotation_matrix(lyaw, lpitch)


        # Floor mesh
        vs = np.array([
            [ cfg.floor_size,  cfg.floor_size, 0],
            [-cfg.floor_size,  cfg.floor_size, 0],
            [-cfg.floor_size, -cfg.floor_size, 0],
            [ cfg.floor_size, -cfg.floor_size, 0]
        ], dtype=np.float32)
        fs = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
        floor_trimesh = trimesh.Trimesh(vertices=vs, faces=fs)
        self.floor_mesh = pyrender.Mesh.from_trimesh(floor_trimesh, smooth=False)

        self.particles = None
        self.particles_color = None

        # Scene and nodes
        self.scene      = pyrender.Scene()
        self.floor_node = self.scene.add(self.floor_mesh)
        self.cam_node   = self.scene.add(self.camera, pose=self.camera_pose)
        self.light_node = self.scene.add(self.light,   pose=self.light_pose)

        # Dynamic node placeholders
        self.object_node  = None
        self.robot_node1  = None
        self.robot_node2  = None
        self.robot_node3  = None
        self.viewer       = None


    def set_particles(self,particles):
        self.particles = particles
        self.particles_color = ((particles + 1.0) * 0.5).astype(np.float32)

    def _reconstruct_solid(self):
        """
        Build a convex-hull surface mesh from the current point cloud.
        """
        hull = trimesh.Trimesh(vertices=self.particles).convex_hull
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.47, 0.0, 0.0, 1])
        return pyrender.Mesh.from_trimesh(hull, smooth=True,material=material)
    
    def render(self):

        # Initialize viewer once
        if self.viewer is None:
            self.viewer = pyrender.Viewer(
                self.scene,
                use_raymond_lighting=True,
                run_in_thread=True,
                window_title="Robot Tissue Interaction"
            )

        if self.solid_mode == "solid":
            cloud = self._reconstruct_solid()
        else:
            N = len(self.particles) 
            sphere_tm = trimesh.creation.uv_sphere(radius=0.002)
            RGBA = np.array([60, 160, 255, 255], dtype=np.uint8)   # pale blue
            sphere_tm.visual.vertex_colors = np.tile(RGBA, (sphere_tm.vertices.shape[0], 1))
            tfs = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], N, axis=0)
            tfs[:, :3, 3] = self.particles.astype(np.float32) 
            cloud = pyrender.Mesh.from_trimesh(sphere_tm, poses=tfs, smooth=False)
        

        with self.viewer.render_lock:
            if self.object_node is None:
                self.object_node = self.scene.add(cloud, name="particles")
            else:
                self.object_node.mesh = cloud


        # Throttle
        time.sleep(1e-3)
        