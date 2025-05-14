import numpy as np
import trimesh
import pyrender
import os
import math
import time

os.environ['PYOPENGL_PLATFORM'] = 'egl'
np.infty = np.inf  # pyrender compatibility

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

def yaw_pitch_from_R(R: np.ndarray) -> tuple[float, float]:
    """
    Extract yaw (around Y) and pitch (around X) from R = Ry(yaw) @ Rx(pitch).
    Returns (yaw_deg, pitch_deg).
    """
    pitch = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
    yaw   = math.degrees(math.atan2( R[0, 2], R[2, 2]))
    return yaw, pitch

class Render:
    def __init__(self, config):
        self.config = config
        
        # --- Camera setup ---
        cam_height = 10.0
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi/6, aspectRatio=1.0)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.camera_pose[:3, 3] = [0.17, -6.7, 4.57]
        self.camera_pose[:3, :3] = rotation_matrix(0, 60)

        # --- Light setup ---s
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self.light_pose = np.eye(4, dtype=np.float32)
        self.light_pose[:3,:3] = rotation_matrix(0, -30)

         # --- Floor mesh ---
        floor_size = 2.0
        vs = np.array([
            [ floor_size,  floor_size, 0],
            [-floor_size,  floor_size, 0],
            [-floor_size, -floor_size, 0],
            [ floor_size, -floor_size, 0]
        ], dtype=np.float32)
        fs = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        floor_tm = trimesh.Trimesh(vertices=vs, faces=fs)
        self.floor_mesh = pyrender.Mesh.from_trimesh(floor_tm, smooth=False)


        ############################
        # -- Link mesh ---
        box_tm = trimesh.creation.box(extents=config.link.size)
        red_rgba = [1.0, 0.0, 0.0, 1.0]
        red_mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=red_rgba,
            metallicFactor=0.0,
            roughnessFactor=1.0
        )
        self.box_mesh = pyrender.Mesh.from_trimesh(box_tm, material=red_mat, smooth=True)
        self.box_pose = np.eye(4, dtype=np.float32)

        self.box_pose[:3, 3] = [-0.5, 0.0, 2]



        # --- Scene setup ---
        self.scene = pyrender.Scene()
        # Add static nodes

        self.floor_node = self.scene.add(self.floor_mesh)
        self.cam_node   = self.scene.add(self.camera,    pose=self.camera_pose)
        self.light_node = self.scene.add(self.light,     pose=self.light_pose)

        # Will be created on first view()
        self.box_node   = None
        self.cloud_node = None
        self.viewer     = None


    def view(self, points: np.ndarray, colors: np.ndarray):
        """
        Update point cloud and render scene in a background thread.
        Prints camera position & orientation each call.
        """

        # Launch the viewer on first call
        if self.viewer is None:
            self.viewer = pyrender.Viewer(
                self.scene,
                use_raymond_lighting=True,
                run_in_thread=True,
                fullscreen=True,
                window_title = "Robot Tissue Interaction",
                viewport_size=(1920, 1080)
            )
            # Give the window a moment to initialize
            time.sleep(0.1)

        # Create a new point cloud mesh
        cloud = pyrender.Mesh.from_points(points, colors=colors)


        self.box_pose[:3, 3] = [-0.5, 0.0, 2]

        # Safely swap out the old cloud under render_lock
        with self.viewer.render_lock:
            if self.cloud_node is not None:
                self.scene.remove_node(self.cloud_node)
            self.cloud_node = self.scene.add(cloud)

            # self.box_node   = self.scene.add(self.box_mesh, pose=self.box_pose)

        # # Print current camera pose
        # self.cam_node = next(n for n in self.scene.get_nodes() if n.camera is not None)
        # M = self.scene.get_pose(self.cam_node)
        # pos = M[:3, 3]
        # R   = M[:3, :3]
        # yaw, pitch = yaw_pitch_from_R(R)
        # print(f"Camera position: {pos}")
        # print(f"Orientation (yaw, pitch): ({yaw:.1f}°, {pitch:.1f}°)")

        # Brief pause so the render thread can catch up
        time.sleep(1e-3)
