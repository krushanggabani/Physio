import taichi as ti
import numpy as np
import pyrender
import trimesh
import math

# Fix NumPy 2.0 removal of np.infty for pyrender compatibility
np.infty = np.inf

# Utility: build combined yaw (around Y-axis) and pitch (around X-axis) rotation matrix
def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
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

class Renderer:
    def __init__(self, camera_height: float = 10.0, floor_size: float = 2.0):
        """
        Set up camera, light, and floor mesh for scene visualization.
        camera_height: Z-axis position of the camera
        floor_size: half-extent of square floor in X and Y
        """
        # Perspective camera (field-of-view 30°)
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi/6, aspectRatio=1.0)
        # Build camera pose (translation + orientation)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.camera_pose[:3, 3] = np.array([0, 0, camera_height], dtype=np.float32)
        self.camera_pose[:3, :3] = rotation_matrix(yaw=0.0, pitch=0.0)

        # Directional light pitched downward by 30°
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self.light_pose = np.eye(4, dtype=np.float32)
        self.light_pose[:3, :3] = rotation_matrix(yaw=0.0, pitch=-math.pi/6)

        # Create a flat floor mesh in the X-Y plane at Z=0
        vs = np.array([
            [ floor_size,  floor_size, 0],
            [-floor_size,  floor_size, 0],
            [-floor_size, -floor_size, 0],
            [ floor_size, -floor_size, 0]
        ], dtype=np.float32)
        fs = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        floor_trimesh = trimesh.Trimesh(vertices=vs, faces=fs)
        self.floor_mesh = pyrender.Mesh.from_trimesh(floor_trimesh, smooth=False)


        # Create a unit sphere (radius=1)
        sphere_trimesh = trimesh.creation.uv_sphere(radius=0.2)
        self.sphere_mesh = pyrender.Mesh.from_trimesh(sphere_trimesh, smooth=True)
        # Position sphere at (3, 0, 1)
        self.sphere_pose = np.eye(4, dtype=np.float32)
        self.sphere_pose[:3, 3] = np.array([0.0, 0.0, 1.5], dtype=np.float32)

        # Create an axis-aligned box (1 x 2 x 0.5)
        box_trimesh = trimesh.creation.box(extents=(0.2, 0.2, 0.5))
        self.box_mesh = pyrender.Mesh.from_trimesh(box_trimesh, smooth=True)
        # Position box at (-3, 0, 0.25) so it sits on the floor
        self.box_pose = np.eye(4, dtype=np.float32)
        self.box_pose[:3, 3] = np.array([-0.5, 0.0, 0.25], dtype=np.float32)



    def render(self, points: np.ndarray, colors: np.ndarray):
        """
        Build the Pyrender scene and launch the viewer.

        points: (N,3) array of XYZ coords
        colors: (N,4) array of RGBA values
        """
        scene = pyrender.Scene()

        # Add point cloud as a mesh of point sprites
        cloud = pyrender.Mesh.from_points(points, colors=colors)
        scene.add(cloud)
        
        # Add primitives: sphere and box
        scene.add(self.sphere_mesh, pose=self.sphere_pose)
        scene.add(self.box_mesh, pose=self.box_pose)

        # Add floor, camera, and light
        scene.add(self.floor_mesh)
        scene.add(self.camera, pose=self.camera_pose)
        scene.add(self.light, pose=self.light_pose)
        
        # Launch interactive viewer (blocks until closed)
        pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == '__main__':
    # Initialize Taichi on CPU (or change to ti.gpu)
    ti.init(arch=ti.cpu)

    # Number of points
    N = 2_000_000
    # Allocate Taichi field for 3D points
    pts_field = ti.Vector.field(3, dtype=ti.f32, shape=N)

    @ti.kernel
    def init_points():
        # Fill with random values in [-1,1]^3
        for i in pts_field:
            pts_field[i] = ti.Vector([ti.random(), ti.random(), ti.random()])

    # Execute the Taichi kernel
    init_points()
    # Transfer data to NumPy
    points_np = pts_field.to_numpy()

    # Build RGBA colors: map XYZ from [-1,1] to [0,1]
    colors = np.ones((N, 4), dtype=np.float32)
    colors[:, :3] = (points_np + 1.0) * 0.5

    # Render the scene
    renderer = Renderer(camera_height=10.0, floor_size=2.0)
    renderer.render(points_np, colors)
