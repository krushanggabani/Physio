import math
import numpy as np
import trimesh
import pyrender
import pyglet
from mpm3dsim import MPM3DSim
import time


np.infty = np.inf


def rotation_matrix(yaw: float, pitch: float) -> np.ndarray:
    """Utility: build combined yaw and pitch rotation matrix"""
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


class OptimizedRobotArm:
    """Pre-built robot arm meshes with transform-only updates"""
    
    def __init__(self, radius=0.025):
        self.radius = radius
        self.link1_mesh = None
        self.link2_mesh = None
        self.roller_mesh = None
        self._create_static_meshes()
    
    def _create_static_meshes(self):
        """Create static meshes once"""
        # Link1 cylinder (will be scaled/transformed as needed)
        cyl1 = trimesh.creation.cylinder(radius=self.radius, height=1.0, sections=12)
        self.link1_mesh = pyrender.Mesh.from_trimesh(
            cyl1, 
            smooth=False, 
            material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[0, 0, 0.3, 1])
        )
        
        # Link2 cylinder 
        cyl2 = trimesh.creation.cylinder(radius=self.radius, height=1.0, sections=12)
        self.link2_mesh = pyrender.Mesh.from_trimesh(
            cyl2, 
            smooth=False, 
            material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[0, 0, 0.3, 1])
        )
        
        # Roller sphere
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=self.radius)  # Reduced subdivisions
        self.roller_mesh = pyrender.Mesh.from_trimesh(
            sphere, 
            smooth=False, 
            material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1, 0, 0, 1])
        )
    
    def get_transforms(self, base_pt, j2, ee):
        """Calculate transform matrices for robot links"""
        # Link 1 transform
        link1_vec = j2 - base_pt
        link1_length = np.linalg.norm(link1_vec)
        link1_center = (base_pt + j2) / 2
        
        # Link 2 transform  
        link2_vec = ee - j2
        link2_length = np.linalg.norm(link2_vec)
        link2_center = (j2 + ee) / 2
        
        # Create transform matrices
        T1 = self._cylinder_transform(base_pt, j2, link1_length)
        T2 = self._cylinder_transform(j2, ee, link2_length)
        T3 = np.eye(4)
        T3[:3, 3] = ee
        
        return T1, T2, T3
    
    def _cylinder_transform(self, start, end, length):
        """Create transform matrix for cylinder from start to end"""
        if length < 1e-7:
            return np.eye(4)
            
        center = (start + end) / 2
        vec = end - start
        z = vec / length
        
        # Create rotation matrix
        v = np.cross([0, 0, 1], z)
        c = np.dot([0, 0, 1], z)
        s = np.linalg.norm(v)
        
        if s < 1e-7:
            R = np.eye(3)
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        
        # Scale matrix for length
        S = np.diag([1, 1, length])
        
        # Combine into transform
        T = np.eye(4)
        T[:3, :3] = R @ S
        T[:3, 3] = center
        
        return T


class Renderer3D:
    def __init__(self, camera_height: float = 3.0, floor_size: float = 2.0, config=None):
        self.config = config
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Camera setup
        self.camera = pyrender.PerspectiveCamera(yfov=np.pi/6, aspectRatio=1.0)
        self.camera_pose = np.eye(4, dtype=np.float32)
        self.camera_pose[:3, 3] = [0, -2.7, 2]
        self.camera_pose[:3, :3] = rotation_matrix(0, 60)

        # Lighting
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        self.light_pose = np.eye(4, dtype=np.float32)
        self.light_pose[:3, :3] = rotation_matrix(0, -30)

        # Floor mesh
        self._create_floor_mesh(floor_size)
        
        # Robot arm
        self.robot_arm = OptimizedRobotArm(radius=0.025)
        
        # Scene setup
        self.scene = pyrender.Scene()
        self.floor_node = self.scene.add(self.floor_mesh)
        self.cam_node = self.scene.add(self.camera, pose=self.camera_pose)
        self.light_node = self.scene.add(self.light, pose=self.light_pose)
        
        # Node references for updates
        self.viewer = None
        self.particle_node = None
        self.robot_nodes = [None, None, None]
        
        # Performance tracking
        self.render_times = []
        
    def _create_floor_mesh(self, floor_size):
        """Create floor mesh once"""
        vs = np.array([
            [ floor_size,  floor_size, 0],
            [-floor_size,  floor_size, 0],
            [-floor_size, -floor_size, 0],
            [ floor_size, -floor_size, 0]
        ], dtype=np.float32)
        fs = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        floor_trimesh = trimesh.Trimesh(vertices=vs, faces=fs)
        self.floor_mesh = pyrender.Mesh.from_trimesh(floor_trimesh, smooth=False)

    def _apply_lod(self, pts):
        """Apply level-of-detail reduction"""
        if self.config is None:
            return pts
            
        max_particles = getattr(self.config, 'max_render_particles', 25000)
        
        if len(pts) > max_particles:
            # Simple distance-based LOD
            indices = np.random.choice(len(pts), max_particles, replace=False)
            return pts[indices]
        return pts

    def _create_particle_colors(self, pts):
        """Create colors for particles with enhanced visibility"""
        # Simple height-based coloring for better visual feedback
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
        if z_max > z_min:
            normalized_z = (pts[:, 2] - z_min) / (z_max - z_min)
            colors = np.column_stack([
                0.3 + 0.7 * normalized_z,  # Enhanced red component
                0.5 + 0.5 * (1 - normalized_z),  # Enhanced green component  
                0.9 * np.ones_like(normalized_z),  # Bright blue component
                np.ones_like(normalized_z)  # Full alpha
            ])
        else:
            # Fallback bright color for better visibility
            colors = np.tile([0.4, 0.7, 1.0, 1.0], (len(pts), 1))
        
        return colors.astype(np.float32)

    def _setup_point_rendering(self):
        """Configure point rendering - simplified version"""
        # Don't call OpenGL functions before context is ready
        # Point size will be handled by pyrender viewer configuration
        pass
        """Create particle mesh based on rendering mode"""
        render_mode = getattr(self.config, 'particle_render_mode', 'points')
        
        if render_mode == "points":
            # Point cloud with larger point size
            return pyrender.Mesh.from_points(
                pts, 
                colors=colors,
                material=pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.5,
                    doubleSided=True
                )
            )
        
        elif render_mode == "small_spheres":
            # Small sphere instances (compromise between performance and quality)
            subdivisions = getattr(self.config, 'sphere_subdivisions', 1)
            sphere_tm = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.004)
            sphere_tm.visual.vertex_colors = np.tile([60, 160, 255, 255], (sphere_tm.vertices.shape[0], 1))
            
            # Create transforms
            N = len(pts)
            tfs = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], N, axis=0)
            tfs[:, :3, 3] = pts.astype(np.float32)
            
            return pyrender.Mesh.from_trimesh(sphere_tm, poses=tfs, smooth=False)
        
        elif render_mode == "instanced_spheres":
            # Original sphere instancing (for comparison/fallback)
            sphere_tm = trimesh.creation.uv_sphere(radius=0.003)
            RGBA = np.array([60, 160, 255, 255], dtype=np.uint8)
            sphere_tm.visual.vertex_colors = np.tile(RGBA, (sphere_tm.vertices.shape[0], 1))
            
            N = len(pts)
            tfs = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], N, axis=0)
            tfs[:, :3, 3] = pts.astype(np.float32)
            
            return pyrender.Mesh.from_trimesh(sphere_tm, poses=tfs, smooth=False)
        
        else:
            # Default to points
            return pyrender.Mesh.from_points(pts, colors=colors)

    def render(self, sim: MPM3DSim):
        """Optimized rendering method"""
        render_start = time.time()
        
        # Get particle data with LOD
        if hasattr(sim, 'get_particle_data_for_rendering'):
            max_particles = getattr(self.config, 'max_render_particles', None)
            pts = sim.get_particle_data_for_rendering(max_particles)
        else:
            pts = sim.x.to_numpy()
            pts = self._apply_lod(pts)
        
        # Create particle mesh with chosen rendering mode
        colors = self._create_particle_colors(pts)
        particle_mesh = self._create_particle_mesh(pts, colors)
        
        # Get robot arm poses
        base_pt = np.array([sim.base_x, sim.base_y, sim.base_z], dtype=np.float32)
        j2 = base_pt + np.array([np.sin(sim.theta1[0]), 0, -np.cos(sim.theta1[0])]) * sim.L1
        ee = sim.roller_center[0].to_numpy()
        
        T1, T2, T3 = self.robot_arm.get_transforms(base_pt, j2, ee)

        # Initialize viewer if needed
        if self.viewer is None:
            # Create viewer with point size configuration
            viewer_kwargs = {
                'use_raymond_lighting': True,
                'run_in_thread': True,
                'window_title': "Optimized Robot Tissue Interaction",
                'viewport_size': (1920, 1080),
                'show_world_axis': False,
                'show_mesh_axes': False,
            }
            
            # Add point size if available in pyrender version
            point_size = getattr(self.config, 'particle_point_size', 8.0)
            try:
                viewer_kwargs['point_size'] = point_size
            except:
                pass  # Older pyrender versions may not support this
            
            self.viewer = pyrender.Viewer(self.scene, **viewer_kwargs)
            
            # Add robot arm nodes once
            self.robot_nodes[0] = self.scene.add(self.robot_arm.link1_mesh, pose=T1)
            self.robot_nodes[1] = self.scene.add(self.robot_arm.link2_mesh, pose=T2)  
            self.robot_nodes[2] = self.scene.add(self.robot_arm.roller_mesh, pose=T3)

        # Update scene (minimized operations)
        with self.viewer.render_lock:
            # Update particles - only remove/add particle node
            if self.particle_node is not None:
                self.scene.remove_node(self.particle_node)
            self.particle_node = self.scene.add(particle_mesh)
            
            # Update robot arm transforms only (no mesh recreation)
            self.robot_nodes[0].matrix = T1
            self.robot_nodes[1].matrix = T2
            self.robot_nodes[2].matrix = T3
        
        # Performance tracking
        render_time = time.time() - render_start
        self.render_times.append(render_time)
        self.frame_count += 1
        
        # Print render performance every 100 frames
        if self.frame_count % 100 == 0:
            avg_render_time = np.mean(self.render_times[-100:])
            render_fps = 1.0 / avg_render_time if avg_render_time > 0 else 0
            print(f"Render Performance: {avg_render_time*1000:.2f}ms avg | "
                  f"Render FPS: {render_fps:.1f} | Particles rendered: {len(pts)}")
            
    def cleanup(self):
        """Cleanup resources"""
        if self.viewer is not None:
            self.viewer.close_external()


class ParticleBuffer:
    """Double-buffered particle rendering for even better performance"""
    
    def __init__(self, max_particles=50000):
        self.max_particles = max_particles
        self.buffer_a = None
        self.buffer_b = None
        self.current_buffer = 'a'
        self.mesh_cache = {}
        
    def get_mesh(self, pts, colors):
        """Get mesh from cache or create new one"""
        buffer_key = f"{len(pts)}_{hash(pts.tobytes())}"
        
        if buffer_key not in self.mesh_cache:
            # Simple point cloud without material parameter
            self.mesh_cache[buffer_key] = pyrender.Mesh.from_points(pts, colors=colors)
            
            # Limit cache size
            if len(self.mesh_cache) > 10:
                # Remove oldest entries
                keys_to_remove = list(self.mesh_cache.keys())[:-5]
                for key in keys_to_remove:
                    del self.mesh_cache[key]
        
        return self.mesh_cache[buffer_key]


class FrustumCuller:
    """Simple frustum culling for particles"""
    
    def __init__(self, camera_pose, fov=np.pi/6):
        self.camera_pose = camera_pose
        self.fov = fov
        
    def cull_particles(self, pts):
        """Remove particles outside camera frustum"""
        # Simple distance culling for now
        camera_pos = self.camera_pose[:3, 3]
        distances = np.linalg.norm(pts - camera_pos, axis=1)
        
        # Keep particles within reasonable distance
        max_distance = 5.0
        mask = distances < max_distance
        
        return pts[mask], mask


# Alternative high-performance renderer for very large particle counts
class HighPerformanceRenderer3D(Renderer3D):
    """Ultra-optimized renderer for maximum performance"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.particle_buffer = ParticleBuffer()
        self.frustum_culler = FrustumCuller(self.camera_pose)
        self.update_counter = 0
        
    def _create_particle_mesh(self, pts, colors):
        """Optimized particle mesh creation for high performance"""
        render_mode = getattr(self.config, 'particle_render_mode', 'small_spheres')
        
        # For high performance, prefer small spheres over points for round particles
        if render_mode == "instanced_spheres":
            print("High performance mode: switching from instanced_spheres to small_spheres")
            render_mode = "small_spheres"
        
        if render_mode == "points":
            # Fast point cloud rendering (square particles)
            return pyrender.Mesh.from_points(pts, colors=colors)
        elif render_mode == "small_spheres":
            # Optimized sphere instances with minimal subdivisions (round particles)
            radius = getattr(self.config, 'sphere_radius', 0.005)
            sphere_tm = trimesh.creation.icosphere(subdivisions=0, radius=radius)  # Force low subdivision for performance
            sphere_tm.visual.vertex_colors = np.tile([100, 200, 255, 255], (sphere_tm.vertices.shape[0], 1))
            
            N = len(pts)
            tfs = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], N, axis=0)
            tfs[:, :3, 3] = pts.astype(np.float32)
            
            return pyrender.Mesh.from_trimesh(sphere_tm, poses=tfs, smooth=False)
        else:
            # Default to small spheres for round particles
            radius = getattr(self.config, 'sphere_radius', 0.005)
            sphere_tm = trimesh.creation.icosphere(subdivisions=0, radius=radius)
            sphere_tm.visual.vertex_colors = np.tile([100, 200, 255, 255], (sphere_tm.vertices.shape[0], 1))
            
            N = len(pts)
            tfs = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], N, axis=0)
            tfs[:, :3, 3] = pts.astype(np.float32)
            
            return pyrender.Mesh.from_trimesh(sphere_tm, poses=tfs, smooth=False)
        
    def render(self, sim: MPM3DSim):
        """Ultra-optimized rendering with all optimizations enabled"""
        render_start = time.time()
        
        # Get particle data with more aggressive LOD
        pts = sim.get_particle_data_for_rendering(
            getattr(self.config, 'max_render_particles', 15000)  # More aggressive LOD
        )
        
        # Apply frustum culling every few frames
        if self.update_counter % 5 == 0:  # Cull every 5 frames
            if getattr(self.config, 'enable_frustum_culling', True):
                pts, _ = self.frustum_culler.cull_particles(pts)
        
        # Reduced color computation for better performance
        if len(pts) > 0:
            colors = self._create_particle_colors(pts)
            particle_mesh = self._create_particle_mesh(pts, colors)
        else:
            particle_mesh = None
        
        # Robot arm updates (same as parent)
        base_pt = np.array([sim.base_x, sim.base_y, sim.base_z], dtype=np.float32)
        j2 = base_pt + np.array([np.sin(sim.theta1[0]), 0, -np.cos(sim.theta1[0])]) * sim.L1
        ee = sim.roller_center[0].to_numpy()
        T1, T2, T3 = self.robot_arm.get_transforms(base_pt, j2, ee)

        # Initialize viewer with performance settings
        if self.viewer is None:
            # Create viewer with basic configuration
            viewer_kwargs = {
                'use_raymond_lighting': True,
                'run_in_thread': True,
                'window_title': "High Performance Robot Tissue Interaction",
                'viewport_size': (1280, 720),  # Lower resolution for performance
                'show_world_axis': False,
                'show_mesh_axes': False,
            }
            
            # Try to add point size configuration
            point_size = getattr(self.config, 'particle_point_size', 8.0)
            try:
                viewer_kwargs['point_size'] = point_size
            except:
                pass  # Ignore if not supported
            
            self.viewer = pyrender.Viewer(self.scene, **viewer_kwargs)
            self.robot_nodes[0] = self.scene.add(self.robot_arm.link1_mesh, pose=T1)
            self.robot_nodes[1] = self.scene.add(self.robot_arm.link2_mesh, pose=T2)  
            self.robot_nodes[2] = self.scene.add(self.robot_arm.roller_mesh, pose=T3)

        # Minimal scene updates
        with self.viewer.render_lock:
            if particle_mesh is not None:
                if self.particle_node is not None:
                    self.scene.remove_node(self.particle_node)
                self.particle_node = self.scene.add(particle_mesh)
            
            # Update robot transforms only every few frames for even better performance
            if self.update_counter % 2 == 0:
                self.robot_nodes[0].matrix = T1
                self.robot_nodes[1].matrix = T2
                self.robot_nodes[2].matrix = T3
        
        self.update_counter += 1
        
        # Performance tracking
        render_time = time.time() - render_start
        self.render_times.append(render_time)
        self.frame_count += 1
        
        if self.frame_count % 50 == 0:  # More frequent updates
            avg_render_time = np.mean(self.render_times[-50:])
            render_fps = 1.0 / avg_render_time if avg_render_time > 0 else 0
            print(f"HP Render: {avg_render_time*1000:.1f}ms | "
                  f"FPS: {render_fps:.1f} | Particles: {len(pts)}")