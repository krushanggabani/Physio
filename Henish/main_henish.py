import time
import sys
import argparse
from mpm3dsim import MPM3DSim
from config import Config
from render import Renderer3D, HighPerformanceRenderer3D

def parse_args():
    parser = argparse.ArgumentParser(description='MPM 3D Simulation with Optimized Rendering')
    parser.add_argument('--performance_mode', action='store_true',
                       help='Use high performance renderer')
    parser.add_argument('--particles', type=int, default=50000, 
                       help='Number of particles')
    parser.add_argument('--max_render_particles', type=int, default=25000,
                       help='Maximum particles to render (LOD)')
    parser.add_argument('--point_size', type=float, default=8.0,
                       help='Point size for particle rendering')
    parser.add_argument('--render_mode', choices=['points', 'small_spheres', 'instanced_spheres'],
                       default='small_spheres', help='Particle rendering mode (small_spheres for round particles)')
    parser.add_argument('--sphere_radius', type=float, default=0.005,
                       help='Radius of sphere particles (0.003-0.010)')
    parser.add_argument('--sphere_subdivisions', type=int, default=1,
                       help='Sphere quality: 0=fast, 1=balanced, 2=high quality')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    
    config = Config()
    config.n_particles = args.particles
    config.max_render_particles = args.max_render_particles
    config.particle_point_size = args.point_size
    config.particle_render_mode = args.render_mode
    config.sphere_radius = args.sphere_radius
    config.sphere_subdivisions = args.sphere_subdivisions
    
    print(f"MPM 3D Simulation Configuration:")
    print(f"  Particles: {args.particles}")
    print(f"  Render mode: {args.render_mode}")
    if args.render_mode in ['small_spheres', 'instanced_spheres']:
        print(f"  Sphere radius: {args.sphere_radius}")
        print(f"  Sphere subdivisions: {args.sphere_subdivisions}")
    else:
        print(f"  Point size: {args.point_size}")
    print(f"  Max render particles: {args.max_render_particles}")
    print(f"  Performance mode: {args.performance_mode}")
    
    sim = MPM3DSim(config)
    sim.init()

    if args.performance_mode:
        render = HighPerformanceRenderer3D(
            camera_height=3.0, 
            floor_size=2.0,
            config=config
        )
        print("Using High Performance Renderer")
    else:
        render = Renderer3D(
            camera_height=3.0, 
            floor_size=2.0,
            config=config
        )
        print("Using Standard Optimized Renderer")
    
    render.render(sim=sim)
    
    # Reduced sleep for better performance monitoring
    time.sleep(0.1)
    
    frame_count = 0
    start_time = time.time()
    
    print("Starting simulation loop... Press Ctrl+C to stop")
    
    try:
        for _ in range(1000):
            sim.step(n_substeps=20)
            render.render(sim=sim)
            
            # Performance monitoring every 100 frames
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Frame {frame_count}: {fps:.2f} FPS | Particles: {sim.n_particles}")
            
            # Minimal sleep to prevent system overload
            time.sleep(1e-6)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        render.cleanup()
        print("Cleanup completed")