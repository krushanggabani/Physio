#!/usr/bin/env python3
"""
Benchmark script to test MPM simulation performance improvements
"""

import time
import numpy as np
from mpm3dsim import MPM3DSim
from config import Config
from render import Renderer3D, HighPerformanceRenderer3D


def benchmark_simulation_only(n_particles=50000, n_frames=100, n_substeps=20):
    """Benchmark simulation performance without rendering"""
    print(f"Benchmarking simulation only: {n_particles} particles, {n_frames} frames")
    
    config = Config()
    config.n_particles = n_particles
    sim = MPM3DSim(config)
    sim.init()
    
    times = []
    start_time = time.time()
    
    for frame in range(n_frames):
        frame_start = time.time()
        sim.step(n_substeps=n_substeps)
        frame_time = time.time() - frame_start
        times.append(frame_time)
        
        if frame % 10 == 0:
            avg_time = np.mean(times[-10:]) if times else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Frame {frame}: {avg_time*1000:.2f}ms ({fps:.1f} FPS)")
    
    total_time = time.time() - start_time
    avg_frame_time = np.mean(times)
    avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
    
    print(f"Simulation Benchmark Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average frame time: {avg_frame_time*1000:.2f}ms")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Particles per second: {n_particles * avg_fps:.0f}")
    
    return avg_frame_time, avg_fps


def benchmark_rendering(renderer_class, n_particles=25000, n_frames=50):
    """Benchmark rendering performance"""
    renderer_name = renderer_class.__name__
    print(f"Benchmarking {renderer_name}: {n_particles} particles, {n_frames} frames")
    
    config = Config()
    config.n_particles = n_particles
    config.max_render_particles = n_particles
    
    sim = MPM3DSim(config)
    sim.init()
    
    renderer = renderer_class(camera_height=3.0, floor_size=2.0, config=config)
    
    # Warmup
    for _ in range(5):
        sim.step(n_substeps=5)
        renderer.render(sim=sim)
    
    times = []
    start_time = time.time()
    
    for frame in range(n_frames):
        sim.step(n_substeps=10)  # Light simulation for render testing
        
        render_start = time.time()
        renderer.render(sim=sim)
        render_time = time.time() - render_start
        times.append(render_time)
        
        if frame % 10 == 0:
            avg_time = np.mean(times[-10:]) if times else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Frame {frame}: {avg_time*1000:.2f}ms render ({fps:.1f} render FPS)")
    
    total_time = time.time() - start_time
    avg_render_time = np.mean(times)
    avg_render_fps = 1.0 / avg_render_time if avg_render_time > 0 else 0
    
    print(f"{renderer_name} Benchmark Results:")
    print(f"  Average render time: {avg_render_time*1000:.2f}ms")
    print(f"  Average render FPS: {avg_render_fps:.2f}")
    
    renderer.cleanup()
    return avg_render_time, avg_render_fps


def compare_renderers():
    """Compare standard vs high performance renderers"""
    print("=" * 60)
    print("RENDERER COMPARISON")
    print("=" * 60)
    
    particle_counts = [10000, 25000, 50000]
    
    for n_particles in particle_counts:
        print(f"\nTesting with {n_particles} particles:")
        print("-" * 40)
        
        # Test standard renderer
        try:
            std_time, std_fps = benchmark_rendering(Renderer3D, n_particles, 30)
        except Exception as e:
            print(f"Standard renderer failed: {e}")
            std_time, std_fps = float('inf'), 0
        
        # Test high performance renderer  
        try:
            hp_time, hp_fps = benchmark_rendering(HighPerformanceRenderer3D, n_particles, 30)
        except Exception as e:
            print(f"High performance renderer failed: {e}")
            hp_time, hp_fps = float('inf'), 0
        
        # Comparison
        if std_time < float('inf') and hp_time < float('inf'):
            speedup = std_time / hp_time
            print(f"Performance improvement: {speedup:.2f}x faster")
            print(f"FPS improvement: {hp_fps/std_fps:.2f}x" if std_fps > 0 else "N/A")
        
        time.sleep(1)  # Brief pause between tests


def memory_usage_test():
    """Test memory usage patterns"""
    print("=" * 60)
    print("MEMORY USAGE TEST")
    print("=" * 60)
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    particle_counts = [1000, 5000, 10000, 25000, 50000]
    
    for n_particles in particle_counts:
        # Measure before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        config = Config()
        config.n_particles = n_particles
        sim = MPM3DSim(config)
        sim.init()
        
        # Run a few steps
        for _ in range(10):
            sim.step(n_substeps=5)
        
        # Measure after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_per_particle = (mem_after - mem_before) / n_particles * 1024  # KB per particle
        
        print(f"{n_particles} particles: {mem_after-mem_before:.1f}MB total, "
              f"{mem_per_particle:.2f}KB per particle")
        
        # Cleanup
        del sim
        time.sleep(0.5)


def main():
    """Run all benchmarks"""
    print("MPM 3D Simulation Performance Benchmark")
    print("=" * 60)
    
    # Simulation benchmark
    print("\n1. SIMULATION PERFORMANCE")
    print("=" * 30)
    benchmark_simulation_only(n_particles=50000, n_frames=50, n_substeps=20)
    
    time.sleep(2)
    
    # Rendering comparison
    print("\n2. RENDERING PERFORMANCE")
    compare_renderers()
    
    time.sleep(2)
    
    # Memory usage
    print("\n3. MEMORY USAGE")
    try:
        memory_usage_test()
    except ImportError:
        print("psutil not available, skipping memory test")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()