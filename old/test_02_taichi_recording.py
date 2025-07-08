import taichi as ti
import numpy as np
import imageio

# Initialize Taichi
ti.init(arch=ti.cpu)  # Use ti.gpu if you have a compatible GPU

# Simulation parameters
res = 512
num_frames = 100
radius = 0.3  # Circle radius
center = ti.Vector([0.5, 0.5])  # Center of the circle
angular_speed = 0.2 * np.pi / num_frames  # Full rotation in num_frames steps

# Fields
ball_pos = ti.Vector.field(2, dtype=ti.f32, shape=())

# GUI for visualization
gui = ti.GUI("Circular Motion", res=(res, res))

# Storage for recording frames
frames = []

@ti.kernel
def update_frame(t: ti.i32):
    theta = angular_speed * t
    ball_pos[None] = center + ti.Vector([ti.cos(theta), ti.sin(theta)]) * radius

# Simulation loop
for t in range(num_frames):
    update_frame(t)
    
    # Draw the ball
    gui.clear(0x112F41)  # Dark blue background
    gui.circle(ball_pos[None], radius=10, color=0xFFFFFF)  # Gold color

    # Save frame
    frame = gui.get_image()
    frame = (np.array(frame) * 255).astype(np.uint8)
    frames.append(np.array(frame))

    gui.show()

# Save animation
imageio.mimsave("circular_motion02.gif", frames, fps=30)
print("Animation saved as circular_motion.gif")
