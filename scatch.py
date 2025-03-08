import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# Generate 10 animated image sequences (10x10 pixels, changing over time)
num_images = 10
image_size = (10, 10)
num_time_steps = 30  # Number of animation frames per image

# Create random animated data (each frame is a 10x10 image that evolves over time)
image_sequences = [np.random.rand(num_time_steps, *image_size) for _ in range(num_images)]

# Create figure and 3D axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Animation variables
z_spacing = 2  # Spacing between slices
max_lift = 10  # How far the image moves up before rotating
num_transition_frames = 30  # Smooth transition steps

def ease_out_quad(t):
    """Easing function for smooth motion (quadratic ease-out)."""
    return 1 - (1 - t) ** 2

def animate(frame):
    ax.clear()
    
    # Hide axis labels for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_title("Dynamic 3D Animation with Extracted Image")

    cycle = frame // num_transition_frames  # Current selected image
    progress = (frame % num_transition_frames) / num_transition_frames  # Progress 0 to 1
    eased_progress = ease_out_quad(progress)  # Apply easing function

    # Determine the current animation step for images
    image_frame = frame % num_time_steps

    for i in range(num_images):
        img = image_sequences[i][image_frame]  # Get evolving image

        x, y = np.meshgrid(range(image_size[0]), range(image_size[1]))

        if i == cycle % num_images:
            # Smooth lift up
            z = np.full_like(x, i * z_spacing + eased_progress * max_lift)

            # Rotate towards user in second half of animation
            rotation_angle = eased_progress * 90  # Rotate up to 90 degrees
            x_rot = x * np.cos(np.radians(rotation_angle)) - z * np.sin(np.radians(rotation_angle))
            z_rot = x * np.sin(np.radians(rotation_angle)) + z * np.cos(np.radians(rotation_angle))

            ax.plot_surface(x_rot, y, z_rot, facecolors=plt.cm.plasma(img), rstride=1, cstride=1, shade=False, alpha=1.0)

        else:
            z = np.full_like(x, i * z_spacing)
            ax.plot_surface(x, y, z, facecolors=plt.cm.plasma(img), rstride=1, cstride=1, shade=False, alpha=0.5)

    ax.view_init(elev=30, azim=210)

# Create animation
frames_total = max(num_images * num_transition_frames, num_time_steps)  # Ensure smooth looping
ani = animation.FuncAnimation(fig, animate, frames=frames_total, interval=50, repeat=True)

plt.show()
