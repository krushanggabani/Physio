import imageio

class Recorder:
    def __init__(self, output_file='output.gif', fps=30):
        self.output_file = output_file
        self.fps = fps
        self.frames = []

    def add_frame(self, frame):
        """
        Add a frame to the recorder.
        frame should be a NumPy array (e.g., from GUI.get_image() or a screenshot).
        """
        self.frames.append(frame)

    def save(self):
        """Write the accumulated frames to the video file."""
        imageio.mimsave(self.output_file, self.frames, fps=self.fps)
