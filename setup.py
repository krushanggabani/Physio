from setuptools import setup

if __name__ =='__main__':
    setup(
        name='Physio',
        version='1.0',
        description='HILS LAB',
        author='Krushang Gabani',
        author_email='krushang@buffalo.edu',
        keywords='Physics Simulation',
        packages=[],
        python_requires='>=3.7',
        install_requires = [
            "cuda-python",
            "gym",
            "imageio",
            "imageio-ffmpeg",
            "matplotlib",
            "numpy",
            "opencv-python",
            "open3d",
            "pandas",
            "scipy",
            "taichi",
            "torch",
            "torchvision",
            "pyyaml",
            "pyrender",
            "yacs"
        ]

    )