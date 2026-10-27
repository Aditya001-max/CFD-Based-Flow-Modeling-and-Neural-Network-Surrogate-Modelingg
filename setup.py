from setuptools import setup, find_packages

setup(
    name="cfd-ml-surrogate",
    version="1.0.0",
    author="Aditya001-max",
    author_email="kaushikadityaraj001@gmail.com",
    description="Deep Learning Surrogate Modeling Pipeline for 2D CFD Steady Flow and Drag Prediction",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Aditya001-max/Computational-Fluid-Dynamics-Machine-Learning-Examples",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tensorflow>=2.12.0",
        "keras>=2.12.0",
        "vtk>=9.2.0",
        "tqdm>=4.65.0",
    ],
)
