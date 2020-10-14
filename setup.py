raise EnvironmentError("The setup.py has been deprecated because using pipy's distribution of tensorflow-gpu seems not to work properly. Use the conda environment.")

import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

repo_name = "tomo2seg"

cnn_segm_path = str((here / "cnn_segm").absolute())
cnn_segm_packages = [f"{cnn_segm_path}/{package}" for package in find_packages(where=cnn_segm_path)]

setup(
    name=repo_name,
    version="0.0.0",
    description="A short description.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/joaopcbertoldo/{repo_name}",
    author="Joao P C Bertoldo",
    author_email="joaopcbertoldo@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    keywords="",
    packages=find_packages(where=repo_name) + cnn_segm_packages,
    python_requires=">=3.6.8",
    install_requires=[
	# todo update versions with those from conda?
        "numpy==1.18.5",  # it must be < 1.19.0 because of an incompatibility with tensorflow
        "scikit-learn==0.23.2",
        "scikit-image==0.17.2",
        "pandas==1.1.0",
        "jupyterlab==2.2.4",
        "jupyter_contrib_nbextensions==0.5.1",
        "matplotlib==3.3.0",
        "seaborn==0.10.1",
        "tensorflow==2.3.1",  # install tensorflow-gpu instead!
        "pymicro==0.4.5",
        "tensorboard==2.3.0",
        #"wandb==0.10.2",  # i cannot install this because it depends on psutil, which can only be installed with python3-devel, which i cannot install without sudo...
        "scipy== 1.5.2",
        "streamlit==0.67.1",
        "Keras==2.4.3",
        "imageio==2.5.0",
        "opencv-python==4.1.0.25",
        "pytest==5.4.1",
        "humanize==3.0.1",
    ],
    extras_require={
        "dev": [
            "pre-commit==2.6.0",
            "flake8==3.8.3",
            "flake8-bugbear==20.1.4",
            "black==19.10b0",
            "isort==5.3.2",
        ],
    },
)
