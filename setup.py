import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# REPLACE ME! REPLACE ME! REPLACE ME! REPLACE ME! REPLACE ME! REPLACE ME! REPLACE ME! 
repo_name = ""

setup(
    name=repo_name,
    version="0.0.0",
    description="A short description.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joaopcbertoldo/{}".,format(repo_name),
    author="Joao P C Bertoldo",
    author_email="joaopcbertoldo@gmail.com",
    classifiers=[],
    keywords="",
    packages=find_packages(where=repo_name),
    python_requires="==3.6.8",
    install_requires=[
        "scikit-learn>=0.23.2",
        "pandas>=1.1.0",
        "jupyterlab>=2.2.4",
        "jupyter_contrib_nbextensions>=0.5.1",
        "matplotlib>=3.3.0",
        "seaborn>=0.10.1",
        "tensorflow>=2.3.1",
    ],
    extras_require={
        "dev": [
            "pre-commit>=2.6.0",
            "flake8>=3.8.3",
            "flake8-bugbear>=20.1.4",
            "black>=19.10b0",
            "isort>=5.3.2",
        ],
    },
)
