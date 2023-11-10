from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess


def custom_command():
    subprocess.run(["chmod", "+x", "./setup.sh"])
    out = subprocess.run(["bash", "./setup.sh"], capture_output=True)


class CustomInstall(install):
    def run(self):
        install.run(self)
        custom_command()


class CustomDevelop(develop):
    def run(self):
        develop.run(self)
        custom_command()


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="chemengsim",
    version="0.0.1",
    author="Alexander Stroh, Matthias Schniewind",
    author_email="matthias.schniewind@kit.edu",
    description="High-throughput simulations for 2D laminar-flow channels in SIMSON",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=["numpy", "scipy", "pandas", "tables", "h5py", "tqdm", "matplotlib", "pyyaml"],
    packages=find_packages(),
    include_package_data=True,
    package_data={"chemengsim": ["*.json", "*.yml"]},
    cmdclass={'install': CustomInstall,
              'develop': CustomDevelop},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords=["materials", "science", "high", "throughput", "fluid", "dynamics", "chemical", "engineering"]
)
