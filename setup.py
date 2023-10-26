from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="chem_eng_ai",
    version="0.0.2",
    author="Yuri Koide, Arjun J. Kaithakkal, Matthias Schniewind, Bradley P. Ladewig, Alexander Stroh, Pascal Friederich",
    author_email="yuri.koide@kit.edu",
    description="Predicting flow properties from channel geometries for Chemical Engineering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=["tensorflow==2.5.0",
                      "scikit-learn==0.24.2",
                      "pandas==1.3.0",
                      "pyyaml==5.4.1"],
    packages=find_packages(),
    include_package_data=True,
    package_data={"keras_addons": ["*.json", "*.yaml"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords=["Chemical engineering", "Machine learning", "Numerical simulation", "Convolutional neural networks", "Fluid dynamics"]
)
