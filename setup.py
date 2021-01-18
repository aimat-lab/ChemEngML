from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="chem_eng_ml",
    version="0.0.1",
    author="Matthias Schniewind",
    author_email="matthias.schniewind@kit.edu",
    description="Predicting flow properties from channel geometries for Chemical Engineering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=["numpy",
                      "pandas",
                      "mlflow",
                      "scikit-learn",
                      "h5py",
                      "tables",
                      "tensorflow>=2.3.0"],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords=["chemical", "engineering", "machine", "learning", "deep", "networks", "neural", "keras"]
)
