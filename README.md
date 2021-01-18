# CNNs to predict Drag coefficient (Cf) and Stanton number (St)

## Publication

Preprint: https://arxiv.org/todo

**Machine Learning for rapid discovery of laminar flow channel wall modifications that enhance heat transfer**

*Alexander Stroh, Matthias Schniewind, Pascal Friederich, Bradley P. Ladewig*


## Installation

Clone this repository and install it in editable mode into a conda environment of your choice using:
```bash
pip install -e ChemEngML/
```

## Usage

To reproduce the learning curve run:
```bash
python ./ChemEngML/scripts/leaning_curve.py /path/to/features.h5 /path/to/labels.h5
```

Features and labels currently have to be requested from the authors but will be provided in an additional resource later.
