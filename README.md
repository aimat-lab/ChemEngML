# CNNs to predict Drag coefficient (Cf) and Stanton number (St)
Clone this repository and install it in editable mode into a conda environment of your choice using:
```bash
pip install -e ChemEngML/
```

To reproduce the learning curve run:
```bash
python ./ChemEngML/scripts/leaning_curve.py /path/to/features.h5 /path/to/labels.h5
```

Features and labels currently have to be requested from the authors but will be provided in an additional resource later.