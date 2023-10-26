# Machine learning for rapid discovery of laminar flow channel wall modifications that enhance heat transfer


## Installation

Clone this repository and install it in editable mode into a conda environment of your choice using:
```bash
pip install -e ChemEngML/
```

## Usage

To run the file:
```bash
python3 train.py train_config.yaml --testing True
```


## Shap
Regarding SHAP library:
https://github.com/shap/shap
https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html#shap.DeepExplainer.explain_row


Regarding the codes:
main.py: contains the main code and you can run it by typing "python3 main.py"


Two things you have to manuallly modify to run the code:
1) Input the "model_base_path" in mlModel.py: location of the ML model
2) Input the location of the "X_train.npy" in hM_shap.py: location of the dataset used for training the ML model
