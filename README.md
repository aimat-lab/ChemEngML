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


## Shap analysis
Regarding SHAP library:
https://github.com/shap/shap
https://shap-lrjball.readthedocs.io/en/latest/generated/shap.DeepExplainer.html#shap.DeepExplainer.explain_row


Regarding the codes:
main.py: contains the main code and you can run it by typing "python3 main.py"


Two things you have to manuallly modify to run the code:
1) Input the "model_base_path" in mlModel.py: location of the ML model
2) Input the location of the "X_train.npy" in hM_shap.py: location of the dataset used for training the ML model


## ChemEngSim (simulations)
When ChemEngSim is located in `./`:
- Create a folder for the channel structures e.g. with `mkdir channels_1`
- Create a folder for logging e.g. with `mkdir log`
- Copy the `config.yml` to your submit directory and set your parameters, especially the relative paths to the above
created directories, as well as to the ChemEngSim package folder. (`channels_i`, `log`, `ChemEngSim`)
 
The general workflow is structured with the scripts in `./ChemEngSim/bash_scripts`. 

#### 1. Create 2D channel geometries
```bash
./ChemEngSim/bash_scripts/create_channels.sh config.yml
```
This will write folder with increasing integers to the `base_dir` defined in th config. Each folder contains the
2D-channel geometry as hdf5 file and as binary for the fortran code (`ibm.bin`), as well as a png of the structure.

#### 2. Start a watcher job for early stopping of the calculations
```bash
./ChemEngSim/bash_scripts/observe_folder.sh config.yml
```
This will observe the `base_dir` and check whether a calculation is running on any of those structures. If this is the
case, it will check the `history.out` file regularly and stop the calculation if the defined level of convergence is 
reached by writing a `stop.now` file to the respective directory. 

####  3. Directly after starting the watcher, submit individual job batches
```bash
./ChemEngSim/bash_scripts/submit_simson_batch.sh config.yml <n> <step>
```
The first number defines the number of the simulation batch and the second number the step size. In turn `0 100` would
calculate channels 0 to 99, `1 100` would calculate channels 100-199, and so on. Depending on the number of generated
channels you have to submit e.g. 30 of those jobs. Make sure that they can finish within 71h and 30min given your
defined time limit in the config. E.g. if it is 30 min. your step size should be 143 (71*2+1).  
If you can not submit enough jobs to calculate all structures at once, simply start a watcher again after the other jobs
finished and submit the remaining jobs. Before you do so, it might be nice to copy the log file of the first watcher
from `./log/early_stopping_channels_1.csv` to a different location or simply rename it so it is not overwritten.
In that way all calculation times are still stored and you can merge the files later on. 

#### 4. Collect the output data
```bash
./ChemEngSim/bash_scripts/postprocess.sh config.yml
```
This will collect all 2D channel geometries into one `features.h5` and calculate Stanton numbers (St) and
drag coefficients (Cf) into `results.h5` and `results.csv`, both located in your `base_dir`.
