#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

# Start this script to observe a folder with generated channels
# It will check whether calculations in this folder are running and stop them when they converged or reached the
# maximum time limit.
# Call this script from any directory and pass the path to the ChemEngSim package from the local dir.
# Syntax: script.sh path/to/ChemEngSim
# Example: ./ChemEngSim/bash_scripts/observe_folder.sh my_custom_config.yml

eval "$(conda shell.bash hook)"
conda activate ChemEngSim

echo "Running in $SLURM_SUBMIT_DIR"
echo "Python: $(python --version)"

config="$SLURM_SUBMIT_DIR/$1"

package=$(grep -o '^[^#]*' "$config" | grep "package_location:" | grep -o '".*"' | sed 's/"//g')
package="$SLURM_SUBMIT_DIR/$package"

python "$package/python_scripts/early_stopping.py" "$config"
