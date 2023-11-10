#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00

#Example: ./ChemEngSim/bash_scripts/postprocessing_without_labels.sh my_custom_config.yml

eval "$(conda shell.bash hook)"
conda activate ChemEngSim

echo "Running in $SLURM_SUBMIT_DIR"
echo "Python: $(python --version)"

config="$SLURM_SUBMIT_DIR/$1"

package=$(grep -o '^[^#]*' "$config" | grep "package_location:" | grep -o '".*"' | sed 's/"//g')
log=$(grep -o '^[^#]*' "$config" | grep "log_dir:" | grep -o '".*"' | sed 's/"//g')
files=$(grep -o '^[^#]*' "$config" | grep "base_dir:" | grep -o '".*"' | sed 's/"//g')
package="$SLURM_SUBMIT_DIR/$package"
log="$SLURM_SUBMIT_DIR/$log"
files="$SLURM_SUBMIT_DIR/$files"

echo "package location: $package"
echo "log location: $log"
echo "file location: $files"

python "$package/python_scripts/postprocessing_without_labels.py" "$files" > "$log/postprocessing_without_labels.out"
