#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=72:00:00

# Usage: ./ChemEngSim/bash_scripts/create_channels.sh my_custom_config.yml

eval "$(conda shell.bash hook)"
conda activate ChemEngSim

echo "Running in $SLURM_SUBMIT_DIR"
echo "Python: $(python --version)"

config="$SLURM_SUBMIT_DIR/$1"

package=$(grep -o '^[^#]*' "$config" | grep "package_location:" | grep -o '".*"' | sed 's/"//g')
package="$SLURM_SUBMIT_DIR/$package"

python "$package/python_scripts/generator.py" "$config"
