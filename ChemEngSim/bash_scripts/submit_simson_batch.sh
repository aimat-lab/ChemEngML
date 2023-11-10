#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=71:30:00

# Example to submit the first 100 channels in folder grids:
# ./ChemEngSim/bash_scripts/submit_simson_batch.sh my_custom_config.yml 0 100

module load compiler/intel/19.1
module load mpi/openmpi/4.0

config="$SLURM_SUBMIT_DIR/$1"

package=$(grep -o '^[^#]*' "$config" | grep "package_location:" | grep -o '".*"' | sed 's/"//g')
log=$(grep -o '^[^#]*' "$config" | grep "log_dir:" | grep -o '".*"' | sed 's/"//g')
files=$(grep -o '^[^#]*' "$config" | grep "sample_dir:" | grep -o '".*"' | sed 's/"//g')
package="$SLURM_SUBMIT_DIR/$package"
log="$SLURM_SUBMIT_DIR/$log"
files="$SLURM_SUBMIT_DIR/$files"

zfill=$(grep -o '^[^#]*' "$config" | grep "zfill:" | grep -Eo '[0-9]{1,}')

{
  echo "Running in $SLURM_SUBMIT_DIR"
  echo "MPI: $(mpirun --version)"
  echo "IFORT: $(ifort --version)"
  echo "Working on: $files"
  echo "Logging to: $log"
  echo "N: $2"
  echo "Step: $3"

  startidx=$(($2*$3))
  endidx=$(($2*$3+$3))
  count=0

  cd $files || return
  for f in */
  do
    echo "$count"
    if [ $count -ge $startidx ] && [ $count -lt $endidx ]
    then
      echo "yay"
      cd "$files/$f" || return
      echo "$f"
      mpirun --bind-to core --map-by core -report-bindings ./bla
    fi
    count=$((count+1))
  done
} > "$log/simson_job_$2_$3.out"
