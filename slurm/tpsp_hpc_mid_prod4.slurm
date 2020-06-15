#!/bin/bash
#SBATCH --job-name=tpsp-results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time 120:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=bcooley@princeton.edu

module purge
module load anaconda3
conda activate python37

srun doit results:results --size mid/ --Mstart 76 --Mend 100
