#!/bin/bash
#SBATCH --job-name=tpsp-results
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time 96:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=bcooley@princeton.edu

module purge
module load anaconda3
conda activate python37

srun doit results:results --size large/
