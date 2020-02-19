#!/bin/bash
#SBATCH --job-name=convexity_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time 12:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=bcooley@princeton.edu

module purge
module load anaconda3
conda activate python37

srun python code/convexity_test.py hpc mini/
