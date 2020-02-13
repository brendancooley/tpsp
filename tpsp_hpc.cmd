#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 2:00:00
#SBATCH –o log.%j
#SBATCH –mail-type=begin
#SBATCH –mail-type=end

module purge
module load anaconda3
conda activate python37

srun doit results
