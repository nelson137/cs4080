#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition hpc3
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time 0-00:05:00
#SBATCH --job-name=example_mpi_job
#SBATCH --output=results-mpi-%j.out
#-------------------------------------------------------------------------------

# Load your modules here:
module load openmpi/openmpi-3.1.4
module list

# Science goes here:
srun ./word_counter genesis.txt


