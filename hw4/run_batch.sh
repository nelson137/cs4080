#!/bin/bash
#-------------------------------------------------------------------------------
# SBATCH CONFIG
#-------------------------------------------------------------------------------
#SBATCH --partition hpc3
#SBATCH --nodes=3
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time 0-00:05:00
#SBATCH --job-name=hw4
#SBATCH --output=results-%j.out
#-------------------------------------------------------------------------------

module load opencv/opencv-4.2.0-openmpi-3.1.3-openblas openmpi/openmpi-3.1.4
module list

srun orterun -np 4 ./homework4 ../../Astronaught.png 4 1024 ./out.png
