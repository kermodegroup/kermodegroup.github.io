#!/bin/sh
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700
#SBATCH --partition=devel


# Avon
module purge
module load GCC/10.3.0 OpenMPI/4.1.1 ScaLAPACK/2.1.0-fb OpenBLAS/0.3.15 GCCcore/11.3.0 Python/3.10.4

export PMIX_MCA_gds=hash

ulimit -s unlimited
ulimit -v unlimited

mkdir trained_gap

srun gap_fit config_file=gap_config