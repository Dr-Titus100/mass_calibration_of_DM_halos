#!/bin/bash
#Account and Email Information
#SBATCH -A tnde  ## User ID
#SBATCH --mail-type=end
#SBATCH --mail-user=titusnyarkonde@u.boisestate.edu
# Specify parition (queue)
#SBATCH --partition=bsudfq
# Join output and errors into output.
#SBATCH -o test_borah.o%j
#SBATCH -e test_borah.e%j
# Specify job not to be rerunable.
#SBATCH --no-requeue
# Job Name.
#SBATCH --job-name="test_borah_login2"
# Specify walltime.
#SBATCH -t 06-23:59:59 
# ###SBATCH --time=48:00:00
# Specify number of requested nodes.
#SBATCH -N 1
# Specify the total number of requested procs:
#SBATCH -n 48
# number of cpus per task
#SBATCH --cpus-per-task=1 
# Number of GPUs per node.
# #SBATCH --gres=gpu:1
# load all necessary modules and ctivate the conda environment
module load slurm
module load gcc/7.5.0
module load gsl/gcc8/2.6
module load openmpi/gcc/64/1.10.7
module load cuda11.0/toolkit/11.0.3 
source /bsuhome/tnde/miniconda3/etc/profile.d/conda.sh
conda activate /bsuhome/tnde/.conda/envs/mass_cal
# Echo commands to stdout (standard output).
# set -x
# Copy your code & data to your R2 Home directory using
# the SFTP (secure file transfer protocol).
# Go to the directory where the actual BATCH file is present.
cd /bsuhome/tnde/Lensing/codes/py/test_borah_login2/GraduateShowcase2024/codes
 # The �python� command runs your python code.
# All output is dumped into test_borah.o%j with �%j� replaced by the Job ID.
## The file Multiprocessing.py must also 
## be in $/home/tnde/P1_Density_Calibration/Density3D
mpirun -np 8 python3 test_borah_login2.py >>log.out
# python3 demo_analytic.py >>log.out


