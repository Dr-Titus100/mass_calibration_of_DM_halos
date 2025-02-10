#!/bin/bash
#SBATCH -J cylinder_richness               # job name
#SBATCH -o "slurm_outputs/cylinder_richness_%j.out"    # output and error file name (%j expands to jobID)
#SBATCH -N 1                        # number of nodes you want to run on
#SBATCH -n 1                       # total number of tasks requested
#SBATCH --cpus-per-task=48         # number of CPU cores per task (within each node)
#SBATCH --mail-type=All
#SBATCH --mail-user=titusnyarkonde@u.boisestate.edu 
#SBATCH -p bsudfq                   # queue (partition) for R2 use defq
#SBATCH -t 06-23:59:59              # run time (hh:mm:ss)
module load slurm
module load gcc/7.5.0
module load gsl/gcc8/2.6
module load openmpi/gcc/64/1.10.7
module load cuda11.0/toolkit/11.0.3 
source /bsuhome/tnde/miniconda3/etc/profile.d/conda.sh
conda activate /bsuhome/tnde/.conda/envs/mass_cal
# conda activate mass_cal2
# Echo commands to stdout (standard output).
set -x
# Copy your code & data to your R2 Home directory using
# the SFTP (secure file transfer protocol).
# Go to the directory where the actual BATCH file is present.
cd /bsuhome/tnde/Lensing/codes/notebooks/mini_uchuu
# The �python� command runs your python code.
# All output is dumped into titus.o%j with �%j� replaced by the Job ID.
## The file Multiprocessing.py must also 
## be in $/home/tnde/P1_Density_Calibration/Density3D
python make_gal_cat_Heidi_fast_final.py mini_uchuu_fid_hod.yml >>log.out