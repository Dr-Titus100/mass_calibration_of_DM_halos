#!/bin/bash

# Params file
files=(
"/bsuhome/tnde/Lensing/codes/notebooks/cardinal/py/cardinal.py"
)

# file_names = ("_test_run0")

for file_name in "${file_names[@]}"; do
    for params_file in "${files[@]}"; do
        # Generate a unique job name based on depth and pec_vel_option
        file_id=$(basename "$file")
        file_id="${file_id%.py}"  # Remove the .py extension
        job_name="${file_id}_MCMC"

        partition="bsudfq"
        nnodes=2
        cpus_per_task=1
        tasks_per_node=48
        nthreads=0

        # Create a temporary SLURM script
        slurm_script=$(mktemp)
        cat > "$slurm_script" << EOL
#!/bin/bash
#SBATCH -J $job_name               # job name
#SBATCH -o "slurm_outputs/${job_name}_%j.out"    # output and error file name (%j expands to jobID)
#SBATCH -N $nnodes                        # number of nodes you want to run on
#SBATCH -n $tasks_per_node                        # total number of tasks requested
#SBATCH --cpus-per-task=$cpus_per_task         # number of CPU cores per task (within each node)
#SBATCH --mail-type=All
#SBATCH --mail-user=titusnyarkonde@u.boisestate.edu 
#SBATCH -p $partition                   # queue (partition) for R2 use defq
#SBATCH -t 06-23:59:59              # run time (hh:mm:ss)

# Activate the conda environment
module load slurm
module load gsl
module load cuda11.0/toolkit/11.0.3
source /bsuhome/tnde/miniforge3/etc/profile.d/mamba.sh
conda activate /bsuhome/tnde/.conda/envs/mass_cal

# Run the program
mpirun -np $tasks_per_node python "$file" --run_name "$file_name"

EOL
    # Submit the SLURM job
    sbatch "$slurm_script"

    # Remove the temporary SLURM script
    rm "$slurm_script"
#                     break 4
done


