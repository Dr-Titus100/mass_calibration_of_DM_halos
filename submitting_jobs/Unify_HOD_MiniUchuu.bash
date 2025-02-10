#!/bin/bash

# Params file
params_files=(
"/bsuhome/shuleicao/pythonscripts/Unify_HOD_pipeline/python_scripts/yml/mini_uchuu/mini_uchuu_fid_hod.yml"
)

# Depths to use
depths=(
30.0
60.0
90.0
120.0
)

pec_vel_options=(
""
"--pec_vel"
)

new_options=(
""
# "--new"
)
# masses=(1e11 3e13 4e13 5e13)
# masses=(2e10 1e11)

# zbins=("zbin1" "zbin2")
zbins=("zbin1")
# zbins=("zbin1" "zbin2" "zbin3")

# Iterate over each depth and pec_vel_option
for new_option in "${new_options[@]}"; do
    for zbin in "${zbins[@]}"; do
        for params_file in "${params_files[@]}"; do
            for depth in "${depths[@]}"; do
                for pec_vel_option in "${pec_vel_options[@]}"; do
                    # Generate a unique job name based on depth and pec_vel_option
                    file_id=$(basename "$params_file")
                    file_id="${file_id%.yml}"  # Remove the .yml extension
                    job_name="${file_id}_MCMC"
            #         job_name="${file_id}_depth${depth}"
                    if [ -n "$pec_vel_option" ]; then
                        job_name="${job_name}_pecvel"
                    fi
                    if [ -n "$new_option" ]; then
                        job_name="${job_name}_new"
                    fi
                    if [ "$zbin" == "zbin1" ] && [ "$depth" == "30.0" ] && [ -z "$pec_vel_option" ]; then
                        partition="bigmem"
                        nnodes=1
                        cpus_per_task=4
                        tasks_per_node=12
                        nthreads=8
                    else
                        partition="bsudfq"
                        nnodes=2
                        cpus_per_task=4
                        tasks_per_node=24
                        nthreads=8
                    fi
                    # Create a temporary SLURM script
                    slurm_script=$(mktemp)
                    cat > "$slurm_script" << EOL
#!/bin/bash
#SBATCH -J $job_name               # job name
#SBATCH -o "slurm_outputs/${job_name}_%j.out"    # output and error file name (%j expands to jobID)
#SBATCH -N $nnodes                        # number of nodes you want to run on
#SBATCH -n $tasks_per_node                        # total number of tasks requested
#SBATCH --cpus-per-task=$cpus_per_task         # number of CPU cores per task (within each node)
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=shuleicao@boisestate.edu 
#SBATCH -p $partition                   # queue (partition) for R2 use defq
#SBATCH -t 06-23:59:59              # run time (hh:mm:ss)

# Activate the conda environment
module load slurm
module load gsl/gcc8/2.6
source /bsuhome/tnde/miniconda3/etc/profile.d/conda.sh
conda activate /bsuhome/shuleicao/.conda/envs/redmapper-env

# Run the program
mpirun -np $tasks_per_node python /bsuhome/shuleicao/pythonscripts/Unify_HOD_pipeline/python_scripts/emcee_likelihood_function_pipeline.py "$params_file" --zbin "$zbin" --depth "$depth" $new_option $pec_vel_option --nthreads $nthreads

EOL
                    # Submit the SLURM job
                    sbatch "$slurm_script"

                    # Remove the temporary SLURM script
                    rm "$slurm_script"
#                     break 4
                done
            done
        done
    done
done
