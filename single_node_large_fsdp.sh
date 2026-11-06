#!/bin/bash

#SBATCH --job-name=fsdp-single-node-llm-example
#SBATCH --partition=a100
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=76000M

WORKING_DIRECTORY=$SLURM_SUBMIT_DIR
VENV_NAME=$WORKING_DIRECTORY/venv
DATA_DIR=$WORKING_DIRECTORY/data
CHECKPOINTS_DIR=$WORKING_DIRECTORY/checkpoints
LOG_DIR=$WORKING_DIRECTORY/logs

# Optional: print useful job info
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "SLURM job ID: $SLURM_JOB_ID"

# Load required modules
module purge
module load python
module load cuda
module load openmpi

# Activate Python virtual environment
source $VENV_NAME/bin/activate

# Set any environment variables or configuration options
export PYTHONUNBUFFERED=1

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Head node IP: $head_node_ip"

# Move to job directory
cd $SLURM_SUBMIT_DIR

# Run the Python script
accelerate launch --use-fsdp --fsdp_auto_wrap_policy=SIZE_BASED_WRAP --num_processes 2 --num_machines 1 --main_process_ip $head_node_ip --main_process_port 29500 llm_fine_tuning.py --num-epochs=2 --model=Qwen/Qwen3-8B

# deactivate virtual environment
deactivate
