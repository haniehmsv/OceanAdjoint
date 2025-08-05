#!/bin/bash
#SBATCH --job-name=ssh_only
#SBATCH --output=Logs/ssh_only_all_data_all_pair.out
#SBATCH --error=Logs/ssh_only_all_data_all_pair.err

#SBATCH --partition=gpuA100x4            # change if needed
#SBATCH --mem=210G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # torchrun spawns 1 process per GPU
#SBATCH --cpus-per-task=32          # For data loading / preprocessing
#SBATCH --gpus-per-node=4
#SBATCH --time=20:00:00                 # change as needed
#SBATCH --gpu-bind=closest          # select a cpu close to gpu on pci bus topology
#SBATCH --constraint="scratch"
#SBATCH --account=bezs-delta-gpu     # <- set this based on `accounts` command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue

# Optional email notifications
#SBATCH --mail-user=mousavih@jpl.nasa.gov
#SBATCH --mail-type=BEGIN,END,FAIL

# Optional but helpful
export OMP_NUM_THREADS=8        # if code is not multithreaded, otherwise set to 8 or 16 # Match cpus-per-task if multithreaded

# Load GPU-compatible Conda
module load anaconda3_gpu
# module load nccl
module list
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn   # Slingshot interconnect
export LOGLEVEL=INFO
export PYTHONNOUSERSITE=1
export PATH=/u/hmousavi/.conda/envs/samudra/bin:$PATH

# Activate your environment on terminal before submitting the batch script as follows: (do not uncomment the next lines)
# module load reset
# module load anaconda3_gpu
# export PATH=/u/hmousavi/.conda/envs/samudra/bin:$PATH
# export PYTHONNOUSERSITE=1
# conda activate samudra
# sbatch job_gpu.slurm

# Show loaded modules and environment for logging/debugging
echo "Running on $(hostname)"
nvidia-smi
module list
conda list | grep torch

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}

if [ ${SLURM_NNODES} -gt 1 ]; then
    # Multi-node: get head node IP
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname --ip-address)
else
    # Single-node: use localhost
    head_node_ip="127.0.0.1"
fi

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

# Pick a random rendezvous ID for torchrun
rdzv_id=$RANDOM
port=$((29500 + RANDOM % 1000))

# Run your Python code
echo "Job is starting on $(hostname) with ${SLURM_GPUS_PER_NODE} GPUs per node and ${SLURM_NNODES} node(s)"

srun torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=${SLURM_GPUS_PER_NODE} \
    --rdzv_id=$rdzv_id \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${head_node_ip}:${port} \
    ssh_only_all_data_all_pair.py

echo "Done."
