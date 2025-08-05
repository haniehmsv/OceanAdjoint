#PBS -S /bin/csh
#PBS -q gpu_normal
#PBS -l select=2:ncpus=36:ngpus=2:mem=240GB:model=mil_a100
#PBS -l place=scatter:excl
#PBS -l walltime=23:00:00
#PBS -j oe 
#PBS -koed     
#PBS -o /nobackup/smousav2/adjoint_learning/SSH_only_parallel/Logs/ssh_only_all_data_all_pair.log
#PBS -m bea
#PBS -N ssh_only
#PBS -r n

# Set env variables and load modules
limit stacksize unlimited
module purge
module use -a /swbuild/analytix/tools/modulefiles
module load miniconda3/v4

# Activate your environment (edit if different)
# setenv LD_LIBRARY_PATH /nobackup/smousav2/.conda/envs/samudra/lib:$LD_LIBRARY_PATH
setenv PYTHONUNBUFFERED 1
set CONDA_PYTHON=/nobackup/smousav2/.conda/envs/samudra/bin/python

# Get node information
setenv NODES "($(cat $PBS_NODEFILE | sort | uniq))"
setenv NUM_NODES ${#NODES[@]}

# Run PyTorch Distributed Job
cd /nobackup/smousav2/adjoint_learning/SSH_only_parallel
echo "Running on master node: `hostname`"
${CONDA_PYTHON} -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=$NUM_NODES \
    --rdzv-id=$JOB_ID \
    --rdzv-backend=c10d \
    ssh_only_all_data_all_pair.py
echo "Done."