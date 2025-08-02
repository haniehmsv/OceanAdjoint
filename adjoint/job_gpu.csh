#PBS -S /bin/csh
#PBS -q gpu_normal
#PBS -l select=1:ncpus=48:ngpus=4:mem=320GB:model=mil_a100
#PBS -l place=scatter:excl
#PBS -l walltime=12:00:00
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

# Navigate to your project directory
cd /nobackup/smousav2/adjoint_learning/SSH_only_parallel

# Run your test script
echo "Running PyTorch model on multiple GPU..."
# ${CONDA_PYTHON} -u ssh_only_small_data_one_pair.py
${CONDA_PYTHON} -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    ssh_only_all_data_all_pair.py
echo "Done."

# Deactivate Conda
#conda deactivate
