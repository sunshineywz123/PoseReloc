#!/bin/bash -l

#SBATCH --job-name=gats_loftr_fix_backbone
#SBATCH --partition=3d_share

#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0

#SBATCH --error=/mnt/lustre/hexingyi/logs/%j.err
#SBATCH --output=/mnt/lustre/hexingyi/logs/%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=hexingyi@sensetime.com

PROJECT_DIR=/mnt/lustre/hexingyi/code/PoseReloc

n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=8
task_name='loftr_train'

ulimit -u 1048576  # IMPORTANT
conda activate 
export PATH=/mnt/cache/hexingyi/miniconda3/envs/posereloc/bin:$PATH
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1


srun python -u ./train_gats_loftr.py +experiment=train_GATs_loftr_1988 trainer.gpus=$n_gpus_per_node trainer.num_nodes=$n_nodes task_name=$task_name