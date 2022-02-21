#!/bin/bash -l

#SBATCH --job-name=gats_loftr_nofb_warp
#SBATCH --partition=3d_share

#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=8
#SBATCH --mem=0

#SBATCH --error=/mnt/lustrenew/hexingyi/logs/%j.err
#SBATCH --output=/mnt/lustrenew/hexingyi/logs/%j.out
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=hexingyi@sensetime.com

PROJECT_DIR=/mnt/cache/hexingyi/code/PoseReloc

n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=8
job_name='gats_loftr_nofb_warp'
task_name='loftr_train'

# ulimit -u 1048576  # IMPORTANT
# conda activate 
export PATH=/mnt/cache/hexingyi/miniconda3/envs/posereloc/bin:$PATH
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1


srun --partition=3d_share --nodes=$n_nodes --gres=gpu:$n_gpus_per_node --ntasks-per-node=8 --mem=0 --job-name=$job_name python -u ./train_gats_loftr.py +experiment=train_GATs_loftr_1988_fine_w5_7000_tmp_0.08_use_coarse_nofb_no_fineatt_pe_linear_gats_loftr_warp trainer.gpus=$n_gpus_per_node trainer.num_nodes=$n_nodes task_name=$task_name