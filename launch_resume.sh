#!/bin/bash
#
#SBATCH --time=48:00:00
#SBATCH --nodes=4
#SBATCH -A tclif038b
#SBATCH -p GpuQ
#SBATCH -J deepcluster_launch_resume
#SBATCH --output=/ichec/home/users/annatruzzi/deepcluster/logs/slurm-%j.out
#SBATCH --error=/ichec/home/users/annatruzzi/deepcluster/logs/slurm-%j.err

INSTANTIATION = $1
epoch = $2
DIR="/ichec/work/tclif038b/ILSVRC2012/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
WORKERS=4
PYTHON="/ichec/home/users/annatruzzi/anaconda3/envs/dc_p27/bin/python"
CHECKPOINTS=5005
EXP="/ichec/work/tclif038b/deepcluster_checkpoints/dc_${i}"
RESUME="/ichec/work/tclif038b/deepcluster_checkpoints/dc_${i}/checkpoint_dc${i}_epoch${epoch}.pth.tar"

${PYTHON} main_parallelcomputing.py ${DIR} --exp ${EXP} --arch ${ARCH} --resume ${RESUME} \
    --lr ${LR} --wd ${WD} --k ${K} --verbose \
    --instantiation ${INSTANTIATION} --checkpoints ${CHECKPOINTS}\
    --start_epoch ${epoch} --sobel --workers ${WORKERS}