#!/bin/bash
#
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH -J train_dc_fromstartingcode
#SBATCH --output=/home/annatruzzi/deepcluster/logs/slurm-%j.out
#SBATCH --error=/home/annatruzzi/deepcluster/logs/slurm-%j.err

DIR="/data/ILSVRC2012/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
K=10
WORKERS=12
PYTHON="/opt/anaconda3/envs/dc_p27/bin/python"
CHECKPOINTS=5005
#RESUME="/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_2/checkpoint_dc2_epoch388.pth.tar"
EPOCHS=50
EXP="/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_2"
SEED=42

mkdir -p ${EXP}
${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
   --lr ${LR} --wd ${WD} --k ${K} --verbose --workers ${WORKERS}\
   --checkpoints ${CHECKPOINTS} --epochs ${EPOCHS} --seed ${SEED}

