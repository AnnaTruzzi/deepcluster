#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J train_dc_nobuggedcodeincomputefeatures
#SBATCH --output=/home/annatruzzi/deepcluster/logs/slurm-%j.out
#SBATCH --error=/home/annatruzzi/deepcluster/logs/slurm-%j.err

DIR="/data/ILSVRC2012/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
WORKERS=12
PYTHON="/opt/anaconda3/envs/dc_p27/bin/python"
CHECKPOINTS=5005
#RESUME="/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_3/checkpoint_dc3_epoch8.pth.tar"
EPOCHS=50
#SEED=42
i=3
EXP="/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_$i"
mkdir -p ${EXP}

${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
 --lr ${LR} --wd ${WD} --k ${K} --verbose --workers ${WORKERS}\
 --instantiation ${i} --checkpoints ${CHECKPOINTS}\
 --epochs ${EPOCHS} --sobel
echo "Started training for instantiation number $i"

