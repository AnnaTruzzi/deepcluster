#!/bin/bash
#
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH -J train_multiple_dc
#SBATCH --output=/home/annatruzzi/deepcluster/logs/slurm-%j.out
#SBATCH --error=/home/annatruzzi/deepcluster/logs/slurm-%j.err

DIR="/data/ILSVRC2012/train"
ARCH="alexnet"
LR=0.05
WD=-5
K=10000
K=10
WORKERS=12
EXP="/home/annatruzzi/checkpoints/multiple_dc_instantiations/"
PYTHON="/opt/anaconda3/envs/dc_p27/bin/python"
CHECKPOINTS=5005

for i in {1..15}
do
   EXP= echo "/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_$i"
   mkdir -p ${EXP}
   CUDA_VISIBLE_DEVICES=0 ${PYTHON} main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
     --lr ${LR} --wd ${WD} --k ${K} --verbose --workers ${WORKERS}
   echo "Started training for instantiation number $i"
done
