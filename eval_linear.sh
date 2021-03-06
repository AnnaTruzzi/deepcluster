#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=18
#SBATCH -J test_dc_1
#SBATCH --output=/home/annatruzzi/deepcluster/logs/slurm-%j.out
#SBATCH --error=/home/annatruzzi/deepcluster/logs/slurm-%j.err


DATA="/data/ILSVRC2012"
#MODELROOT="${HOME}/deepcluster/models"
MODEL="/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_1/checkpoint_dc1_epoch420.pth.tar"
#MODEL="/home/annatruzzi/deepcluster_models/alexnet/checkpoint_dc.pth.tar"
EXP="/home/annatruzzi/deepcluster_eval/original"

PYTHON="/opt/anaconda3/envs/dc_p27/bin/python"

mkdir -p ${EXP}

${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 5 --lr 0.01 \
  --wd -7 --tencrops --verbose --exp ${EXP} --workers 12
