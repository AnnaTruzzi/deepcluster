#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH -J checkLoss
#SBATCH --output=/home/annatruzzi/deepcluster/logs/slurm-%j.out
#SBATCH --error=/home/annatruzzi/deepcluster/logs/slurm-%j.err


DATA="/data/ILSVRC2012"
PYTHON="/opt/anaconda3/envs/dc_p27/bin/python"
#MODEL="/home/annatruzzi/deepcluster_models/alexnet/checkpoint_dc.pth.tar"

for (( i=0; i<421; i+=10 )); do
    MODEL="/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_1/checkpoint_dc1_epoch${i}.pth.tar"
    EXP="/home/annatruzzi/deepcluster_eval/dc_1/checkpoint${i}"
    mkdir -p ${EXP}
    echo ${EXP}
    ${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --epochs 10 --conv 5 --lr 0.01 \
     --wd -7 --tencrops --verbose --exp ${EXP} --workers 24 
done
