#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=18
#SBATCH -J checkLoss
#SBATCH --output=/home/annatruzzi/deepcluster/logs/slurm-%j.out
#SBATCH --error=/home/annatruzzi/deepcluster/logs/slurm-%j.err


DATA="/data/ILSVRC2012"
PYTHON="/opt/anaconda3/envs/dc_p27/bin/python"

for (( i=0; i<50; i+=2 )); do
    MODEL="/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_2/checkpoint_dc2_epoch${i}.pth.tar"
    EXP="/home/annatruzzi/deepcluster_eval/dc_2/checkpoint${i}"
    mkdir -p ${EXP}
    echo ${EXP}
    ${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --epochs 20 --conv 5 --lr 0.01 \
     --wd -7 --tencrops --verbose --exp ${EXP} --workers 18 
done
