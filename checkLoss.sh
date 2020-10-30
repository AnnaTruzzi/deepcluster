#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH -J checkLoss
#SBATCH --output=/home/annatruzzi/deepcluster/logs/slurm-%j.out
#SBATCH --error=/home/annatruzzi/deepcluster/logs/slurm-%j.err


DATA="/data/ILSVRC2012"
PYTHON="/opt/anaconda3/envs/dc_p27/bin/python"

for (( i=26; i<50; i+=2 )); do
    MODEL="/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_3/checkpoint_dc3_epoch${i}.pth.tar"
    EXP="/home/annatruzzi/deepcluster_eval/dc_3/checkpoint${i}"
    mkdir -p ${EXP}
    echo ${EXP}
    ${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --epochs 20 --conv 5 --lr 0.01 \
     --wd -7 --tencrops --verbose --exp ${EXP} --workers 24
done
