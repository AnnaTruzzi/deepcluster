#!/bin/bash
#
#SBATCH --time=48:00:00
#SBATCH --nodes=4
#SBATCH -A tclif038b
#SBATCH -p GpuQ
#SBATCH -J deepcluster_training
#SBATCH --output=/ichec/home/users/annatruzzi/deepcluster/logs/slurm-%j.out
#SBATCH --error=/ichec/home/users/annatruzzi/deepcluster/logs/slurm-%j.err

PYTHON="/ichec/home/users/annatruzzi/anaconda3/envs/dc_p27/bin/python"

for i in {1..15}
do
  ${PYTHON} launch_dependencies.py -i ${i}
  echo "Started training for instantiation number $i"
done
