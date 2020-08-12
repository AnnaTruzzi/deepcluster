# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DATA="/data/ILSVRC2012"
#MODELROOT="${HOME}/deepcluster/models"
MODEL="/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_1/checkpoint_dc1_epoch199.pth.tar"
EXP="/home/annatruzzi/deepcluster_eval"

PYTHON="/opt/anaconda3/envs/dc_p27/bin/python"

mkdir -p ${EXP}

${PYTHON} eval_linear.py --model ${MODEL} --data ${DATA} --conv 5 --lr 0.01 \
  --wd -7 --tencrops --verbose --exp ${EXP} --workers 12
