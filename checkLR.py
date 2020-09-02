import argparse
import os
import scipy.io
import pickle
import time
from collections import OrderedDict

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import datetime

def main():
    checkpoint = torch.load(checkpth+'checkpoint_dc1_epoch281.pth.tar')
    print(checkpoint['optimizer'])

if __name__ == '__main__':
#    global args
#    args = parser.parse_args()
    checkpth = '/home/annatruzzi/checkpoints/multiple_dc_instantiations/dc_1/'
    main()