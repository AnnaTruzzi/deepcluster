import argparse
import os
import pickle
import time

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

import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler

modelpth = '/home/CUSACKLAB/annatruzzi/deepcluster_models/alexnet'
model = torch.load(modelpth+'model.caffemodel')

#The models in Caffe format expect BGR inputs that range in [0, 255]. 
# You do not need to subtract the per-color-channel mean image since 
# the preprocessing of the data is already included in our released models.
dataloader = torch.utils.data.DataLoader(dataset,
                            batch_size=args.batch,
                            num_workers=args.workers,
                            pin_memory=True)



if __name__ == "__main__":
    # change with deepcllustering layers
    for layer in ['ConvNdBackward5','ConvNdBackward9','ConvNdBackward13','ConvNdBackward16','ConvNdBackward19']:
    attribute_and_visualise(layer=layer)


