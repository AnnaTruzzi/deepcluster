import argparse
import os
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

import clustering
import models
from util import AverageMeter, Logger, UnifLabelSampler

parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                    choices=['alexnet', 'vgg16'], default='alexnet',
                    help='CNN architecture (default: alexnet)')
parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster', '--k', type=int, default=10000,
                    help='number of cluster for k-means (default: 10000)')
parser.add_argument('--lr', default=0.05, type=float,
                    help='learning rate (default: 0.05)')
parser.add_argument('--wd', default=-5, type=float,
                    help='weight decay pow (default: -5)')
parser.add_argument('--reassign', type=float, default=1.,
                    help="""how many epochs of training between two consecutive
                    reassignments of clusters (default: 1)""")
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--batch', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: None)')
parser.add_argument('--checkpoints', type=int, default=25000,
                    help='how many iterations between two checkpoints (default: 25000)')
parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
parser.add_argument('--exp', type=str, default='', help='path to exp folder')
parser.add_argument('--verbose', action='store_true', help='chatty')

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def _store_feats(layer, inp, output):
    """An ugly but effective way of accessing intermediate model features
    """
    _model_feats = []
    _model_feats.append(np.reshape(output, (len(output), -1)).numpy())


def save_tensor(self, input, output):
        layer_list.append(input)

def compute_features(dataloader, model, N):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    act = {}
    for i, input_tensor in enumerate(dataloader):
        input_var, label = torch.autograd.Variable(input_tensor[0].cuda()),input_tensor[2]
        aux = model(input_var).data.cpu().numpy()

        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            features[i * args.batch:] = aux

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
        layer_act = {}
        for i,layer in enumerate(model.features):
            layer_act[i] = layer.register_forward_hook(_store_feats)
        act[label[0]] = layer_act
    return act


def get_activations(offset):
    with torch.no_grad():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        tra = [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]
#        dataset = datasets.ImageFolder(offset, transform=transforms.Compose(tra))
        dataset = ImageFolderWithPaths(offset, transform=transforms.Compose(tra))
        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=1,
                                                num_workers=args.workers,
                                                pin_memory=True)
        features = compute_features(dataloader, model, len(dataset))
       
        #######  EMBED THIS CODE INTO THE PRETRAINED MODEL (code from eval_linear.py, line 196)
        #for m in model.features.modules():
        #    if not isinstance(m, nn.Sequential):
        #        x = m(x)
        #    if isinstance(m, nn.ReLU):
        #        if count == conv:
        #            return x
        #    count = count + 1

        return features


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    modelpth = '/home/CUSACKLAB/annatruzzi/deepcluster_models/alexnet/'
    checkpoint = torch.load(modelpth+'checkpoint.pth.tar')['state_dict']
    checkpoint_new = OrderedDict()
    for k, v in checkpoint.items():
        name = k.replace(".module", '') # remove 'module.' of dataparallel
        checkpoint_new[name]=v

    model = models.alexnet(sobel=True, bn=True, out=10000) 
    model.load_state_dict(checkpoint_new)
    model.cuda()
    #layers = ['ConvNdBackward1','ConvNdBackward2','ConvNdBackward3','ConvNdBackward4','ConvNdBackward5','ConvNdBackward9','ConvNdBackward13','ConvNdBackward16','ConvNdBackward19']
    image_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/stimuli/'  
    act = get_activations(image_pth)

    with open('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_activations.pickle', 'wb') as handle:
        pickle.dump(act, handle, protocol=pickle.HIGHEST_PROTOCOL)




for i, input_tensor in enumerate(dataloader):
    print input_tensor