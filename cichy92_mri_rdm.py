import os
import scipy.io
import pickle
import numpy as np
from itertools import compress
from scipy.spatial import distance
from sklearn.manifold import MDS
from statsmodels.stats.anova import AnovaRM
from matplotlib import pyplot as plt
import re
from scipy.cluster.hierarchy import dendrogram, linkage
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from itertools import combinations
import scipy.io
import h5py
import hdf5storage
from scipy import stats
from scipy.spatial.distance import squareform
import collections
from gensim.models import KeyedVectors

def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}


def rdm_plot(rdm, vmin, vmax, main, outname, labels = None):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(rdm,vmin=vmin,vmax=vmax)
    plt.colorbar()
    ticks = np.arange(0,118,1)
    if labels:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels,rotation = 90, fontsize = 8)
        ax.set_yticklabels(labels,fontsize = 8)
    fig.suptitle(main)
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()
    plt.show()
    fig.savefig(outname)
    plt.close(fig)


if __name__ == '__main__':
    fmri_path = '/home/CUSACKLAB/annatruzzi/deepcluster/cichy2016/algonautsChallenge2019/Training_Data/92_Image_Set/target_fmri.mat'
    
    ###### load fmri rdms and re-order them by lch similarity
    fmri_mat = loadmat(fmri_path)
    EVC = np.mean(fmri_mat['EVC_RDMs'],axis = 0)
    IT = np.mean(fmri_mat['IT_RDMs'], axis = 0)

    model = KeyedVectors.load_word2vec_format('/home/CUSACKLAB/annatruzzi/deepcluster/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

    rdm_plot(EVC, vmin = 0, vmax = 0.7, main = 'EVC', outname = 'rdm_EVC_92images.png')
    rdm_plot(IT, vmin = 0, vmax = 0.7, main = 'IT', outname = 'rdm_IT_92images.png')
    