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

## load activations dictionary 
def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file)
    return dict


def rdm(act_matrix):
    rdm_matrix = distance.squareform(distance.pdist(act_matrix,metric='correlation'))
    return rdm_matrix

def reorder_od(dict1,order):
   new_od = collections.OrderedDict([(k,None) for k in order if k in dict1])
   new_od.update(dict1)
   return new_od

def lch_order(keys,names_dict,syn_dict):
    ## get LCH distance for the images from the respective synsets and order them by hierarchical clustering + get respective (comprehensible) labels 
    labels_dict = [label.split('/')[-1] for label in keys]
    synsets_list = []
    for key in labels_dict:
        for item in syn_dict.items():
            if item[0] in key:
                syn = wn.synset(item[1])
                #print syn.definition()
                synsets_list.append(syn)

    cmb = list(combinations(synsets_list,2))
    lch_list = []
    x = []
    y = []
    for item in cmb:
        x.append(item[0])
        y.append(item[1])
        lch = item[0].lch_similarity(item[1])
        lch_list.append(lch)

    x = np.array(x,dtype = str)
    y = np.array(y, dtype = str)
    lch_list = np.array(lch_list,dtype = float)
    lch_matrix = np.stack((x,y,(1./lch_list)),axis = 1)
    Z = linkage(lch_matrix[:,2], 'ward')  #optimal_ordering = True

    labels_img = []
    for syn in synsets_list:
        for item in syn_dict.items():
            if item[1] == str(syn).split("'")[1]:
                labels_img.append(names_dict[item[0]])

    plt.figure()
    den = dendrogram(Z,
                orientation='top',
                labels=labels_img,
                leaf_font_size=9,
                distance_sort='ascending',
                show_leaf_counts=True)
    plt.show()
    orderedNames = den['ivl']
    return orderedNames


def numeric_order(keys,ordered_names_dict):
    orderedNames = []
    for item in sorted(ordered_names_dict.items()):
        orderedNames.append(item[1])
    return orderedNames


def sq(x):
    return squareform(x, force='tovector', checks=False)

#defines the spearman correlation
def spearman(model_rdm, rdms):
    model_rdm_sq = sq(model_rdm)
    return [stats.spearmanr(rdm, model_rdm)[0] for rdm in rdms]


#computes spearman correlation (R) and R^2, and ttest for p-value.
def fmri_rdm(model_rdm, fmri_rdms):
    corr = spearman(model_rdm, fmri_rdms)
    corr_squared = np.square(corr)
    return np.mean(corr_squared), stats.ttest_1samp(corr_squared, 0)[1]


def evaluate(submission, targets, target_names=['EVC_RDMs', 'IT_RDMs']):
    out = {name: fmri_rdm(submission, targets[name]) for name in target_names}
    out['score'] = np.mean([x[0] for x in out.values()])
    return out


def rdm_plot(rdm, vmin, vmax, labels, main, outname):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(rdm,vmin=vmin,vmax=vmax)
    plt.colorbar()
    ticks = np.arange(0,118,1)
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


def main(layers,act,img_names,img_synsets):
    act = load_dict(act)
    img_names = load_dict(img_names)
    img_names_ordered = reorder_od(img_names, sorted(img_names.keys()))
    img_synsets = load_dict(img_synsets)
    img_synsets_ordered = reorder_od(img_synsets, sorted(img_synsets.keys()))

    #############   get list of ordered activation keys starting from labels_img and img_names dict (choose between lch distance or numeric ordering)
#    orderedNames = lch_order(act.keys(),img_names,img_synsets)
    orderedNames = numeric_order(act.keys(),img_names_ordered)
    orderedKeys = []
    for name in orderedNames:
        number = [i[0] for i in img_names_ordered.items() if i[1] == name]
        key = [k for k in act.keys() if '%s.jpg' %number[0] in k]
        orderedKeys.append(key[0])

    ############ order activations dictionary and reorganize it in order to get one dictionary per layer and image (instead of one dictionary per image)
    ordered_act = reorder_od(act,orderedKeys)
    
    layer_dict = collections.OrderedDict()
    for layer in range(0,len(layers)):
        layer_dict[layer] = collections.OrderedDict()
        for item in ordered_act.items():
            layer_dict[layer][item[0]] = np.mean(np.mean(np.squeeze(item[1][layer]),axis = 2), axis = 1)


    ####### transform layer activations in matrix and calculate rdms
    layer_matrix_list = []
    dc_rdm = []
    for l in layer_dict.keys():
        curr_layer = np.array([layer_dict[l][i] for i in orderedKeys])
        layer_matrix_list.append(curr_layer)
        dc_rdm.append(rdm(curr_layer))


    ###### load fmri rdms
    fmri_mat = hdf5storage.loadmat('/home/CUSACKLAB/annatruzzi/cichy2016/neural_net/algonautsChallenge2019/Training_Data/118_Image_Set/target_fmri.mat')
    EVC = np.mean(fmri_mat['EVC_RDMs'],axis = 0)
    IT = np.mean(fmri_mat['IT_RDMs'], axis = 0)
    fmri_rdm_dict = {'EVC_RDMs' : EVC, 'IT_RDMs' : IT}


    ######### evaluate dc vs fmri
    with open('dc_fmri_scores.txt', 'w') as f:
        for layer in range(0,len(layers)):
            out = evaluate(dc_rdm[layer], fmri_rdm_dict)
            print('=' * 20)
            print('dc%s_fMRI results:' %str(layer+1))
            print('Squared correlation of model to EVC (R**2): {}'.format(out['EVC_RDMs'][0]), '  and significance: {}'.format(out['EVC_RDMs'][1]))
            print('Squared correlation of model to IT (R**2): {}'.format(out['IT_RDMs'][0]), '  and significance: {}'.format(out['IT_RDMs'][1]))
            print('SCORE (average of the two correlations): {}'.format(out['score'])) 
            f.write('=' * 20 + '\n')
            f.write('dc%s_fMRI results:' %str(layer+1) + '\n')
            f.write('Squared correlation of model to EVC (R**2): {}'.format(out['EVC_RDMs'][0]) + '\n')
            f.write('Squared correlation of model to IT (R**2): {}'.format(out['IT_RDMs'][0]) + '\n')
            f.write('SCORE (average of the two correlations): {}'.format(out['score']) + '\n') 

        #evc_percentNC = ((out['EVC_RDMs'][0])/nc118_EVC_R2)*100.      #evc percent of noise ceiling
        #it_percentNC = ((out['IT_RDMs'][0])/nc118_IT_R2)*100.         #it percent of noise ceiling
        #score_percentNC = ((out['score'])/nc118_avg_R2)*100.      #avg (score) percent of noise ceiling
 

    ####### DC plots with numeric order
    rdm_plot(dc_rdm[0], vmin = 0, vmax = 1, labels = orderedNames, main = 'DC layer 1', outname = 'rdm_dc1.png')
    rdm_plot(dc_rdm[1], vmin = 0, vmax = 1, labels = orderedNames, main = 'DC layer 2', outname = 'rdm_dc2.png')
    rdm_plot(dc_rdm[2], vmin = 0, vmax = 1, labels = orderedNames, main = 'DC layer 3', outname = 'rdm_dc3.png')
    rdm_plot(dc_rdm[3], vmin = 0, vmax = 1, labels = orderedNames, main = 'DC layer 4', outname = 'rdm_dc4.png')
    rdm_plot(dc_rdm[4], vmin = 0, vmax = 1, labels = orderedNames, main = 'DC layer 5', outname = 'rdm_dc5.png')

    ####### fmri plots with numeric order
    rdm_plot(EVC, vmin = 0, vmax = 0.8, labels = orderedNames, main = 'EVC', outname = 'rdm_EVC.png')
    rdm_plot(IT, vmin = 0, vmax = 0.8, labels = orderedNames, main = 'IT', outname = 'rdm_IT.png')

    ####### DC plots with lch order
#    rdm_plot(dc_rdm[0], vmin = 0, vmax = 0.5, labels = orderedNames, main = 'DC layer 1', outname = 'rdm_dc1_LCHorder.png')
#    rdm_plot(dc_rdm[1], vmin = 0, vmax = 1, labels = orderedNames, main = 'DC layer 2', outname = 'rdm_dc2_LCHorder.png')
#    rdm_plot(dc_rdm[2], vmin = 0, vmax = 1.2, labels = orderedNames, main = 'DC layer 3', outname = 'rdm_dc3_LCHorder.png')
#    rdm_plot(dc_rdm[3], vmin = 0, vmax = 1.2, labels = orderedNames, main = 'DC layer 4', outname = 'rdm_dc4_LCHorder.png')
#    rdm_plot(dc_rdm[4], vmin = 0, vmax = 1.2, labels = orderedNames, main = 'DC layer 5', outname = 'rdm_dc5_LCHorder.png')


if __name__ == '__main__':
    layers = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5']
    act = '/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_activations.pickle'
    img_names = '/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_img_names.pickle'
    img_synsets = '/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_img_synsets.pickle'
    main(layers,act,img_names,img_synsets)

