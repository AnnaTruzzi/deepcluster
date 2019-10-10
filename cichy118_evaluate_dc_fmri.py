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
    lch_matrix = np.stack((x,y,lch_list),axis = 1)
    Z = linkage(lch_matrix[:,2], 'ward')

    labels_img = []
    for syn in synsets_list:
        for item in syn_dict.items():
            if item[1] == str(syn).split("'")[1]:
                labels_img.append(names_dict[item[0]])

    #plt.figure()
    den = dendrogram(Z,
                orientation='top',
                labels=labels_img,
                leaf_font_size=9,
                distance_sort='descending',
                show_leaf_counts=True)
    #plt.show()
    orderedNames = den['ivl']
    return orderedNames


def numeric_order(keys,names_dict):
    orderedNames = []
    for item in sorted(names_dict.items()):
        orderedNames.append(item[1])
    return orderedNames


def sq(x):
    return squareform(x, force='tovector', checks=False)

#defines the spearman correlation
def spearman(model_rdm, rdms):
    #model_rdm_sq = sq(model_rdm)
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



def main(layers):
    act = collections.OrderedDict()
    act = load_dict('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_activations.pickle')
    img_names = collections.OrderedDict()
    img_names = load_dict('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_img_names.pickle')
    img_synsets = collections.OrderedDict()
    img_synsets = load_dict('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_img_synsets.pickle')
    
    #############   get list of ordered activation keys starting from labels_img and img_names dict
#    orderedNames = lch_order(act.keys(),img_names,img_synsets)
    orderedNames = numeric_order(act.keys(),img_names)
    orderedKeys = []
    for name in orderedNames:
        number = [i[0] for i in img_names.items() if i[1] == name]
        key = [i for i in act.keys() if number[0] in i]
        orderedKeys.append(key[0])

    ordered_act = collections.OrderedDict()
    ordered_act = {k: act[k] for k in orderedKeys}
    

    layer_dict = {}
    for layer in range(0,len(layers)):
        layer_dict[layer] = {name : np.mean(np.mean(np.squeeze(value[layer]),axis = 2), axis = 1) for name,value in act.items()}




    ####### order layer matrices accordingly to the hierarchical clustering of labels based on lch
    layer1_matrix = np.array([layer1[i] for i in orderedKeys])
    layer2_matrix = np.array([layer2[i] for i in orderedKeys])
    layer3_matrix = np.array([layer3[i] for i in orderedKeys])
    layer4_matrix = np.array([layer4[i] for i in orderedKeys])
    layer5_matrix = np.array([layer5[i] for i in orderedKeys])

    ## calculate rdms
    dc1 = rdm(layer1_matrix)
    dc2 = rdm(layer2_matrix)
    dc3 = rdm(layer3_matrix)
    dc4 = rdm(layer4_matrix)
    dc5 = rdm(layer5_matrix)

    dc_rdm = [dc1, dc2, dc3, dc4, dc5]

    ###### load fmri rdms
    fmri_mat = hdf5storage.loadmat('/home/CUSACKLAB/annatruzzi/cichy2016/neural_net/algonautsChallenge2019/Training_Data/118_Image_Set/target_fmri.mat')
    EVC = np.mean(fmri_mat['EVC_RDMs'],axis = 0)
    IT = np.mean(fmri_mat['IT_RDMs'], axis = 0)
    fmri_rdm_dict = {'EVC_RDMs' : EVC, 'IT_RDMs' : IT}

    dc1_EVC = evaluate(dc1, fmri_rdm_dict)
    #evc_percentNC = ((out['EVC_RDMs'][0])/nc118_EVC_R2)*100.      #evc percent of noise ceiling
    #it_percentNC = ((out['IT_RDMs'][0])/nc118_IT_R2)*100.         #it percent of noise ceiling
    #score_percentNC = ((out['score'])/nc118_avg_R2)*100.      #avg (score) percent of noise ceiling
    print('=' * 20)
    print('dc1_fMRI results:')
    print('Squared correlation of model to EVC (R**2): {}'.format(dc1_EVC['EVC_RDMs'][0]), '  and significance: {}'.format(dc1_EVC['EVC_RDMs'][1]))
    print('Squared correlation of model to IT (R**2): {}'.format(dc1_EVC['IT_RDMs'][0]), '  and significance: {}'.format(dc1_EVC['IT_RDMs'][1]))
    print('SCORE (average of the two correlations): {}'.format(dc1_EVC['score'])) 

    dc2_EVC = evaluate(dc2, fmri_rdm_dict)
    #evc_percentNC = ((out['EVC_RDMs'][0])/nc118_EVC_R2)*100.      #evc percent of noise ceiling
    #it_percentNC = ((out['IT_RDMs'][0])/nc118_IT_R2)*100.         #it percent of noise ceiling
    #score_percentNC = ((out['score'])/nc118_avg_R2)*100.      #avg (score) percent of noise ceiling
    print('=' * 20)
    print('dc2_fMRI results:')
    print('Squared correlation of model to EVC (R**2): {}'.format(dc2_EVC['EVC_RDMs'][0]), '  and significance: {}'.format(dc2_EVC['EVC_RDMs'][1]))
    print('Squared correlation of model to IT (R**2): {}'.format(dc2_EVC['IT_RDMs'][0]), '  and significance: {}'.format(dc2_EVC['IT_RDMs'][1]))
    print('SCORE (average of the two correlations): {}'.format(dc2_EVC['score'])) 

    dc3_EVC = evaluate(dc3, fmri_rdm_dict)
    #evc_percentNC = ((out['EVC_RDMs'][0])/nc118_EVC_R2)*100.      #evc percent of noise ceiling
    #it_percentNC = ((out['IT_RDMs'][0])/nc118_IT_R2)*100.         #it percent of noise ceiling
    #score_percentNC = ((out['score'])/nc118_avg_R2)*100.      #avg (score) percent of noise ceiling
    print('=' * 20)
    print('dc3_fMRI results:')
    print('Squared correlation of model to EVC (R**2): {}'.format(dc3_EVC['EVC_RDMs'][0]), '  and significance: {}'.format(dc3_EVC['EVC_RDMs'][1]))
    print('Squared correlation of model to IT (R**2): {}'.format(dc3_EVC['IT_RDMs'][0]), '  and significance: {}'.format(dc3_EVC['IT_RDMs'][1]))
    print('SCORE (average of the two correlations): {}'.format(dc3_EVC['score'])) 

    dc4_EVC = evaluate(dc4, fmri_rdm_dict)
    #evc_percentNC = ((out['EVC_RDMs'][0])/nc118_EVC_R2)*100.      #evc percent of noise ceiling
    #it_percentNC = ((out['IT_RDMs'][0])/nc118_IT_R2)*100.         #it percent of noise ceiling
    #score_percentNC = ((out['score'])/nc118_avg_R2)*100.      #avg (score) percent of noise ceiling
    print('=' * 20)
    print('dc4_fMRI results:')
    print('Squared correlation of model to EVC (R**2): {}'.format(dc4_EVC['EVC_RDMs'][0]), '  and significance: {}'.format(dc4_EVC['EVC_RDMs'][1]))
    print('Squared correlation of model to IT (R**2): {}'.format(dc4_EVC['IT_RDMs'][0]), '  and significance: {}'.format(dc4_EVC['IT_RDMs'][1]))
    print('SCORE (average of the two correlations): {}'.format(dc4_EVC['score'])) 

    dc5_EVC = evaluate(dc5, fmri_rdm_dict)
    #evc_percentNC = ((out['EVC_RDMs'][0])/nc118_EVC_R2)*100.      #evc percent of noise ceiling
    #it_percentNC = ((out['IT_RDMs'][0])/nc118_IT_R2)*100.         #it percent of noise ceiling
    #score_percentNC = ((out['score'])/nc118_avg_R2)*100.      #avg (score) percent of noise ceiling
    print('=' * 20)
    print('dc5_fMRI results:')
    print('Squared correlation of model to EVC (R**2): {}'.format(dc5_EVC['EVC_RDMs'][0]), '  and significance: {}'.format(dc5_EVC['EVC_RDMs'][1]))
    print('Squared correlation of model to IT (R**2): {}'.format(dc5_EVC['IT_RDMs'][0]), '  and significance: {}'.format(dc5_EVC['IT_RDMs'][1]))
    print('SCORE (average of the two correlations): {}'.format(dc5_EVC['score'])) 


    ####### DC plot
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(dc5,vmin=0,vmax=1.2)
    plt.colorbar()
    ticks = np.arange(0,118,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(orderedNames,rotation = 90, fontsize = 3)
    ax.set_yticklabels(orderedNames,fontsize = 8)
    #plt.show()
    fig.savefig('rdm5.png', dpi = (800))
    plt.close(fig)

    ####### EVC plot
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(EVC,vmin=0,vmax=0.6)
    plt.colorbar()
    ticks = np.arange(0,118,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(orderedNames,rotation = 90, fontsize = 3)
    ax.set_yticklabels(orderedNames,fontsize = 8)
    #plt.show()
    fig.savefig('rdmEVC.png', dpi = (800))
    plt.close(fig)

    ####### IT plot
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(IT,vmin=0,vmax=0.8)
    plt.colorbar()
    ticks = np.arange(0,118,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(orderedNames,rotation = 90, fontsize = 3)
    ax.set_yticklabels(orderedNames,fontsize = 8)
    #plt.show()
    fig.savefig('rdmIT.png', dpi = (800))
    plt.close(fig)



if __name__ == '__main__':
    layers = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5']
    main(layers)

