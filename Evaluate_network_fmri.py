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
import re
from gensim.models import KeyedVectors
import skbio


## load activations dictionary 
def load_dict(path):
    with open(path,'rb') as file:
        dict=pickle.load(file)
    return dict

def reorder_od(dict1,order):
   new_od = collections.OrderedDict([(k,None) for k in order if k in dict1])
   new_od.update(dict1)
   return new_od
   
def loadmat(matfile):
    try:
        f = h5py.File(matfile)
    except (IOError, OSError):
        return io.loadmat(matfile)
    else:
        return {name: np.transpose(f.get(name)) for name in f.keys()}

def lch_order(ordered_names_dict,syn_dict,w2v_dict):
    ## get LCH distance for the images from the respective synsets and order them by hierarchical clustering + get respective (comprehensible) labels 
    cmb = list(combinations(ordered_names_dict.values(),2))
    lch_list = []
    x = []
    y = []
    for combo in cmb:
        x.append(combo[0])
        y.append(combo[1])
        syn1 = wn.synset(syn_dict[ordered_names_dict.keys()[ordered_names_dict.values().index(combo[0])]])
        syn2 = wn.synset(syn_dict[ordered_names_dict.keys()[ordered_names_dict.values().index(combo[1])]])
        lch = syn1.lch_similarity(syn2)
        lch_list.append(lch)

    x = np.array(x,dtype = str)
    y = np.array(y, dtype = str)
    lch_list = np.array(lch_list,dtype = float)
    lch_matrix = np.stack((x,y,(1./lch_list)),axis = 1)  ## IMPORTANT: the dendrogram function consideres smallest numbers as index of closeness, lch does the opposite --> 1/lch
    Z = linkage(lch_matrix[:,2], 'ward')  #optimal_ordering = True

    plt.figure()
    den = dendrogram(Z,
                orientation='top',
                labels=ordered_names_dict.values(),
                leaf_font_size=9,
                distance_sort='ascending',
                show_leaf_counts=True)
    plt.show()
    orderedNames = den['ivl']
    return orderedNames


def presentation_order(ordered_names_dict,syn_dict,w2v_dict):
    orderedNames = []
    for item in sorted(ordered_names_dict.items()):
        orderedNames.append(item[1])
    return orderedNames


def w2v_order(ordered_names_dict,syn_dict,w2v_dict):
    orderedNames = []
    model = KeyedVectors.load_word2vec_format('/home/CUSACKLAB/annatruzzi/GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
    w2v = []
    for key in ordered_names_dict.keys():
        value = w2v_dict[key]
        if len(value.split(' ')) == 1:
            w2v.append(model[value])
        else:
            word1 = value.split(' ')[0]
            word2 = value.split(' ')[1]
            vector = model[word1] + model[word2]
            w2v.append(vector)

    w2v = np.array(w2v,dtype = float)
    rdm_w2v = distance.squareform(distance.pdist(w2v,metric='correlation')) 
    Z = linkage(squareform(rdm_w2v), 'ward')

    den = dendrogram(Z,
        orientation='top',
        labels=ordered_names_dict.values(),
        leaf_font_size=9,
        distance_sort='ascending',
        show_leaf_counts=True)
    plt.show()
    orderedNames = den['ivl']

    return orderedNames


def rdm(act_matrix):
    rdm_matrix = distance.squareform(distance.pdist(act_matrix,metric='correlation'))
    return rdm_matrix


def rdm_plot(rdm, vmin, vmax, labels, main, outname):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(rdm,vmin=vmin,vmax=vmax)
    plt.colorbar()
    ticks = np.arange(0,len(labels),1)
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


def main(layers, network_used, comparison_with, order_method,training):
    
    ''' Calcutates the correlation between activations in a neural network and brain activations in early visual cortex (EVC) and IT

    It can be applied to activations in 2 networks, 2 and from two differnt set of images, cichy118 and niko92.
    also, it is possible to order the items accordingly to lch similarity or word2vec similarity rather than in the presentation order if necessary

    Args:
        layers: list of layers of the network for which we have activations
        network_used: choose between DC vs alexnet
        comparison_with: choose between cichy118 vs niko92
        order_method: choose between presentation vs lch vs w2v
        training: choose between pretrained vs untrained
    
    Returns:
        RDMs and correlations between neural network RDMs and fmri RDMs
    '''
    
    act_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/' + comparison_with + '_activations_'+ training + '.pickle'
    img_names_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/' + comparison_with + '_img_names.pickle'
    img_synsets_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/' + comparison_with + '_img_synsets.pickle'
    img_w2v_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/' + comparison_with + '_img_w2v.pickle'

    number = (re.findall(r'\d+', comparison_with))
    fmri_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/algonautsChallenge2019/Training_Data/'+ number[0]+'_Image_Set/target_fmri.mat'

    act = load_dict(act_pth)
    img_names = load_dict(img_names_pth)
    img_names = reorder_od(img_names, sorted(img_names.keys()))
    img_synsets = load_dict(img_synsets_pth)
    img_synsets = reorder_od(img_synsets, sorted(img_synsets.keys()))
    img_w2v = load_dict(img_w2v_pth)
    img_w2v = reorder_od(img_w2v, sorted(img_w2v.keys()))

    #############   get list of lch ordered activation keys starting from labels_img and img_names dict
    order_function = order_method + '_order(img_names,img_synsets,img_w2v)'
    orderedNames = eval(order_function)
    orderedKeys = []
    for name in orderedNames:
        number = [i[0] for i in img_names.items() if i[1] == name]
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


    ###### load fmri rdms and re-order them by lch similarity
    fmri_mat = loadmat(fmri_pth)
    EVC = np.mean(fmri_mat['EVC_RDMs'],axis = 0)
    IT = np.mean(fmri_mat['IT_RDMs'], axis = 0)

    if order_method is not 'presentation':
        EVC_tovector = squareform(EVC, force='tovector', checks=False)
        IT_tovector = squareform(IT, force='tovector', checks=False)
        EVC_ordered = []
        IT_ordered = []
        cmb_mri = list(combinations(img_names.values(),2))
        cmb_mri_EVC = zip(cmb_mri, EVC_tovector)
        cmb_mri_IT = zip(cmb_mri, IT_tovector)
        cmb_orderedNames = list(combinations(orderedNames,2))
        for combination in cmb_orderedNames:
            print combination
            for item_EVC in cmb_mri_EVC:
                if (item_EVC[0][0] == combination[0] and item_EVC[0][1] == combination[1]) or (item_EVC[0][0] == combination[1] and item_EVC[0][1] == combination[0]):
                    EVC_ordered.append(item_EVC[1])
                    print item_EVC
            for item_IT in cmb_mri_IT:
                if (item_IT[0][0] == combination[0] and item_IT[0][1] == combination[1]) or (item_IT[0][0] == combination[1] and item_IT[0][1] == combination[0]):
                    IT_ordered.append(item_IT[1])
                    print item_IT
        EVC = squareform(np.asarray(EVC_ordered), checks=False)
        IT = squareform(np.asarray(IT_ordered), checks=False)
    
    fmri_rdm_dict = {'EVC_RDMs' : EVC, 'IT_RDMs' : IT}

    
    ######### evaluate dc vs fmri
    out_file_name = '_'.join([network_used, comparison_with])
    with open('correlation_' + out_file_name + '_' + training + '_mantel.txt', 'w') as f:
        for layer in range(0,len(layers)):
            #EVC_corr,EVC_pvalue = stats.spearmanr(squareform(dc_rdm[layer],force='tovector',checks=False), squareform(fmri_rdm_dict['EVC_RDMs'],force='tovector',checks=False))
            #IT_corr,IT_pvalue = stats.spearmanr(squareform(dc_rdm[layer],force='tovector',checks=False), squareform(fmri_rdm_dict['IT_RDMs'],force='tovector',checks=False))
            EVC_corr,EVC_pvalue = skbio.math.stats.distance.mantel(dc_rdm[layer],fmri_rdm_dict['EVC_RDMs'], method = 'spearman', permutations = 10000)
            IT_corr,IT_pvalue = skbio.math.stats.distance.mantel(dc_rdm[layer],fmri_rdm_dict['IT_RDMs'], method = 'spearman', permutations = 10000)
            f.write('=' * 20 + '\n')
            f.write('dc%s_fMRI results:' %str(layer+1) + '\n')
            f.write('Spearman correlation of model to EVC: {}'.format(EVC_corr) + ' and p value:{}'.format(EVC_pvalue) + '\n')
            f.write('Squared correlation of model to IT: {}'.format(IT_corr) + ' and p value:{}'.format(IT_pvalue) + '\n')


    ####### DC plots
    for i,layer in enumerate(layers):
        main = network_used + ' layer '+str(layer)
        outname = '_'.join(['rdm',out_file_name, order_method,layer, training])
        rdm_plot(dc_rdm[i], vmin = 0, vmax = 1, labels = orderedNames, main = main, outname = outname + '.png')
    

    ####### fmri plots
    rdm_plot(fmri_rdm_dict['EVC_RDMs'], vmin = 0, vmax = 0.8, labels = orderedNames, main = 'EVC', outname = '_'.join(['rdm_EVC',out_file_name,order_method]) + '.png')
    rdm_plot(fmri_rdm_dict['IT_RDMs'], vmin = 0, vmax = 0.8, labels = orderedNames, main = 'IT', outname = '_'.join(['rdm_IT',out_file_name,order_method]) + '.png')



if __name__ == '__main__':
    
    layers_dc = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5']
    main(layers_dc, 'DC' ,'niko92', 'presentation', 'pretrained')
