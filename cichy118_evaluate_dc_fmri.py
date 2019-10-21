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

def lch_order(keys,ordered_names_dict,syn_dict,w2v_dict):
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
    lch_matrix = np.stack((x,y,(1./lch_list)),axis = 1)  ## IMPORTANT: the dendrogram function consideres smallest numbers as index of closeness, lch does the opposite --> 1/lch
    Z = linkage(lch_matrix[:,2], 'ward')  #optimal_ordering = True

    labels_img = []
    for syn in synsets_list:
        for item in syn_dict.items():
            if item[1] == str(syn).split("'")[1]:
                labels_img.append(ordered_names_dict[item[0]])

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


def presentation_order(keys,ordered_names_dict,syn_dict,w2v_dict):
    orderedNames = []
    for item in sorted(ordered_names_dict.items()):
        orderedNames.append(item[1])
    return orderedNames


def w2v_order(keys,ordered_names_dict,syn_dict,w2v_dict):
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

#defines the spearman correlation
def spearman(model_rdm, rdm):
    out_values = stats.spearmanr(rdm, model_rdm)[0] 
    return out_values

#computes spearman correlation (R) and R^2, and ttest for p-value.
def fmri_rdm(model_rdm, fmri_rdms):
    corr = spearman(model_rdm, fmri_rdms)
    corr_squared = np.square(corr)
    return np.mean(corr_squared), stats.ttest_1samp(corr_squared, 0)[1]

def evaluate(submission, targets, target_names=['EVC_RDMs', 'IT_RDMs']):
    out = {}
    for name in target_names:
        out[name] = fmri_rdm(submission, targets[name])
    out['score'] = np.mean([x[0] for x in out.values()])
    return out


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
    act_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/' + comparison_with + '_activations_'+ training + '.pickle'
    img_names_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/' + comparison_with + '_img_names.pickle'
    img_synsets_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/' + comparison_with + '_img_synsets.pickle'
    img_w2v_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/' + comparison_with + '_img_w2v.pickle'

    number = (re.findall('\d+', comparison_with))
    fmri_pth = '/home/CUSACKLAB/annatruzzi/cichy2016/algonautsChallenge2019/Training_Data/'+ number[0]+'_Image_Set/target_fmri.mat'

    act = load_dict(act_pth)
    img_names = load_dict(img_names_pth)
    img_names = reorder_od(img_names, sorted(img_names.keys()))
    img_synsets = load_dict(img_synsets_pth)
    img_synsets = reorder_od(img_synsets, sorted(img_synsets.keys()))
    img_w2v = load_dict(img_w2v_pth)
    img_w2v = reorder_od(img_w2v, sorted(img_w2v.keys()))

    #############   get list of lch ordered activation keys starting from labels_img and img_names dict
    order_function = order_method + '_order(act.keys(),img_names,img_synsets,img_w2v)'
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
                if (item_EVC[0][0] == combination[0] and item_EVC[0][1] == combination[1]) or (item_EVC[0][0] == combination[1] and item_EVC[0][1] == combination[0]):
                    IT_ordered.append(item_IT[1])
                    print item_IT
        EVC = squareform(np.asarray(EVC),force='tovector', checks=False)
        IT = squareform(np.asarray(IT),force='tovector', checks=False)
    
    fmri_rdm_dict = {'EVC_RDMs' : EVC, 'IT_RDMs' : IT}

    
    ######### evaluate dc vs fmri
    out_file_name = '_'.join([network_used, comparison_with, order_method])
    with open('correaltion_' + out_file_name + '.txt', 'w') as f:
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
 

    ####### DC plots
    for i,layer in enumerate(layers):
        main = network_used + ' layer '+str(layer)
        outname = 'rdm_' + out_file_name + '_' + layer
        rdm_plot(dc_rdm[i], vmin = 0, vmax = 1, labels = orderedNames, main = main, outname = outname + '.png')
    

    ####### fmri plots

    rdm_plot(fmri_rdm_dict['EVC_RDMs'], vmin = 0, vmax = 0.8, labels = orderedNames, main = 'EVC', outname = 'rdm_EVC_' + out_file_name + '.png')
    rdm_plot(fmri_rdm_dict['IT_RDMs'], vmin = 0, vmax = 0.8, labels = orderedNames, main = 'IT', outname = 'rdm_IT_' + out_file_name + '.png')



if __name__ == '__main__':

    layers_dc = ['ReLu1', 'ReLu2', 'ReLu3', 'ReLu4', 'ReLu5']
    main(layers_dc, 'DC' ,'niko92', 'w2v', 'pretrained')

