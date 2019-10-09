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
    labels_dict = [label.split('/')[-1] for label in keys]
    for item in labels_dict:
        name = [i[1] for i in names_dict.items() if i[0] in item]
        orderedNames.append(name[0])
    return orderedNames



def main():
    ## define images names
    img_names = {'001':'orange','002':'bench','003':'remote','004':'car','005':'stove','006':'man','007':'table','008':'apple','009':'chart','010':'dog','011':'fox','012':'bus','013':'train','014':'ipod',
                    '015':'pizza','016':'bird','017':'horse','018':'laptop','019':'bear','020':'basketball','021':'piano','022':'guitar','023':'baseball','024':'seal','025':'chair','026':'orangutan','027':'bowl',
                    '028':'tiger','029':'moped','030':'tie','031':'printer','032':'lion','033':'neil','034':'drum','035':'bow','036':'fig','037':'butterfly','038':'lamp','039':'banana','040':'sofa',
                    '041':'lemon','042':'hoover','043':'hammer','044':'sunglasses','045':'bike','046':'rabbit','047':'measuring jug','048':'elephant','049':'microwave','050':'volleyball','051':'strawberry',
                    '052':'sheep','053':'frog','054':'washing machine','055':'fridge','056':'turtle','057':'axe','058':'helmet','059':'camel','060':'lipstick','061':'airplane','062':'dishwasher','063':'burger',
                    '064':'backpack','065':'purse','066':'hamster','067':'microphone','068':'mushroom','069':'cow','070':'violin','071':'orca','072':'cucumber','073':'termometer','074':'pineapple','075':'harp',
                    '076':'squirrel','077':'notebook','078':'trumpet','079':'plastic bag','080':'jug','081':'snail','082':'toaster','083':'benjo','084':'screwdriver','085':'snowbike','086':'pig',
                    '087':'pomegranate','088':'accordion','089':'tennis racket','090':'bagel','091':'trombone','092':'syringe','093':'fish','094':'hair drier','095':'starfish','096':'hot dog','097':'ladybug',
                    '098':'aircraft carrier','099':'jellyfish','100':'caffee machine','101':'television','102':'pretzel','103':'artichok','104':'golf ball','105':'alarm clock','106':'traffic light',
                    '107':'bottle','108':'pan','109':'weights','110':'football ball','111':'bellpepper','112':'corkscrew','113':'tennis ball','114':'computer mouse','115':'mug','116':'keyboard',
                    '117':'can opener','118':'fan'}

    img_synsets = {'001':'orange.n.01','002':'bench.n.01','003':'remote_control.n.01','004':'car.n.01','005':'stove.n.01','006':'man.n.01','007':'table.n.02','008':'apple.n.01','009':'cart.n.01','010':'dog.n.01',
                    '011':'fox.n.01','012':'bus.n.01','013':'train.n.01','014':'ipod.n.01','015':'pizza.n.01','016':'bird.n.01','017':'horse.n.01','018':'laptop.n.01','019':'bear.n.01','020':'basketball.n.02',
                    '021':'piano.n.01','022':'guitar.n.01','023':'baseball.n.01','024':'seal.n.09','025':'chair.n.01','026':'orangutan.n.01','027':'bowl.n.01','028':'tiger.n.02','029':'moped.n.01','030':'necktie.n.01',
                    '031':'printer.n.03','032':'lion.n.01','033':'nail.n.02','034':'drum.n.01','035':'bow.n.04','036':'fig.n.04','037':'butterfly.n.01','038':'lamp.n.02','039':'banana.n.02',
                    '040':'sofa.n.01','041':'lemon.n.01','042':'vacuum.n.04','043':'hammer.n.02','044':'sunglasses.n.01','045':'bicycle.n.01','046':'rabbit.n.01','047':'beaker.n.01','048':'elephant.n.01',
                    '049':'microwave.n.02','050':'volleyball.n.02','051':'strawberry.n.01','052':'sheep.n.01','053':'frog.n.01','054':'washer.n.03','055':'electric_refrigerator.n.01','056':'turtle.n.02','057':'ax.n.01',
                    '058':'helmet.n.02','059':'camel.n.01','060':'lipstick.n.01','061':'airplane.n.01','062':'dishwasher.n.01','063':'hamburger.n.01','064':'backpack.n.01','065':'bag.n.04','066':'hamster.n.01',
                    '067':'microphone.n.01','068':'mushroom.n.05','069':'cow.n.02','070':'violin.n.01','071':'killer_whale.n.01','072':'cucumber.n.02','073':'thermometer.n.01','074':'pineapple.n.02','075':'harp.n.01',
                    '076':'squirrel.n.01','077':'notebook.n.01','078':'cornet.n.01','079':'plastic_bag.n.01','080':'pitcher.n.02','081':'snail.n.01','082':'toaster.n.02','083':'banjo.n.01',
                    '084':'screwdriver.n.01','085':'snowmobile.n.01','086':'hog.n.03','087':'pomegranate.n.02','088':'accordion.n.01','089':'racket.n.04','090':'bagel.n.01','091':'trombone.n.01',
                    '092':'syringe.n.01','093':'fish.n.01','094':'hand_blower.n.01','095':'starfish.n.01','096':'frank.n.02','097':'ladybug.n.01','098':'aircraft_carrier.n.01','099':'jellyfish.n.02',
                    '100':'coffee_maker.n.01','101':'television_receiver.n.01','102':'pretzel.n.01','103':'artichoke.n.01','104':'golf_ball.n.01','105':'alarm_clock.n.01','106':'traffic_light.n.01',
                    '107':'bottle.n.01','108':'pan.n.01','109':'weight.n.02','110':'ball.n.01','111':'sweet_pepper.n.02','112':'corkscrew.n.01','113':'tennis_ball.n.01','114':'mouse.n.04','115':'mug.n.04',
                    '116':'keyboard.n.01','117':'can_opener.n.01','118':'fan.n.01'}

    act = load_dict('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_activations.pickle')
    ## create one dictionary per layer
    layer1 = {}
    layer2 = {}
    layer3 = {}
    layer4 = {}
    layer5 = {}

    for item in act.items():
        layer1[item[0]] = np.mean(np.mean(np.squeeze(item[1][0]),axis = 2), axis = 1)
        layer2[item[0]] = np.mean(np.mean(np.squeeze(item[1][1]),axis = 2), axis = 1)
        layer3[item[0]] = np.mean(np.mean(np.squeeze(item[1][2]),axis = 2), axis = 1)
        layer4[item[0]] = np.mean(np.mean(np.squeeze(item[1][3]),axis = 2), axis = 1)
        layer5[item[0]] = np.mean(np.mean(np.squeeze(item[1][4]),axis = 2), axis = 1)


    #############   get list of ordered activation keys starting from labels_img and img_names dict
#    orderedNames = lch_order(act.keys(),img_names,img_synsets)
    orderedNames = numeric_order(act.keys(),img_names)
    orderedKeys = []
    for name in orderedNames:
        number = [i[0] for i in img_names.items() if i[1] == name]
        key = [i for i in act.keys() if number[0] in i]
        orderedKeys.append(key[0])


    ####### order layer matrices accordingly to the hierarchical clustering of labels based on lch
    layer1_matrix = np.array([layer1[i] for i in orderedKeys])
    layer2_matrix = np.array([layer2[i] for i in orderedKeys])
    layer3_matrix = np.array([layer3[i] for i in orderedKeys])
    layer4_matrix = np.array([layer4[i] for i in orderedKeys])
    layer5_matrix = np.array([layer5[i] for i in orderedKeys])

    ## calculate rdms
    rdm1 = rdm(layer1_matrix)
    rdm2 = rdm(layer2_matrix)
    rdm3 = rdm(layer3_matrix)
    rdm4 = rdm(layer4_matrix)
    rdm5 = rdm(layer5_matrix)


    ####### rdm plot
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(rdm5,vmin=0,vmax=1.2)
    plt.colorbar()
    ticks = np.arange(0,118,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(orderedNames,rotation = 90, fontsize = 8)
    ax.set_yticklabels(orderedNames,fontsize = 8)
    plt.show()


    ###### load fmri rdms
    fmri_mat = hdf5storage.loadmat('/home/CUSACKLAB/annatruzzi/cichy2016/neural_net/algonautsChallenge2019/Training_Data/118_Image_Set/target_fmri.mat')
    EVC = np.mean(fmri_mat['EVC_RDMs'],axis = 0)
    IT = np.mean(fmri_mat['IT_RDMs'], axis = 0)

    ####### EVC plot
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(EVC,vmin=0,vmax=1.2)
    plt.colorbar()
    ticks = np.arange(0,118,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(orderedNames,rotation = 90, fontsize = 8)
    ax.set_yticklabels(orderedNames,fontsize = 8)
    plt.show()

    ####### IT plot
    fig=plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(IT,vmin=0,vmax=1.2)
    plt.colorbar()
    ticks = np.arange(0,118,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(orderedNames,rotation = 90, fontsize = 8)
    ax.set_yticklabels(orderedNames,fontsize = 8)
    plt.show()




if __name__ == '__main__':
    main()

