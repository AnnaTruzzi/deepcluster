import os
import scipy.io
import pickle
import numpy as np
from itertools import compress
from scipy.spatial import distance
from sklearn.manifold import MDS
from statsmodels.stats.anova import AnovaRM
from matplotlib import pyplot as plt

with open('/home/CUSACKLAB/annatruzzi/cichy2016/cichy118_activations.pickle','rb') as file:
    act=pickle.load(file)

img_names = {'1':'orange','2':'bench','3':'remote','4':'car','5':'stove','6':'man','7':'table','8':'apple','9':'chart','10':'dog','11':'fox','12':'bus','13':'train','14':'ipod','15':'pizza','16':'bird',
                '17':'horse','18':'laptop','19':'bear','20':'basketball','21':'piano','22':'guitar','23':'baseball','24':'seal','25':'chair','26':'orangutan','27':'bowl','28':'tiger','29':'moped',
                '30':'tie','31':'printer','32':'lion','33':'neil','34':'drum','35':'bow','36':'pomgranade','37':'butterfly','38':'lamp','39':'banana','40':'sofa','41':'lemon','42':'hoover','43':'hammer',
                '44':'sunglasses','45':'bike','46':'rabbit','47':'measuring jug','48':'elephant','49':'microwave','50':'volleyball','51':'strawberry','52':'sheep','53':'frog','54':'washing machine',
                '55':'fridge','56':'turtle','57':'axe','58':'helmet','59':'camel','60':'lipstick','61':'airplane','62':'dishwasher','63':'burger','64':'backpack','65':'purse','66':'hamster','67':'microphone',
                '68':'mushroom','69':'cow','70':'violin','71':'orca','72':'cucumber','73':'termometer','74':'pineapple','75':'harp','76':'squirrel','77':'notebook','78':'trumpet','79':'plastic bag',
                '80':'jug','81':'snail','82':'toaster','83':'benjo','84':'screwdriver','85':'snowbike','86':'pig','87':'pomegranate','88':'accordion','89':'tennis racket','90':'bagel','91':'trombone',
                '92':'syringe','93':'fish','94':'hair drier','95':'starfish','96':'hot dog','97':'ladybug','98':'aircraft carrier','99':'jellyfish','100':'caffee machine','101':'television','102':'pretzel',
                '103':'artichok','104':'golf ball','105':'alarm clock','106':'traffic light','107':'bottle','108':'pan','109':'weights','110':'football ball','111':'bellpepper','112':'corkscrew',
                '113':'tennis ball','114':'computer mouse','115':'mug','116':'keyboard','117':'can opener','118':'fan'}


## create dict per layer
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

orderedNames = act.keys()
labels_dict = [label.split('/')[-1] for label in act.keys()]
labels_img = []
for key in labels_dict:
    for item in img_names.items():
        if item[0] in key:
            labels_img.append(item[1])

layer1_matrix = np.array([layer1[i] for i in orderedNames])
layer2_matrix = np.array([layer2[i] for i in orderedNames])
layer3_matrix = np.array([layer3[i] for i in orderedNames])
layer4_matrix = np.array([layer4[i] for i in orderedNames])
layer5_matrix = np.array([layer5[i] for i in orderedNames])


rdm1 = distance.squareform(distance.pdist(layer1_matrix,metric='euclidean'))
rdm2 = distance.squareform(distance.pdist(layer2_matrix,metric='euclidean'))
rdm3 = distance.squareform(distance.pdist(layer3_matrix,metric='euclidean'))
rdm4 = distance.squareform(distance.pdist(layer4_matrix,metric='euclidean'))
rdm5 = distance.squareform(distance.pdist(layer5_matrix,metric='euclidean'))

fig=plt.figure()
ax = fig.add_subplot(111)
plt.imshow(rdm5,vmin=0,vmax=3)
plt.colorbar()
ticks = np.arange(0,118,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(labels_img,rotation = 90, fontsize = 8)
ax.set_yticklabels(labels_img,fontsize = 8)
plt.show()


