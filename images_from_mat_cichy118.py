import scipy.io
from PIL import Image
import numpy as np
import os

outpth = '/home/CUSACKLAB/annatruzzi/cichy2016/stimuli/cichy/jpg_images'
mat = scipy.io.loadmat('/home/CUSACKLAB/annatruzzi/cichy2016/stimuli/cichy/algonautsChallenge2019/Training_Data/118_Image_Set/118images.mat')

for i in range(0,len(mat['visual_stimuli'][('pixels')][0])):
    img = Image.fromarray(mat['visual_stimuli'][('pixels')][0][i], 'RGB')
    outname = os.path.join(outpth,mat['visual_stimuli'][('names')][0][i][0].encode("utf-8"))
    img.save(outname)
