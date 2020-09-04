import pickle
import numpy as np

pth = '/home/annatruzzi/deepcluster_eval/dc_1/log/loss_log'
with open (pth,'rb') as f:
    data = pickle.load(f)
    print(data)
 
