import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os

def main(pth):
    loss_list = []
    for num in range(0,37,2):
        directory = pth+'checkpoint'+str(num)+'/log/loss_log'
        print(directory)
        try:
            with open (directory,'rb') as f:
                data = pickle.load(f)
                loss_list.append(data[-1])
                print(len(data))
        except:
            continue
    plt.plot(loss_list)
    plt.savefig('LossPlot_dc3.png',bbox_inches='tight')
    plt.close()


 
if __name__ == '__main__':
    pth = '/home/annatruzzi/deepcluster_eval/dc_3/'
    main(pth)
