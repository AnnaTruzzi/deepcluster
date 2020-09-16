import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def main(pth):
    loss_list = []
    for directory in glob.glob(pth+'*'):
        print(directory)
        filename = os.path.join(directory,'log/loss_log')
        print(filename)
        with open (filename,'rb') as f:
            data = pickle.load(f)
            loss_list.append(data[-1])
            print(len(data))
    plt.plot(loss_list)
    plt.savefig('LossPlot_dc1.png',bbox_inches='tight')
    plt.close()


 
if __name__ == '__main__':
    pth = '/home/annatruzzi/deepcluster_eval/dc_1/'
    main(pth)
