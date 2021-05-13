import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys 
import os

matplotlib.use('Agg')

trnpath=sys.argv[1]

subs = os.listdir(trnpath)

for sub in subs:
    bdirs = os.listdir(trnpath + '/' + sub + '/')
    for bdir in bdirs:
        Xtrain= np.load(trnpath + '/' + sub + '/' + bdir+'/'+'X_train_20000.npy')
        plt.figure()
        plt.hist(Xtrain.flatten(),100)
        plt.savefig(trnpath + '/' + sub + '/' + bdir+'/'+'X_train_hist.png')