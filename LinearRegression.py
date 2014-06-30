# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:27:58 2014

@author: anhhh11
"""

import numpy as np
import matplotlib.pyplot as plt        
import time
from functools import partial
from a1.LFD import *
from a1.LFD.Perceptron import *
import numpy as np        
import matplotlib.animation as animation


def get_new_data(N=100):
    d=2
    X = np.random.uniform(low=-1,high=1,size=[N,d])
    b0, b1 = Simulation.Simulation.random_line_betas()
    y = np.sign(b0 + b1*X[:,0] - X[:,1])
    return X,y

def ln_weight(X,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

X,y = get_new_data(N=100)
X = np.column_stack((X,ones(X.shape[0])))
w = ln_weight(X,y)
#w = np.linalg.lstsq(X,y)[0]
ycolors = y
ycolors = np.array(ycolors,dtype='str')
ycolors[ycolors=="1.0"] = "green"
ycolors[ycolors=="-1.0"] = "red"
Line.Utils.abline(w[1],w[0],X[:,0])
plt.scatter(X[:,0],X[:,1],c=ycolors)
plt.draw()
