# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:27:58 2014

@author: anhhh11
"""

import numpy as np
import matplotlib.pyplot as plt        
from a1.LFD.LinearRegression import *
from a1.LFD import *

def get_new_data(N=100):
    d=2
    X = np.random.uniform(low=-1,high=1,size=[N,d])
    b0, b1 = Simulation.Simulation.random_line_betas()
    y = np.sign(b0 + b1*X[:,0] - X[:,1])
    return X,y

def ln_weight(X,y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

np.random.seed(123456)
X,y = get_new_data(N=100)
lg = LinearRegression(X,y)
lg.learn()
w = lg.weight_result()
lg.validator.is_complete()
print lg.validator.get_number_wrong_predicted_cases()
# = np.column_stack((X,ones(X.shape[0])))
#w = ln_weight(X,y)
#print w
#print np.linalg.lstsq(X,y)[0]

ycolors = y
ycolors = np.array(ycolors,dtype='str')
ycolors[ycolors=="1.0"] = "green"
ycolors[ycolors=="-1.0"] = "red"
Line.Utils.abline(-w[2]/w[1],-w[0]/w[1],X[:,0])
plt.scatter(X[:,0],X[:,1],c=ycolors)
plt.draw()
