# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt        
import time
from a1.LFD import *
from a1.LFD.Perceptron import *
import numpy as np        


np.random.seed(1234)
#Customer given you data (X,y)
#Assume that X from `unknown distribution`, y = f(X), with f is real target function
# With X have more than 2 dimensions we can use PCM to reduce to 2 dim to plot easily
# For learning perception algorithm, we simulate assign `unknown dist` here is
# distribution that has `uniform dist for each column of data`
d = 2
X = np.random.uniform(low=-1,high=1,size=[10,2])
b0, b1 = Simulation.Simulation.random_line_betas()
y = np.sign(b0 + b1*X[:,0] - X[:,1])
# X,y is what customer give, all above just for simulating
print X
print y
# y = ax+b
# ax + by + c =0
# w is (a,b,c)
# line for w is y = -a/b*x - c/b
p = Perceptron(X,y,3000)
w = p.w
fig=plt.figure()
plt.ion()
plt.show()
def draw(w):
    ycolors = y
    ycolors = np.array(ycolors,dtype='str')
    ycolors[ycolors=="1.0"] = "green"
    ycolors[ycolors=="-1.0"] = "red"
    Line.Utils.abline(-w[2]/w[1],-w[0]/w[1],X[:,0])
    plt.scatter(X[:,0],X[:,1],c=ycolors,s=50)
    plt.draw()
    time.sleep(0.1)
p.learn()
draw(p.w)