# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt        
import time
from functools import partial
from a1.LFD import *
from a1.LFD.Perceptron import *
import numpy as np        
import matplotlib.animation as animation

np.random.seed(0)
#Customer given you data (X,y)
#Assume that X from `unknown distribution`, y = f(X), with f is real target function
# With X have more than 2 dimensions we can use PCM to reduce to 2 dim to plot easily
# For learning perception algorithm, we simulate assign `unknown dist` here is
# distribution that has `uniform dist for each column of data`
# X,y is what customer give, all above just for simulating
# y = ax+b
# ax + by + c =0
# w is (a,b,c)
# line for w is y = -a/b*x - c/b

# Qestion 7

def get_new_data(N=100,d=2):
    X = np.random.uniform(low=-1,high=1,size=[N,d])
    b0, b1 = Simulation.Simulation.random_line_betas()
    y = np.sign(b0 + b1*X[:,0] - X[:,1])
    return X,y
    
def iteration_used(iter):
    X,y = get_new_data()
    #draw_points(X,y)
    p = Perceptron(X,y,iter)
    #p.learn(partial(draw_line,X))
    p.learn()
    draw_line(X,p.weight_result())
    draw_points(X,y)
    return (p.is_complete() and p.iteration_used) or -1

def main():
    #plt.figure()
    #plt.ion()
    #plt.show()
    iter_used = []
    for i in xrange(1):
        iter_used.append(iteration_used(iter=200))
    iter_used   = np.array(iter_used)
    number_error_case = iter_used[iter_used<0].size
    iter_used_to_coverage = iter_used[iter_used>-1].size
    # max_iter = 10000: mean iteration: 46.9583, n of error case 2865
    print "Mean of iterations to coverage:",np.mean(iter_used_to_coverage) 
    print "Number of error case for N=10, in 1000 time tries:",number_error_case 
def draw_points(X,y):
    ycolors = y
    ycolors = np.array(ycolors,dtype='str')
    ycolors[ycolors=="1.0"] = "green"
    ycolors[ycolors=="-1.0"] = "red"
    plt.scatter(X[:,0],X[:,1],c=ycolors,s=50)
    plt.draw()
    
def draw_line(X,w):
    Line.Utils.abline(-w[2]/w[1],-w[0]/w[1],X[:,0])
    plt.draw()
    time.sleep(0.5)
    
main()


