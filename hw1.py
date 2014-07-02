# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
from functools import partial
from sklearn.cluster.tests.test_k_means import n_samples
from a1.LFD import *
from a1.LFD.Perceptron import *
from a1.LFD.LinearRegression import *
import numpy as np        
import matplotlib.animation as animation
from a1.LFD.Validator import Validator

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

def iteration_used(iter,data_size):
    X,y = get_new_data(N=data_size)
    #draw_points(X,y)
    p = Perceptron(X,y,iter)
    q = LinearRegression(X,y)
    q.learn()
    p.set_weight(q.weight_result())

    #p.learn(partial(draw_line,X))
    p.learn()
    #draw_line(X,p.weight_result())
    #draw_points(X,y)
    return (p.validator.is_complete() and p.iteration_used) or -1

def hw1():
    iter_used = []
    n_trial = 10000
    data_size = 10
    for i in xrange(n_trial):
        iter_used.append(iteration_used(iter=100,data_size=data_size))
    iter_used  = np.array(iter_used)
    number_error_case = iter_used[iter_used<0].size
    iter_used_to_coverage = iter_used[iter_used>-1] # 4.531

    print "Mean of iterations to coverage:",np.mean(iter_used_to_coverage) 
    print "Number of error case for N={0}, in {1} time tries:".format(data_size,n_trial),number_error_case

def hw2_q5():
    #plt.figure()
    g = np.empty([10000,3])
    X_pop,y_pop = get_new_data(1000)
    E_in = 0
    E_out = 0
    number_of_trail = 10000
    lg_whole = LinearRegression(X_pop,y_pop)
    lg_whole._prepare()
    for i in xrange(number_of_trail):
        select = np.random.choice(X_pop.shape[0],size=100,replace=False)
        X = X_pop[select]
        y = y_pop[select]
        #draw_points(X,y)
        lg = LinearRegression(X,y)
        lg.learn()
        g[i] = lg.weight_result()
        #draw_line(X,g[i])
        lg.is_complete()
        lg.validator.get_predict_error_proportion()
        #plt.draw()
        E_in += lg.validator.get_predict_error_proportion()
        
        lg_whole.set_weights(g[i])
        lg_whole._config_validator()
        lg_whole.is_complete()
        E_out += lg_whole.validator.get_predict_error_proportion()

    return (E_in/number_of_trail,E_out/number_of_trail)
    
#def hw2_q6():
#    number_of_trail = 1000
#    E_out = 0
#    X,y = X_pop,y_pop
#    lg = LinearRegression(X,y)
#    lg._prepare()
#    #lg.learn()
#    #lg.is_complete()
#    for i in xrange(number_of_trail):
#        lg.set_weights(g[i])
#        lg._config_validator()
#        lg.is_complete()
#        E_out += lg.validator.get_predict_error_proportion()
#    return E_out/number_of_trail


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
    time.sleep(0.05)


def generate_distribution(N, d, n_sample):
    X = np.random.uniform(low=-1, high=1, size=[N, d])
    y = np.sign(np.power(X[:, 0], 2) +
                np.power(X[:, 1], 2) -
                0.6)
    X = np.column_stack((X,
                         X[:, 0] * X[:, 1],
                         np.power(X[:, 0], 2),
                         np.power(X[:, 1], 2)))
    error_pos = np.random.choice(N, size=n_sample, replace=False)
    y_orgin = np.copy(y)
    y[error_pos] = - y[error_pos]
    return X, y, y_orgin


def hw2_nonlinear_transformation():
    n_trial = 1000
    e_in = 0.0
    e_out = 0.0
    w = None
    for i in xrange(n_trial):
        X, y, y_orgin = generate_distribution(N=1000, d=2, n_sample=1000*10/100)
        lg = LinearRegression(X,y)
        lg.learn()
        lg.is_complete()
        if not isinstance(w,np.ndarray):
            w = lg.weight_result()
        w += lg.weight_result()
        e_in += lg.validator.get_predict_error_proportion()


        X_out, y_out, y_out_orgin = generate_distribution(N=10000,d=2,n_sample=10000*10/100)
        lg_out = LinearRegression(X_out,y_out)
        lg_out.learn()
        lg_out.is_complete()

        validator = Validator()
        validator.set(y_out,lg.assumption_apply(X_out))
        validator.is_complete()
        e_out += validator.get_predict_error_proportion()


    print "E(in)={0}, E(out)={1}".format(e_in/n_trial,e_out/n_trial)
    print "Best weight is {0}".format(np.round(w/1000,3))
#print hw1()
#hw2_q5()
hw2_nonlinear_transformation()