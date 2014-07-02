from a1.LFD.LearningLine import LearningLine
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)


def hw1():
   # max_iter = 10000: mean iteration: 46.9583, n of error case 2865

    iter_used = []

    beta0,beta1 = np.random.uniform(-1,1,2)
    line_pop = LearningLine(beta0,beta1)

    line_sam = LearningLine(1,1)

    n = 1
    number_error_case = 0
    iter_used_to_coverage = 0
    for i in xrange(n):
        line_sam.set_iteration(100)
        X = np.random.uniform(-1,1,size=[100,2])
        line_sam.learn(line_pop,X)
        number_error_case += len(line_sam.missclassify(line_pop,X))
        iter_used_to_coverage += line_sam.get_interation_used()

    print "Mean of iterations to coverage:",iter_used_to_coverage/1000.0
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
    time.sleep(0.05)

hw1()