__author__ = 'anhhh11'
import numpy as np
import itertools
import sympy
from a1.LFD.Perceptron import Perceptron
from math import exp
import math
def hoeffding_inequality(M=None,N=None,epsilon=None,most=None,solve=""):
    """

    @param M: number of hypothesis
    @param N: number of examples
    @param most: max probability
    @param solve: params want to solve
    @return:
    """
    M = M or sympy.Symbol('M')
    N = N or sympy.Symbol('N')
    epsilon = epsilon or sympy.Symbol('epsilon')
    most = most or sympy.Symbol('most')

    result = sympy.solve(2*M*sympy.exp(-2*(epsilon**2)*N) - most,eval(solve))
    return result


def find_breakpoint(dim=3,start_N = 0,tries_time=10,limit_N = 100):
    d = dim
    N = start_N
    n_tries = tries_time
    all_coverage = True
    while True:
        for i in range(n_tries):
            X = np.random.uniform(-1,1,size=[N,d])
            dichotomies = np.array([ a for a in itertools.product([-1,1],repeat=N)])
            all_coverage = True
            for j in range(dichotomies.shape[0]):
                pla = Perceptron(X,dichotomies[j,:])
                pla.learn()
                if not pla.is_complete():
                    #print pla.validator.get_predict_error_proportion()
                    all_coverage = False
                    break
        if all_coverage and N < limit_N:
            N += 1
            continue
        elif N >= limit_N:
            print "Cannot find particular N within [{0},{1}]".format(0,limit_N)
            return 2**(N-1)
        else:
            break
    return N

#Ques 1
print hoeffding_inequality(M=1,epsilon=0.05,most=0.03,solve="N")
#Ques 2
print hoeffding_inequality(M=10,epsilon=0.05,most=0.03,solve="N")
#Ques 3
print hoeffding_inequality(M=100,epsilon=0.05,most=0.03,solve="N")
#Ques 4
print "Breakpoint is {0}".format(find_breakpoint(dim=3,tries_time=20))

