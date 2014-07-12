import decimal

__author__ = 'anhhh11'
from sympy import Symbol,solve,exp
from hw3 import grow_function
import math
import sympy
import numpy as np

def grow_function_sym(N = Symbol("N"),k = 0):
    """
    @type k int
    """
    from sympy import factorial,summation,combsimp
    ksym = Symbol("ksym")

    def C(k,n):
        return factorial(n) / (factorial(k)*factorial(n-k))
    return combsimp(summation(C(ksym,N),(ksym,0,k-1)))


def vc_inequality(epsilon=None,d_vc=None,at_most=None,N=None):
    N = N or Symbol('N')
    epsilon = epsilon or Symbol('epsilon')
    eq = 4*((2*N)**d_vc)*exp((-1.0/8)*(epsilon**2)*N)
    return eq

def vc_inequality_test(epsilon=None,d_vc=None,at_most=None,N_test = []):
    """

    @param : epsilon Confident interval error rate
    @param: d_vc max number of point H can shatter
    @param: N number of point
    @param: at_most P(error rate >= epsilon) <= at_most
    """
    N = Symbol("N")
    eq = vc_inequality(epsilon,d_vc,at_most)
    cmp = map(lambda x: abs(eq.subs(N,x) - at_most),N_test)
    min_pos = cmp.index(min(cmp))
    return (N_test[min_pos],min_pos)

#Ques 1
N_test = [400000,420000,440000,460000,480000]
r = vc_inequality_test(d_vc=10, at_most=0.05,epsilon=0.05,N_test = N_test)
print(r)

#Ques 2
from math import exp,sqrt,log
N = decimal.Decimal(1)
d_vc = 50
at_most = decimal.Decimal(.05)
epsilon = decimal.Decimal(.05)
mN = N**d_vc
m2N = (2*N)**d_vc
mNp2 = decimal.Decimal((N**2)**d_vc)
def log(a):
    return decimal.Decimal(math.log(a))
def sqrt(a):
    return decimal.Decimal(math.sqrt(a))
original_vc_bound = sqrt(8/N*log(4*m2N/at_most))
rademacher_penatly_bound = sqrt(2*log(2*N*mN) / N) + sqrt(2/N*log(1/at_most)) + 1/N
parrondo_vandenbroek = sqrt(1/N*(2*epsilon + log(6*m2N / at_most)))
devroye = sqrt((1/(2*N))*(4*epsilon*(1+epsilon)+(log(4*mNp2/at_most))))

bounds = [original_vc_bound,rademacher_penatly_bound,parrondo_vandenbroek,devroye]
#parrondo_vandenbroek

print(map(lambda x: x ,bounds)) #devroye

#Ques 3
#devroye
