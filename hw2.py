# -*- coding: utf-8 -*-
import numpy as np
import numba as nb
from multiprocessing import Pool
np.random.seed(1234)

def experiment():
    time_flip_per_coin = 10    
    head_prop = 0.5
    num_coints = 1000
    freqs = np.empty([1000],dtype=float)
    for i in range(num_coints):
        freqs[i] = np.mean(np.random.binomial(n=1,p=head_prop,size=time_flip_per_coin))    
    return np.array([freqs.min(),
                         freqs[0],
                        np.random.choice(freqs,size=1,replace=False)[0]])

#@nb.jit(nb.f8[:]())
results = np.empty([3],dtype=float)

def sum_result(r):
    global results
    results += r
    return results
    
def main():
    pool = Pool(4)
    num_trails = 1000
    for j in range(num_trails):
        pool.apply_async(experiment,callback=sum_result)
    pool.close()    
    pool.join()
    global results
    results /= num_trails
    return np.round(results,4)
print main()