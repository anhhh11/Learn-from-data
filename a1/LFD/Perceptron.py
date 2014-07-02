# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:28:10 2014

@author: anhhh11
"""
import numpy as np
from Learning import Learning
class Perceptron(Learning):    
    def __init__(self,X,y,iteration=1000):
        super(Perceptron,self).__init__(X,y)
        self.set_weight()
        self.iteration = iteration or 0
        
    def set_weight(self,w=None):
        def check_mismatch_size_errors_X_w(X,w):
            if not (hasattr(self,'X') and hasattr(self,'w')):
                return        
                if (self.X.shape[1] != self.w.size):
                    raise Exception("""
                    Numbers of rows are mismatch, X has {0} columns but w has {1}
                    """.format(self.X.shape[1],self.w.size))
        if not isinstance(w,np.ndarray):
            super(Perceptron,self).set_weights(np.repeat(0.0,self.X.shape[1]))
        else:
            super(Perceptron,self).set_weights(w)
            check_mismatch_size_errors_X_w(self.X,self.w)
            self.setted_weight = True

    def learn(self,callback=None):
        super(Perceptron,self).learn()
        accuracy = 0.0
        for i in xrange(self.iteration):
            applies = self.assumption_apply(self.X)
            self.validator.set(self.y,applies)
            if (self.validator.is_complete()): 
                break
            if self.validator.get_predict_error_proportion() > accuracy:
                self._update_weights(self._random_wrong_case_position())
            if callback:
                callback(self.w)
        self._update_iteration_used(i)
    
    def assumption_apply(self,X):
        return np.sign((X*self.w).sum(axis=1))
        
    def _random_wrong_case_position(self):
        return np.random.choice(self.validator.get_wrong_predicted_cases_positions(),
                                         size=1,
                                         replace=False)[0]

    def _update_weights(self,random_wrong_case):
        self.w += self.X[random_wrong_case]*self.y[random_wrong_case]                                     

    def _prepare(self):
        super(Perceptron,self)._prepare()
        if not hasattr(self,'setted_weight'):
            self.w = np.insert(self.w,0,1)

    def _choose_init_weights(self):
        self.w = self.X[np.random.choice(self.X.shape[0],size=1,replace=False)[0]]

    def _update_iteration_used(self,iteration_stop_index):
        self.iteration_used = iteration_stop_index + 1
        
