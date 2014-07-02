# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:28:10 2014

@author: anhhh11
"""
import numpy as np 
from Validator import Validator
class Learning(object):    
    def __init__(self,X,y,validator=None):
        """
        @type self.validator: Validator
        """
        self.set_X(X)
        self.set_y(y)
        self.validator = validator or Validator()

    def learn(self):
        self._prepare()

    def assumption_apply(self,X):
        raise Exception("Not implemented")

    def predicts(self):
        raise Exception("Not implemented")

    def is_complete(self):
        return self.validator.is_complete()

    def model(self):
        return [self.X[:,0:-1],self.y]

    def result(self):
        return [self.predicts,self.w]
        
    def weight_result(self):
        return self.w

    def set_weights(self,w=None):
        self.w = w        
        
    def set_X(self,X):
        self._check_mismatch_size_errors_X_y()        
        self.X = X        
        
    def set_y(self,y):
        self._check_mismatch_size_errors_X_y()
        self.y = y

    def _prepare(self):
        if not hasattr(self,'added_threshold'):
            self.X = self.add_threshold(self.X)
            self.added_threshold = True

    def _check_mismatch_size_errors_X_y(self):
        if not (hasattr(self,'X') and hasattr(self,'y')):
            return
        if (self.X.shape[0] != self.y.size):
            raise Exception("""
            Numbers of rows are mismatch, X has {0} rows but y has {1}
            """.format(self.X.shape[0],self.y.size))
    

    def _update_weights(self,random_wrong_case):
        raise Exception("Not implemented")
        
    def add_threshold(self,X):
        return np.column_stack((np.repeat(1,X.shape[0]),X))
              
    def _config_validator(self):        
        self.validator.set(self.y,self.assumption_apply(self.X))

