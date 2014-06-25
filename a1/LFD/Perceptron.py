# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:28:10 2014

@author: anhhh11
"""
import numpy as np
class Perceptron:    
    def __init__(self,X,y,iteration=1000):
        self.set_X(X)
        self.set_y(y)
        self.set_w()
        self.iteration = iteration

    def set_X(self,X):
        self.__check_mismatch_size_errors_X_y()        
        self.X = X        
        
    def set_y(self,y):
        self.__check_mismatch_size_errors_X_y()
        self.y = y
        
    def set_w(self,w=None):
        if not w:
            self.w = np.repeat(0.0,self.X.shape[1])
        else:        
            self.w = w
            self.__check_mismatch_size_errors_X_w()

    def __check_mismatch_size_errors_X_y(self):
        if not (hasattr(self,'X') and hasattr(self,'y')):
            return
        if (self.X.shape[0] != self.y.size):
            raise Exception("""
            Numbers of rows are mismatch, X has {0} rows but y has {1}
            """.format(self.X.shape[0],self.y.size))

    def __check_mismatch_size_errors_X_w(self):
        if not (hasattr(self,'X') and hasattr(self,'w')):
            return        
        if (self.X.shape[1] != self.w.size):
            raise Exception("""
            Numbers of rows are mismatch, X has {0} columns but w has {1}
            """.format(self.X.shape[1],self.w.size))
        
    def __predicts(self):
        self.predicts =  np.sign((self.X*self.w).sum(axis=1))
        self.wrong_predicted_cases_poisitions = self.__wrong_predicted_cases_positions()

    def __predicted_result_verification(self):
        return (self.predicts == self.y)

    def __wrong_predicted_cases_positions(self):
        return np.where(self.__predicted_result_verification() == False)[0]

    def __random_wrong_case_position(self):
        return np.random.choice(self.wrong_predicted_cases_poisitions,
                                         size=1,
                                         replace=False)[0]
    def __update_weights(self,random_wrong_case):
        self.w += self.X[random_wrong_case]*self.y[random_wrong_case]                                     

    def __add_threshold(self):
        self.X = np.column_stack((self.X,np.repeat(1,self.X.shape[0])))
        self.w = np.append(self.w,1)
    
    def __choose_init_weights(self):
        self.w = self.X[np.random.choice(self.X.shape[0],size=1,replace=False)[0]]
        
    def is_complete(self):
        return self.wrong_predicted_cases_poisitions.size==0

    def wrong_cases_remain(self):
        """
        Wrong cases remain
        
        -----
        Results
        [ Wrong case positions, number of wrong cases]
        """
        return [self.wrong_predicted_cases_poisitions,
                self.wrong_predicted_cases_poisitions.size]
    def learn(self,callback=None):
        self.__choose_init_weights()
        if not hasattr(self,'added_threshold'):
            self.__add_threshold()
            self.added_threshold = True
        for i in xrange(self.iteration):
            self.__predicts()
            if (self.is_complete()): break
            self.__update_weights(self.__random_wrong_case_position())
            if callback:
                callback(self.w)
    def model(self):
        """
        Get model
        
        ------
        Returns
        [X,y]
        """
        return [self.X[:,0:-1],self.y]

    def result(self):
        return [self.predicts,self.w]
        
    def weight_result(self):
        return self.w

