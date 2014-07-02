# -*- coding: utf-8 -*-
import numpy as np
import copy
from numpy import ndarray
class LearningLine(object):
    def __init__(self,beta0=1,beta1=1):
        self.beta0 = float(beta0)
        self.beta1 = float(beta1)
        self.w = self.betas_to_weights()
        self.iteration_used = 0

    def betas_to_weights(self):
        beta0,beta1 = self.beta0,self.beta1
        if beta0 == 0:
            return np.array([0,-beta1,1])
        else:
            return np.array([1,
                             beta1/beta0,
                             -1.0/beta0])
    def get_betas(self):
        w = self.get_weight()
        return np.array([-w[0]/w[2],
                         -w[1]/w[2]
                         ])

    def get_weight(self):
        return np.copy(self.w)


    def missclassify(self, line,X):
        """


        @rtype : tuple
        @type line: LearningLine
        @type X: ndarray
        """
        return np.array(np.where(self.classify(X)==line.classify(X)))

    def missclassify_proportion(self,line,X):
        return float(self.missclassify(line,X).size) / X.shape[0]

    def classify(self, X):
        """

        @type X: ndarray or list
        """
        if not isinstance(X,ndarray):
            raise Exception("X must be numpy array")
        if X.size == X.shape[0]:
            X = np.array([X])
        X = np.column_stack((np.repeat(1,X.shape[0]),X))
        return np.sign(np.dot(X,self.w))

    def get_interation_used(self):
        return self.iteration_used

    def set_iteration(self,iter):
        self.iteration = iter
        self.iteration_used = 0

    def adjust_inplace(self,x):
        self.w += x


    def adjust(self,x):
        """
        @type x: ndarray
        """
        copied_self = LearningLine(self.beta0,
                                    self.beta1)
        copied_self.adjust_inplace(x)
        return copied_self

    def learn(self,line,X):

        while self.iteration_used < self.iteration:
            missclassify = self.missclassify(line,X)[0]

            if missclassify.size == 0:
                return True

            a = np.random.choice(missclassify,size=1,replace=False)[0]
            random_case = X[a,:]
            random_case = np.insert(random_case,0,1)
            self.adjust_inplace(random_case)

            self.iteration_used += 1
        return False