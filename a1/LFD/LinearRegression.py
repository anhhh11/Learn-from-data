# -*- coding: utf-8 -*-
from Learning import Learning
import numpy as np
class LinearRegression(Learning):
    def learn(self):
        super(LinearRegression,self).learn()
        if not hasattr(self,'manual_set_weights'):
            self.calculate_weight()
        self._config_validator()

    def set_weights(self,w=None):
        super(LinearRegression,self).set_weights(w)
        self.manual_set_weights = True

    def calculate_weight(self):
        self.w = np.dot(np.dot(np.linalg.pinv(np.dot(self.X.T,self.X)),
                          self.X.T),
                   self.y)

    def assumption_apply(self,X):
        if (self.w.shape[0] - X.shape[1]) == 1:
            X = self.add_threshold(X)
        assumption_result = np.dot(X,self.w)
        #(X*self.w).sum(axis=1)
        if np.unique(self.y).size==2:
            return np.sign(assumption_result)
        else:
            assumption_result
