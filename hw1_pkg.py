import sklearn

__author__ = 'anhhh11'
from sklearn import linear_model
from a1.LFD.LearningLine import LearningLine
import numpy as np

np.random.seed(1234)
beta0,beta1 = np.random.uniform(-1,1,2)
line_pop = LearningLine(beta0,beta1)
X = np.random.uniform(-1,1,size=[100,2])
y = line_pop.classify(X)

percept = linear_model.Perceptron()
percept.fit(X,y)
print percept.coef_
print percept.intercept_
print percept.score(X,y)
print percept.get_params()