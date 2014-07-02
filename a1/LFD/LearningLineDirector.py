__author__ = 'anhhh11'
from LearningLine import LearningLine
class LearningLineDirector(object):
    def __init__(self,iteration=100):
        self.iteration = iteration
        self.iteration_used = 0
        self.learningLine = LearningLine()

    def set_data(self,X,y):
        self.X = X
        self.y = y

    def get_interation_used(self):
        return self.iteration_used

    def set_iteration(self,iter):
        self.iteration = iter

    def adjust_inplace(self,x):
        if self.iteration_used > self.iteration:
            return
        self.w += x
        self.iteration_used += 1

    def missclassify_to_result(self,X,y):
        return np.where(self.classify(X)==y)
