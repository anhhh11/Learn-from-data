# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:28:10 2014

@author: anhhh11
"""
import Line
import numpy as np
class Simulation:
    @staticmethod
    def random_line_betas():
        two_points = np.random.uniform(low=-1,high=1,size=[1,4])
        intercept,slope = Line.Utils.two_points_to_beta(*(two_points.tolist()[0]))
        return (intercept,slope)
    