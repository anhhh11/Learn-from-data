# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
class Utils:
    @staticmethod
    def two_points_to_beta(x0,y0,x1,y1):
        """
        Returns (intercept,slope)
        """
        slope = (y1 - y0)/(x1-x0)
        intercept = y1 - slope*x1
        return (intercept,slope)
    @staticmethod        
    def abline(intercept,slope,x):
        #x = np.linspace(0,1,1000)
        y = intercept + x*slope
        plt.plot(x,y)
    