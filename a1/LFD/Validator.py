# -*- coding: utf-8 -*-
import numpy as np
class Validator(object):
    def set(self,y,predicts):
        self.y = y
        self.predicts = predicts
        self.checked = False
        
    def is_complete(self):
        if not self.checked:
            self.checked = True
            self.wrong_predicted_cases_poisitions = np.where(self._check() == False)[0]
        return self.get_number_wrong_predicted_cases()==0

    def get_wrong_predicted_cases_positions(self):
        return self.wrong_predicted_cases_poisitions

    def get_number_wrong_predicted_cases(self):
        return self.get_wrong_predicted_cases_positions().size        

    def get_predict_error_proportion(self):
        return self.get_number_wrong_predicted_cases()/float(self.y.size)
        
    def _check(self):
        return (self.predicts == self.y)

