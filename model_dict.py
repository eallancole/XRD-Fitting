#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:41:26 2023

@author: benkupernk
"""
import numpy as np

class Model_dict():
    def __init__(self):
        self.main_dict = {}
        
    def add_model (self, keys_list, vals_list):
            i = str(len(self.main_dict.keys()))
            model_name = 'model_%s' % i
            self.main_dict[model_name] = {}
            
            for key, val in zip(keys_list, vals_list):
                self.main_dict[model_name][key] = val
                
    def order_by_val(self, val = 'amplitudes'):
        for model in self.main_dict.values():
            if val in model.keys():
                sort_idx = list(np.argsort(model.get(val)))
                sort_idx.reverse()
            for key, val in model.items():
                model[key] = [val[i] for i in sort_idx]