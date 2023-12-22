# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:19:59 2023

@author: Elizabeth Allan-Cole
"""
# abstract peak class. do not instancate directly. has atributes all peaks will have
class peak():
    def __inti__(self, name, center_list, promainnce_list, sig_list, amp_list, chisqu_fit_value):
        self.name = name
        self.center_list = center_list
        self.promainnce_list = promainnce_list
        self.sig_list = sig_list
        self.amp_list = amp_list
        self.chisqu_fit_value = chisqu_fit_value
        