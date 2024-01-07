# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:19:59 2023

@author: Elizabeth Allan-Cole
"""
# abstract peak class. do not instancate directly. has atributes all peaks will have
class Peak(): 
    def __init__(self, name, q_min, q_max, chi_squared, sigma, amplitude): 
        self.name = name
        self.q_min = q_min 
        self.q_max = q_max 
        self.chi_squared = chi_squared 
        self.sigma = sigma 
        self.amplitude = amplitude
  
# this is the subclass of class "Class" 
class Graphite_Peak(Peak): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        
        
  
# driver code 
test = Graphite_Peak('frank', 1.75, 1.9, 500, 0.005, 5) 
print(test.name, test.amplitude, test.q_min)