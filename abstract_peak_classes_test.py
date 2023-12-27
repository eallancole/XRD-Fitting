# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:19:59 2023

@author: Elizabeth Allan-Cole
"""
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import find_peaks

# abstract peak class. do not instancate directly. has atributes all peaks will have
class Peak(): 
    def __init__(self, name, q_min, q_max, chi_squared, sigma, amplitude): 
        self.name = name
        self.q_min = q_min 
        self.q_max = q_max 
        self.chi_squared = chi_squared 
        self.sigma = sigma 
        self.amplitude = amplitude
        
    
    def normalize_1d_array(self, arr):
        t_min = 0
        t_max = 10
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)    
        for i in arr:
            temp = (((i - min(arr))*diff)/diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr
    
    def my_find_peaks(self, data, height=None, threshold=None, distance=None, 
                      prominence=None, width=None, wlen=None, rel_height=0.5, 
                      plateau_size=None):
        return find_peaks(data, height, threshold, distance, prominence, width, wlen, rel_height, plateau_size)
  
    
 
class Graphite_LiC12(Peak): 
    def __init__(self,name = 'Graphite_LiC12', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = .005, amplitude = 5): 
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
    def find_best_sub_peak(self, new_y, cutoff = .5):
        new_y_norm = self.normalize_1d_array(new_y)
        best_val = cutoff
        best_peak_class = self
        # loop through subclasses and compare how the data fits
        for sub_class in Graphite_LiC12.__subclasses__():
            sub_df = pd.read_csv(sub_class.example_file_path)
            sub_df = sub_df[(sub_df['q'] >= self.q_min) & (sub_df['q'] <= self.q_max)]
            sub_y = np.array(sub_df.I) 
            sub_y_norm = self.normalize_1d_array(sub_y)
            
            # make sure each is same length
            if len(sub_y_norm) != len(new_y_norm):
                # use interpolation to make them the same lenth by god!!
                zoom_rate = sub_y_norm.shape[0] / new_y_norm.shape[0]
                new_y_norm = zoom(new_y_norm, zoom_rate)
                
            # compare the lines
            difference = np.subtract(new_y_norm, sub_y_norm)
            difference = np.absolute(difference)
            difference = np.sum(difference)
            difference = difference /(len(sub_y )* 10)
            if difference < best_val:
                best_val = difference
                best_peak_class = sub_class
            
        print('here', best_peak_class)
        return best_peak_class()
                

            
            
        
class Graphite_One_Big(Graphite_LiC12):
    example_file_path = r"Peak\Graphite_LiC12\Graphite_One_Big\fit_26.chi"
    
    def __init__(self,name = 'Graphite_One_Big', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = .005, amplitude = 5):
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
        
class Graphite_Small_Big(Graphite_LiC12):
    example_file_path = r"\Peak\Graphite_LiC12\Graphite_Small_Big\fit_72.csv"
    
    def __init__(self,name = 'Graphite_Small_Big', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = [0.002, .005], amplitude = [0.2, 3]):
        
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
    def my_find_peaks(self, data, height=None, threshold=None, distance=None, 
                      prominence=[None], width=None, wlen=None, rel_height=0.5, 
                      plateau_size=None):
        return find_peaks(data, height, threshold, distance, prominence, width, wlen, rel_height, plateau_size)
  
    
class Graphite_One_Small(Graphite_LiC12):
    example_file_path = r"\Peak\Graphite_LiC12\Graphite_One_Small\fit_32.csv"
    
    def __init__(self,name = 'Graphite_One_Small', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = .005, amplitude = 5):
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
  
    
  
  
    
# # driver code 
# test = Graphite_LiC12('frank', 1.75, 1.9, 500, 0.005, 5) 
# print('here', test.name)
# sub_classes = Graphite_LiC12.__subclasses__()
# #df = pd.read_csv(r"C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Class-testing\Peak\fit_66.csv")
# df = pd.read_csv(r"C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Class-testing\Peak\fit_28.csv")
# df = df[(df['q'] >= test.q_min) & (df['q'] <= test.q_max)]
# df.plot(x='q', y='I')
# print('returned', test.find_best_sub_peak(df))