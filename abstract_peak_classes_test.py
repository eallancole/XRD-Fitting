# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:19:59 2023

@author: Elizabeth Allan-Cole
"""
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths
import user_fit_operations as ufo
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import LinearModel, GaussianModel, ExponentialModel, ConstantModel, PowerLawModel, PolynomialModel, LorentzianModel, VoigtModel
from lmfit.model import save_modelresult, load_modelresult
import model_dict as md

DEBUG = True

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
      
    def index_to_xdata(self, xdata, indices):
         # interpolate the values from signal.peak_widths to xdata
         ind = np.arange(len(xdata))
         f = interp1d(ind,xdata)
         return f(indices)
 
    # get some inital values to describe the data. Peak centers, height width, ect.
    # very general subclasses can make this more specific.
    def intial_values(self, data, sliced_q, height=None, threshold=None, distance=None, 
                      prominence=1, width=None, wlen=None, rel_height=0.5, 
                      plateau_size=None):
        
 
        centers, vals = find_peaks(data, height = .1, prominence = .1)
        widths, width_heights, left_ips, right_ips = peak_widths(data, centers)
        
        widths = self.index_to_xdata(sliced_q, widths)
        left_ips = self.index_to_xdata(sliced_q, left_ips)
        right_ips = self.index_to_xdata(sliced_q, right_ips)
        
        keys_list = ['centers', 'sigmas', 'amplitudes']
        
        vals_list = [np.array(np.take(sliced_q, centers)), np.array((right_ips - left_ips)/2.35),np.array(vals['prominences'])]
        peak_vals = md.Model_dict()
        peak_vals.add_model(keys_list, vals_list)
        
        if DEBUG:
            print('DEBUGING IN INITAL VALUES FUNCTION')
            plt.plot(sliced_q,data)
            plt.plot(sliced_q[centers], data[centers], "x")
            plt.hlines(width_heights, left_ips, right_ips, color='r')
            plt.xlabel('x values')
            plt.ylabel('y values')
            plt.show() 
            
        # TODO add slope check to catch double peaks
            
        return peak_vals
     
    
    def make_all_modles(self, peak_vals):
        model_list = []
        for val in peak_vals.main_dict.values():
            model_list.append(self.make_model(val))
            
        return model_list
    
    def make_model(self, peak_vals):
        background = LinearModel(prefix=('b' + '_'))  
        pars = background.make_params()
        
        model = background
        
        # TODO make this better initial guesses     
        slope1 = 0 
        int1 = 50
        
        # For linear background
        pars = background.make_params()
        pars['b' + '_slope'].set(slope1)
        pars['b' + '_intercept'].set(int1)
        
        for i in range(len(peak_vals.values())):
            center = peak_vals['centers'][i]
            sigma = peak_vals['sigmas'][i]
            amplitude = peak_vals['amplitudes'][i]
            
            # create prefex for each peak
            pref = 'v'+str(i)+'_'   
            peak = VoigtModel(prefix=pref)
            
            # set the parimiters for each peak
            pars.update(peak.make_params())
            peak = VoigtModel(prefix=pref)
            pars[pref+'center'].set(value=center, min= center - 0.025, max= center + 0.025)
            pars[pref+'sigma'].set(value=sigma, max = sigma * 5)
            pars[pref+'amplitude'].set(amplitude, min = 0)
            pars[pref+'gamma'].set(value=sigma, vary=True, expr='', min = 0)
            
            model += peak

        return (model, pars)
    
    
    def check_fit(self, best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot):
        fit_good = True
        # check the chi squared
        if  best_model.chisqr >= self.chi_squared:
            fit_good = False
            print('Model chi squared too large. Model chi = %s chi limit = %s' % (best_model.chisqr, self.chi_squared))
        print('hi from super')
        return fit_good
            
    
 
class Graphite_LiC12(Peak): 
    def __init__(self,name = 'Graphite_LiC12', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = .005, amplitude = 5): 
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
    def find_best_sub_peak(self, new_y, cutoff = .75):
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
        return best_peak_class()
    
    
    def check_fit(self, best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot):
        fit_good = super().check_fit(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot) 
        print('hi from other')
        comps = best_model.eval_components(x=sliced_q)
        model_fwhm_list = []
        for prefex in comps.keys():
            if prefex != 'b_':
                model_fwhm_list.append(best_model.params[str(prefex)+'fwhm'].value)
        if max(model_fwhm_list) > 0.03:
            fig_good = False
        return fit_good

      

# def check_FWHM(sliced_q, best_model, peak_name):
#     model_fwhm_list = []
#     comps = best_model.eval_components(x=sliced_q)
    
#     for prefex in comps.keys():
#         if prefex != 'b_':
#             model_fwhm_list.append(best_model.params[str(prefex)+'fwhm'].value)
    
#     print('fwhm list: ', model_fwhm_list)
    
#     if peak_name == 'Graphite_LiC12':
#         while max(model_fwhm_list) > 0.03: 
#             print('FWHM too big')
            
#             return True
        
#     if peak_name == 'LiC6':
#         #If multiple borad peaks appear in the fit, go to user fit
#         max_fwhm = 0.03
#         if len(list(filter(lambda x: x >= max_fwhm, model_fwhm_list))) >= 2:
#             print('potato')
#             return True
        
#         while max(model_fwhm_list) > 0.075: 
            
#             print('FWHM too big')
#             return True
        
#     else:
#         return False      
            
        
class Graphite_One_Big(Graphite_LiC12):
    example_file_path = '/Users/benkupernk/Documents/GitHub/XRD-Fitting/Peak/Graphite_LiC12/Graphite_One_Big/fit_26.csv'
    
    def __init__(self,name = 'Graphite_One_Big', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = .005, amplitude = 5):
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
        
class Graphite_Small_Big(Graphite_LiC12):
    example_file_path = "/Users/benkupernk/Documents/GitHub/XRD-Fitting/Peak/Graphite_LiC12/Graphite_Small_Big/fit_72.csv"
    
    def __init__(self,name = 'Graphite_Small_Big', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = [0.002, .005], amplitude = [0.2, 3]):
        
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
    def my_find_peaks(self, data, height=None, threshold=None, distance=None, 
                      prominence=[None], width=None, wlen=None, rel_height=0.5, 
                      plateau_size=None):
        return find_peaks(data, height, threshold, distance, prominence, width,
                          wlen, rel_height, plateau_size)
  
# don't have right filepath and can't find example
# class Graphite_One_Small(Graphite_LiC12):
#     example_file_path = r"/Users/benkupernk/Documents/GitHub/XRD-Fitting/Peak/Graphite_LiC12/Graphite_One_Small/fit_32.csv"
    
#     def __init__(self,name = 'Graphite_One_Small', q_min = 1.75, q_max = 1.9, 
#                  chi_squared = 500, sigma = .005, amplitude = 5):
#         super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
class Squiggly_Line(Graphite_LiC12):
    example_file_path = r"/Users/benkupernk/Documents/GitHub/XRD-Fitting/Peak/Graphite_LiC12/squiggly_Line/fit_1.csv"
    
    def __init__(self,name = 'Squiggly_Line', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = [.005, .005], amplitude = [5, 5]):
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude) 
        print('the class is %s' % self.name)
        
    # its just background so make a line
    def make_model(self, peak_vals):
        background = LinearModel(prefix=('b' + '_'))  
        pars = background.make_params()
        
        model = background
        
        # TODO make this better initial guesses     
        slope1 = 0 
        int1 = 50
        
        # For linear background
        pars = background.make_params()
        pars['b' + '_slope'].set(slope1)
        pars['b' + '_intercept'].set(int1)
        return (model, pars)
    
    # again just background so who cares?
    def check_fit(self, best_model, sliced_q, sliced_I, sig, amp, q_max, q_min,
                  chisqu_fit_value, x_motor, y_motor, peak_name, plot):
        
        return True
        
        
    
  
  
    
# # # driver code 
# test = Graphite_LiC12('frank', 1.75, 1.9, 500, 0.005, 5) 
# print('here', test.name)
# sub_classes = Graphite_LiC12.__subclasses__()
# #df = pd.read_csv(r"C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Class-testing\Peak\fit_66.csv")
# df = pd.read_csv(r"C:\Users\benk\Documents\GitHub\XRD-Fitting\Peak\Graphite_LiC12\Squiggly_Line\fit_3.csv")
# df = df[(df['q'] >= test.q_min) & (df['q'] <= test.q_max)]
# df.plot(x='q', y='I')
# print('returned', test.find_best_sub_peak(np.array(df.I)))