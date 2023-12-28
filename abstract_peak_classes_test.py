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
      
    
    # get some inital values to describe the data. Peak centers, height width, ect.
    # very general subclasses can make this more specific.
    def intial_values(self, data, sliced_q, height=None, threshold=None, distance=None, 
                      prominence=1, width=None, wlen=None, rel_height=0.5, 
                      plateau_size=None):
        def index_to_xdata(xdata, indices):
            # interpolate the values from signal.peak_widths to xdata
            ind = np.arange(len(xdata))
            f = interp1d(ind,xdata)
            return f(indices)
        
        peaks, vals = find_peaks(data, height = .1, prominence = .1)
        widths, width_heights, left_ips, right_ips = peak_widths(data, peaks)
        
        widths = index_to_xdata(sliced_q, widths)
        left_ips = index_to_xdata(sliced_q, left_ips)
        right_ips = index_to_xdata(sliced_q, right_ips)
        peak_vals = {}
        peak_vals['centers'] = np.array(np.take(sliced_q, peaks))
        peak_vals['sigmas'] = np.array((right_ips - left_ips)/2.35)
        peak_vals['amplitude'] = np.array(vals['prominences'])
        
        if DEBUG:
            print('DEBUGING IN INITAL VALUES FUNCTION')
            plt.plot(sliced_q,data)
            plt.plot(sliced_q[peaks], data[peaks], "x")
            plt.hlines(width_heights, left_ips, right_ips, color='r')
            plt.xlabel('x values')
            plt.ylabel('y values')
            plt.show() 
            
        return peak_vals
     
    
    def make_models(self, q_max, q_min, model_centers, sig, amp):
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
        
        
        # background = PolynomialModel(prefix=('b' + '_'))
        # pars = background.make_params()
        
        # model = background
        
        # # initial guesses     
        # a = 1
        # b = 1
        # c = 1
        # pars = background.make_params()
        # pars['b' + '_c0'].set(a)
        # pars['b' + '_c1'].set(b)
        # pars['b' + '_c2'].set(b)
        
          
        for peak, center in enumerate(model_centers):
            # create prefex for each peak
            pref = 'v'+str(peak)+'_'
            #peak = GaussianModel(prefix=pref)
            peak = VoigtModel(prefix=pref)
            # set the parimiters for each peak
            pars.update(peak.make_params())
            #pars[pref+'center'].set(value=center, min=q_min, max=q_max)
            pars[pref+'center'].set(value=center, min= center - 0.025, max= center + 0.025)
            pars[pref+'sigma'].set(value=sig, max = sig * 5)
            pars[pref+'amplitude'].set(amp, min = 0)
            pars[pref+'gamma'].set(value=sig, vary=True, expr='', min = 0)
            
            model = model + peak

        return (model, pars)
            

        
    #using the inital values from the intial_values function make some lmfit models to run
    def get_lmfit_models(self, peak_vals):

        num_peaks = len(peak_vals['centers'])
        new_center_list = []
        
        # Creates target gueses close to the identified peaks (+/- 10% sigma away from center) 
        for center in range(num_peaks):
            new_center_list.append(ufo.make_center_list(center_list[center], self.sigma))
        
        new_center_list = ufo.iterate_centers(new_center_list)
        
        model_list = get_prom_model_list(self.q_max, self.q_min, new_center_list, self.sigma, self.amplitude, self.name)
      
    
 
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
        return best_peak_class()
                

            
            
        
class Graphite_One_Big(Graphite_LiC12):
    example_file_path = r"C:\Users\benk\Documents\GitHub\XRD-Fitting\Peak\Graphite_LiC12\Graphite_One_Big\fit_26.csv"
    
    def __init__(self,name = 'Graphite_One_Big', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = .005, amplitude = 5):
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
        
class Graphite_Small_Big(Graphite_LiC12):
    example_file_path = r"C:\Users\benk\Documents\GitHub\XRD-Fitting\Peak\Graphite_LiC12\Graphite_Small_Big\fit_72.csv"
    
    def __init__(self,name = 'Graphite_Small_Big', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = [0.002, .005], amplitude = [0.2, 3]):
        
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
    def my_find_peaks(self, data, height=None, threshold=None, distance=None, 
                      prominence=[None], width=None, wlen=None, rel_height=0.5, 
                      plateau_size=None):
        return find_peaks(data, height, threshold, distance, prominence, width, wlen, rel_height, plateau_size)
  
    
class Graphite_One_Small(Graphite_LiC12):
    example_file_path = r"C:\Users\benk\Documents\GitHub\XRD-Fitting\Peak\Graphite_LiC12\Graphite_One_Small\fit_32.csv"
    
    def __init__(self,name = 'Graphite_One_Small', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = .005, amplitude = 5):
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
class Squiggly_Line(Graphite_LiC12):
    example_file_path = r"C:\Users\benk\Documents\GitHub\XRD-Fitting\Peak\Graphite_LiC12\Squiggly_Line\fit_3.csv"
    
    def __init__(self,name = 'Graphite_One_Small', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = [.005, .005], amplitude = [5, 5]):
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude) 
        print('the class is %s' % self.name)
        
    
  
  
    
# # # driver code 
# test = Graphite_LiC12('frank', 1.75, 1.9, 500, 0.005, 5) 
# print('here', test.name)
# sub_classes = Graphite_LiC12.__subclasses__()
# #df = pd.read_csv(r"C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Class-testing\Peak\fit_66.csv")
# df = pd.read_csv(r"C:\Users\benk\Documents\GitHub\XRD-Fitting\Peak\Graphite_LiC12\Squiggly_Line\fit_3.csv")
# df = df[(df['q'] >= test.q_min) & (df['q'] <= test.q_max)]
# df.plot(x='q', y='I')
# print('returned', test.find_best_sub_peak(np.array(df.I)))