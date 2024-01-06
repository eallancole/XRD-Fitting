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
import user_fit_operations_test as ufo
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import LinearModel, GaussianModel, PolynomialModel, LorentzianModel, VoigtModel, PseudoVoigtModel
from lmfit.model import save_modelresult, load_modelresult
import model_dict as md

DEBUG = False

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
        
 
        centers, vals = find_peaks(data, prominence = 1)
        prom_array = np.array(vals['prominences'])
        print('Prominences: ', prom_array)
        
        keys_list = ['centers', 'sigmas', 'amplitudes', 'prominences']
        
        # Get guesses for FWHM of peaks using scipy
        widths_half, width_heights_half, left_ips_half, right_ips_half = peak_widths(data, centers, rel_height= 0.5)
        
        widths_half = self.index_to_xdata(sliced_q, widths_half)
        left_ips_half = self.index_to_xdata(sliced_q, left_ips_half)
        right_ips_half = self.index_to_xdata(sliced_q, right_ips_half)
        
        # Make array for sigma guesses (uses FWHM guesses)
        sig_guess_array = np.array((right_ips_half - left_ips_half)/2.35) #Sigma is approximately sqrt(8*ln(2)) = 2.35
        # print('Sigma: ', sig_guess_array)
        # Cacluate the width at the base of the peak to make amplitude guesses
        widths_base, width_heights_base, left_ips_base, right_ips_base = peak_widths(data, centers, rel_height = 0.9)
        
        widths_base = self.index_to_xdata(sliced_q, widths_base)
        left_ips_base = self.index_to_xdata(sliced_q, left_ips_base)
        right_ips_base = self.index_to_xdata(sliced_q, right_ips_base)
        
        # Calculate the area of a triangle to appoximate peak amplitude (0.5 * base * prominence)
        amp_guess_array = np.array(0.5 * (right_ips_base - left_ips_base) * prom_array)  
        # TODO figure out a if we need a ~50% ish fudge factor on triangle area, in case amplitude does not quite match up
        # print('Amplitude: ', amp_guess_array)
        
        # List of values for model discutionary: center array, signma guess array, amplitude guess array 
        vals_list = [np.array(np.take(sliced_q, centers)), sig_guess_array, amp_guess_array, prom_array]
        # TODO Consider just using a dictionary here
       # peak_vals = md.Model_dict()
        #peak_vals.add_model(keys_list, vals_list)
        
        peak_vals = {}
        for key, val in zip(keys_list, vals_list):
            peak_vals[key] = val
        
        if DEBUG:
            print('DEBUGING IN INITAL VALUES FUNCTION')
            plt.plot(sliced_q,data)
            plt.plot(sliced_q[centers], data[centers], "x")
            plt.hlines(width_heights_half, left_ips_half, right_ips_half, color='r')
            plt.hlines(width_heights_base, left_ips_base, right_ips_base, color='b')
            plt.xlabel('x values')
            plt.ylabel('y values')
            plt.show() 
            
        # TODO add slope check to catch double peaks
            
        return peak_vals
    
    def secondary_values(self, peak_vals, data, sliced_q):
        return peak_vals
     
    
    def make_all_modles(self, peak_vals):
        model_list = []
        #for val in peak_vals.values():
        model_list.append(self.make_model(peak_vals))
            
        return model_list
    
    def make_model_background(self, peak_vals):
        # TODO Fix this code so that the models that are created are better
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
        
    def make_model_peaks(self, peak_vals, model, pars):
        
        for i in range(len(peak_vals['centers'])):
            center = peak_vals['centers'][i]
            sigma = peak_vals['sigmas'][i]
            amplitude = peak_vals['amplitudes'][i]
        
            # create prefex for each peak
            pref = 'v'+str(i)+'_'
            peak = PseudoVoigtModel(prefix=pref)  #Use for a Pseudo Voigt Model
            # peak = VoigtModel(prefix=pref) #Use for a Voigt Model
            
            # set the parimiters for each peak
            pars.update(peak.make_params())
            pars[pref+'center'].set(value = center, min= center - 0.025, max= center + 0.025)
            pars[pref+'sigma'].set(value = sigma, max = sigma * 1.25) #reduce this guess after scipy fixes! (1.2 maybe?)
            pars[pref+'amplitude'].set(value = amplitude, min = 0, max = amplitude * 2)
            #pars[pref+'gamma'].set(value=sigma, vary=True, expr='', min = 0) #Use for a Voigt Model
            pars[pref+'fraction'].set(value=0.5, vary=True) #Use for a Pseudo Voigt Model

            model += peak

        return (model, pars)
    
    def make_model(self, peak_vals):
        background_model, pars = self.make_model_background(peak_vals)
        
        model, pars = self.make_model_peaks(peak_vals, background_model, pars)
        
        return (model, pars)
        
    
    # def make_model(q_max, q_min, model_centers, sig_list, amp_list, peak_name):
    #     if 'Graphite_LiC12' not in peak_name: 
    #         background = LinearModel(prefix=('b' + '_'))  
    #         pars = background.make_params()
            
    #         model = background
            
    #         # initial guesses     
    #         slope1 = 0 
    #         int1 = 50
            
    #         # For linear background
    #         pars = background.make_params()
    #         pars['b' + '_slope'].set(slope1)
    #         pars['b' + '_intercept'].set(int1)
        
    #     else: 
    #         background = PolynomialModel(degree = 3, prefix=('b' + '_'))
    #         pars = background.make_params()
            
    #         model = background
            
    #         # initial guesses     
    #         a, b, c, d = 1, 1, 1, 1
    #         pars = background.make_params()
    #         pars['b' + '_c0'].set(value = a)
    #         pars['b' + '_c1'].set(value = b)
    #         pars['b' + '_c2'].set(value = c)
    #         pars['b' + '_c3'].set(value = d)
        
          
    #     for index, center in enumerate(model_centers):
    #         # create prefex for each peak
    #         pref = 'v'+str(index)+'_'
    #         peak = PseudoVoigtModel(prefix=pref)  #Use for a Pseudo Voigt Model
    #         # peak = VoigtModel(prefix=pref) #Use for a Voigt Model
    #         # set the parimiters for each peak
    #         pars.update(peak.make_params())
    #         #pars[pref+'center'].set(value=center, min=q_min, max=q_max)
    #         pars[pref+'center'].set(value=center, min= center - 0.025, max= center + 0.025)
    #         pars[pref+'sigma'].set(value=sig_list[index], max = sig_list[index] * 2) #reduce this guess after scipy fixes! (1.2 maybe?)
    #         pars[pref+'amplitude'].set(value = amp_list[index], min = 0)
    #         #pars[pref+'gamma'].set(value=sig, vary=True, expr='', min = 0) #Use for a Voigt Model
    #         pars[pref+'fraction'].set(value=0.5, vary=True) #Use for a Pseudo Voigt Model
            
    #         model = model + peak
    #     return (model, pars)
    
    # def check_FWHM(self, sliced_q, best_model, peak_name):
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
    
    
    def check_fit(self, best_model, sliced_q, data):
        fit_good = True
        # check the chi squared
        if  best_model.chisqr >= self.chi_squared:
            fit_good = False
            print('Model chi squared too large. Model chi = %s chi limit = %s' % (best_model.chisqr, self.chi_squared))
        #print('hi from super')
        

        return fit_good
    
 
class Graphite_LiC12(Peak): 
    def __init__(self,name = 'Graphite_LiC12', q_min = 1.75, q_max = 1.9, 
                 chi_squared = 500, sigma = .005, amplitude = 5): 
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
    def make_model_background(self, peak_vals):
        
        background = PolynomialModel(degree = 3, prefix=('b' + '_'))
        pars = background.make_params()
        
        model = background
        
        # initial guesses     
        a, b, c, d = 1, 1, 1, 1
        pars = background.make_params()
        pars['b' + '_c0'].set(value = a)
        pars['b' + '_c1'].set(value = b)
        pars['b' + '_c2'].set(value = c)
        pars['b' + '_c3'].set(value = d)
        
        if DEBUG:
            print('Fitting Graphite with polynomial background')
        
        return (model, pars)
    
    def check_sub_class_fit(self, peak_vals, sliced_q, data):
        return True
    
    def find_best_sub_peak(self, peak_vals, sliced_q, data, cutoff = .25):
        new_y_norm = np.array(self.normalize_1d_array(data))
        best_val = cutoff
        best_peak_class = 'default'
        # loop through subclasses and compare how the data fits
        for sub_class in Graphite_LiC12.__subclasses__():
            sub_df = pd.read_csv(sub_class.example_file_path)
            sub_df = sub_df[(sub_df['q'] >= self.q_min) & (sub_df['q'] <= self.q_max)]
            sub_y = np.array(sub_df.I) 
            sub_y_norm = np.array(self.normalize_1d_array(sub_y))
            
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
            if DEBUG:
                print('sunbclass = %s diffrence = %s' % (sub_class, difference))
            
            if difference < best_val and sub_class().check_sub_class_fit(peak_vals, sliced_q, data):
                best_val = difference
                best_peak_class = sub_class
        if best_peak_class == 'default':
            return self
        else:
            return best_peak_class()
    
    
    def check_fit(self, best_model, sliced_q, data, peak_vals):
        
        # Increase the chi squared for very large peaks         
        if len(peak_vals['prominences']) != 0 and max(peak_vals['prominences']) > 1000:
            self.chi_squared = 5000
            
        fit_good = super().check_fit(best_model, sliced_q, data)         
        
        comps = best_model.eval_components(x=sliced_q)
        model_fwhm_list = []
        for prefex in comps.keys():
            if prefex != 'b_':
                model_fwhm_list.append(best_model.params[str(prefex)+'fwhm'].value)
        if len(model_fwhm_list) == 0:
            fit_good = True
            print('Just background Bitches')
        
        elif max(model_fwhm_list) > 0.03:
            fit_good = False
            print('The FWHM is too big!')

        
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
            
        
# class Graphite_One_Big(Graphite_LiC12):
#     example_file_path = r'C:\Users\Elizabeth Allan-Cole\Documents\GitHub\XRD-Fitting\Peak\Graphite_LiC12\Graphite_One_Big\fit_26.csv'
    
#     def __init__(self,name = 'Graphite_One_Big', q_min = 1.75, q_max = 1.9, 
#                  chi_squared = 500, sigma = .005, amplitude = 5):
#         super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
        
# class Graphite_Small_Big(Graphite_LiC12):
#     example_file_path = r'C:\Users\Elizabeth Allan-Cole\Documents\GitHub\XRD-Fitting\Peak\Graphite_LiC12\Graphite_Small_Big\fit_72.csv'
    
#     def __init__(self,name = 'Graphite_Small_Big', q_min = 1.75, q_max = 1.9, 
#                  chi_squared = 500, sigma = [0.002, .005], amplitude = [0.2, 3]):
        
#         super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
#     # def my_find_peaks(self, data, height=None, threshold=None, distance=None, 
#     #                   prominence=[None], width=None, wlen=None, rel_height=0.5, 
#     #                   plateau_size=None):
#     #     return find_peaks(data, height, threshold, distance, prominence, width,
#     #                       wlen, rel_height, plateau_size)
    
    
# class Graphite_Big_Small(Graphite_LiC12):
#     example_file_path = r'C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Class-testing\Peak\Graphite_LiC12\Graphite_Big_Small\data_0.csv'
    
#     def __init__(self,name = 'Graphite_Big_Small', q_min = 1.75, q_max = 1.9, 
#                  chi_squared = 500, sigma = [0.005, .002], amplitude = [3, 0.1]):
        
#         super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
    # def my_find_peaks(self, data, height=None, threshold=None, distance=None, 
    #                   prominence=[None], width=None, wlen=None, rel_height=0.5, 
    #                   plateau_size=None):
    #     return find_peaks(data, height, threshold, distance, prominence, width,
    #                       wlen, rel_height, plateau_size)
  
# don't have right filepath and can't find example
# class Graphite_One_Small(Graphite_LiC12):
#     example_file_path = r"/Users/benkupernk/Documents/GitHub/XRD-Fitting/Peak/Graphite_LiC12/Graphite_One_Small/fit_32.csv"
    
#     def __init__(self,name = 'Graphite_One_Small', q_min = 1.75, q_max = 1.9, 
#                  chi_squared = 500, sigma = .005, amplitude = 5):
#         super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude)
        
class Squiggly_Line(Graphite_LiC12):
    example_file_path = r'C:\Users\Elizabeth Allan-Cole\Documents\GitHub\XRD-Fitting\Peak\Graphite_LiC12\squiggly_Line\fit_3.csv'
    
    def __init__(self,name = 'Squiggly_Line', q_min = 1.75, q_max = 1.9, 
                  chi_squared = 500, sigma = [.005, .005], amplitude = [5, 5]):
        super().__init__(name, q_min, q_max, chi_squared, sigma, amplitude) 
        #print('the class is %s' % self.name)
        
    # its just background so make a line
    def make_model(self, peak_vals):
        model, pars = self.make_model_background(peak_vals)
        return (model, pars)
    
    def check_sub_class_fit(self, peak_vals, sliced_q, data):
        if len(peak_vals['prominences']) == 0 or max(peak_vals['prominences']) > 1.15:
            return False
        else:
            return True
    
    # again just background so who cares?
    def check_fit(self, best_model, sliced_q, data, peak_vals):
        
        return True
        
        
    
  
  
    
# # # driver code 
# test = Graphite_LiC12('frank', 1.75, 1.9, 500, 0.005, 5) 
# print('here', test.name)
# sub_classes = Graphite_LiC12.__subclasses__()
# #df = pd.read_csv(r"C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Class-testing\Peak\fit_66.csv")
# df = pd.read_csv(r"C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Class-testing\Peak\data_72.csv")
# df = df[(df['q'] >= test.q_min) & (df['q'] <= test.q_max)]
# df.plot(x='q', y='I')
# print('returned', test.find_best_sub_peak(np.array(df.I)))