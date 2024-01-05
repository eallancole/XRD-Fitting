# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 09:59:17 2023

@author: Elizabeth Allan-Cole
"""

import peak_fitter_functions as pf
import user_fit_operations as ufo
import Specific_peak_fitting_functions as lpf
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import integrate
from scipy.signal import find_peaks
from scipy.integrate import simpson
from scipy.integrate import quad
from pathlib import Path
from os import listdir, chdir
from os.path import isfile, join
import regex as re
from lmfit import Model
from lmfit.models import LinearModel, GaussianModel, VoigtModel, PseudoVoigtModel, PolynomialModel, QuadraticModel
from lmfit.model import save_modelresult, load_modelresult
import math
import time
import itertools as it
from joblib import Parallel, delayed





def make_model(q_max, q_min, model_centers, sig, amp, peak_name):
    if 'Graphite-LiC12' not in peak_name: 
        background = LinearModel(prefix=('b' + '_'))  
        pars = background.make_params()
        
        model = background
        
        # initial guesses     
        slope1 = 0 
        int1 = 50
        
        # For linear background
        pars = background.make_params()
        pars['b' + '_slope'].set(slope1)
        pars['b' + '_intercept'].set(int1)
    
    else: 
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
    
      
    for peak, center in enumerate(model_centers):
        # create prefex for each peak
        pref = 'v'+str(peak)+'_'
        peak = PseudoVoigtModel(prefix=pref)
        # peak = VoigtModel(prefix=pref)
        # set the parimiters for each peak
        pars.update(peak.make_params())
        #pars[pref+'center'].set(value=center, min=q_min, max=q_max)
        pars[pref+'center'].set(value=center, min= center - 0.025, max= center + 0.025)
        pars[pref+'sigma'].set(value=sig, max = sig * 2) #reduce this guess after scipy fixes! (1.2 maybe?)
        pars[pref+'amplitude'].set(amp, min = 0)
        #pars[pref+'gamma'].set(value=sig, vary=True, expr='', min = 0) #Use for a Voigt Model
        pars[pref+'fraction'].set(value=0.5, vary=True) #Use for a Voigt Model
        
        model = model + peak
    return (model, pars)

def get_prom_model_list(q_max, q_min, center_list, sig, amp, peak_name):
    
    model_list = []
    
    for centers in range(len(center_list)):
        model_list.append(make_model(q_max, q_min, center_list[centers], sig, amp, peak_name))
    
    return(model_list)  

def run_model(sliced_q, sliced_I, model, pars):
    
    model_result = model.fit(sliced_I, pars, x = sliced_q, nan_policy = 'omit')
    return(model_result)

def fit_data(sliced_q, sliced_I, q_max, q_min, num_of_centers, sig, amp, chisqu_fit_value, x_motor, y_motor, peak_name, plot):
    chisqr = 1000000000
    initial_fit = False
    fwhm_too_big = False
    
    while chisqr >= chisqu_fit_value:

        #if more_peaks is True and num_peaks >= max_peak_allowed:
        if initial_fit == True:
            #print('Still Tomato???: ', sig, amp, chisqu_fit_value, center_list)
            #print("TURN THE USER FIT BACK ON")
            best_model = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot)
            
            chisqr = best_model.chisqr
            print('Final chi squared: ' + str(chisqr))    
            return best_model
        
        # Get the peak center ('peak') and the prominences of the peak
        peaks, properties = find_peaks(sliced_I, prominence = (1, None), width = (0,None))
        print('prom:', properties['prominences'])
        print('width:', properties['widths'])
        prom = properties['prominences']
        
        
        #center_list should be the q value corresponding to peaks(peaks is the index for the peaks found with find_peaks using peak prominence)
        center_list = np.take(sliced_q, peaks)
        #print('Old Tomato: ', sig, amp, chisqu_fit_value, center_list)
        if peak_name == 'NMC-003':
            sig, amp, chisqu_fit_value, center_list = lpf.NMC_003_peak_conditions(prom, center_list, sig, amp, chisqu_fit_value)
        
        if peak_name == 'NMC-other': 
            sig, amp, chisqu_fit_value, center_list = lpf.NMC_other_peak_conditions(prom, center_list, sig, amp, chisqu_fit_value)
        
        #print('New Tomato: ', sig, amp, chisqu_fit_value, center_list)
        num_peaks = len(center_list)
        new_center_list = []
        
        # Creates target gueses close to the identified peaks (+/- 10% sigma away from center) 
        for center in range(num_peaks):
            new_center_list.append(ufo.make_center_list(center_list[center], sig))
        
        new_center_list = ufo.iterate_centers(new_center_list)
        
        # returns a list of tuples. first value is the model second value is the pars. This looks like this ((model, pars), (model, pars), ...)
        model_list = get_prom_model_list(q_max, q_min, new_center_list, sig, amp, peak_name)
        
        model_result_list = []
       
       # if you want you can change to -1!! might be good if the thing happens a lot
        model_result_list = Parallel(n_jobs=2)(delayed(run_model)(sliced_q, sliced_I, model[0], model[1])for model in model_list)
        
        # sort the model results for best chi squared value
        results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
        best_model = results_sorted[0]
        chisqr = best_model.chisqr
        print('number of peaks found', num_peaks)
        
        if peak_name == 'NMC-other' and chisqu_fit_value == 1:
            #Hit a 3 peak case where peaks sisn't fit well - define a new chi-squared value to solve and default to automated ufo 
            chisqu_fit_value = 2000
            best_model = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot)
            
        
        # If there are no peaks use prominemce to detemine and return a line
        if peak_name != 'Li':
            len_prominence = len(properties['prominences'])
            if len_prominence != 0:
                val_promenence = max(properties['prominences']) 
    
            if len_prominence == 0 or val_promenence < 1.15: #Previous value 1.15
                best_model = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot)
                chisqr = best_model.chisqr
                print('Final chi squared: ' + str(chisqr))
                pf.plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)

                return best_model
        
        # get centers from the best model
        # if the centers are too close together, the model likely miss fit 1 peak as 2
        # this sets the model to call the user fit if the centers are too close together 
        if peak_name == 'Graphite-LiC12':
            model_center_list = []
            comps = best_model.eval_components(x=sliced_q)
    
            for prefex in comps.keys():
                if prefex != 'b_':
                    model_center_list.append(best_model.params[str(prefex)+'center'].value)
         
            if len(model_center_list) > 1:
                center_dif = float(model_center_list[1]) - float(model_center_list[0])
                if center_dif < 0.015:
                    if min(properties['prominences']) < 3:
                        best_model = pf.reduce_centers(center_dif, model_center_list, num_peaks, q_max, q_min, sig, amp, sliced_q, sliced_I, x_motor, y_motor, peak_name, chisqu_fit_value, plot)
                        
                        #Check FWHM of peak 
                        fwhm_too_big = pf.check_FWHM(sliced_q, best_model, peak_name)
                        if fwhm_too_big == True:
                            best_model = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot)
                            
                        pf.plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                        chisqr = best_model.chisqr
                        print('Final chi squared: ' + str(chisqr))
                        
                        return best_model
        
        # There is a broad polymer peak overlapping with the LiC6 peak that we need to de-convolute
        # This will increase the centers guesses to find the LiC6 peak
        if peak_name =='LiC6':
            if chisqr >= chisqu_fit_value:
                if len(center_list) == 1:
                    print('Attempting to increase number of peaks')
                    center = float(center_list[0])
                    center_list = [center - 0.01, center + 0.01]
                    
                    new_center_list = []
                    
                    # Creates target gueses close to the identified peaks (+/- 10% sigma away from center) 
                    for center in range(len(center_list)):
                        new_center_list.append(ufo.make_center_list(center_list[center], sig))
                    
                    new_center_list = ufo.iterate_centers(new_center_list)
                    model_list = get_prom_model_list(q_max, q_min, new_center_list, sig, amp, peak_name)
                    model_result_list = Parallel(n_jobs=2)(delayed(run_model)(sliced_q, sliced_I, model[0], model[1])for model in model_list)
                    
                    results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
                    best_model = results_sorted[0]
                    chisqr = best_model.chisqr
                    pf.plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                    
                    fwhm_too_big = pf.check_FWHM(sliced_q, best_model, peak_name)
                    if fwhm_too_big == True:
                        best_model = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot) 
                        print('Final chi squared: ' + str(chisqr))
                        return best_model
                    
                    if chisqr <= chisqu_fit_value:
                        print('Final chi squared: ' + str(chisqr))
                        return best_model
                    
        # If peak name is Li we need to call Li fitting fucntions to fit the voigt NMC peak
        if peak_name =='Li':
            
            # If the NMC peak is really big than we need to handle it differently
            last_I = sliced_I[-1]
            max_I = max(sliced_I)               
            if chisqr >= chisqu_fit_value:
                if max_I == last_I: 
                    center_list = np.take(sliced_q, peaks)
                    new_center_list = np.append(center_list, last_I)
                    new_center_list = [new_center_list]
                    model_list = get_prom_model_list(q_max, q_min, new_center_list, sig, amp, peak_name)
                    model_result_list = Parallel(n_jobs=2)(delayed(run_model)(sliced_q, sliced_I, model[0], model[1])for model in model_list)
                    
                    results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
                    best_model = results_sorted[0]
                    chisqr = best_model.chisqr

                    if chisqr <= chisqu_fit_value:
                        pf.plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                        print('Final chi squared: ' + str(chisqr))
                        return best_model
        
        
        # While chi sdquared is too big, this will trigger the user fit on the next loop
        initial_fit = True
    
    
    fwhm_too_big = pf.check_FWHM(sliced_q, best_model, peak_name)
    if fwhm_too_big == True:
        best_model = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot)    
            
        chisqr = best_model.chisqr
        print('Final chi squared: ' + str(chisqr))
        pf.plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                
        return best_model
    
    pf.plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
    chisqr = best_model.chisqr
    
    #print('\n\nFinal Fit Report: \n\n', best_model.fit_report())
    print('Final chi squared: ' + str(chisqr))
    
    return best_model

def master_function(read_sample_file, num_of_centers,  data_path, q_min, q_max,  sample_name, sig, amp, chisqu_fit_value, peak_name, plot, general_input_folder):
    
    # Make a dataframe of the entire XRD pattern
    df = pf.make_dataframe(read_sample_file, data_path)
    
    #TODO Normalize data
    #df_norm = normalize_data(df)
    df_norm = df
    
    # Get xy_motor positions
    x_motor, y_motor = pf.get_xy_motor(read_sample_file, data_path, general_input_folder)
    
    # Slice the dataframe to desired q range
    sliced_q, sliced_I = pf.get_points(df_norm, q_min, q_max)

    # get the best fit for the data
    best_model = fit_data(sliced_q, sliced_I, q_max, q_min, num_of_centers, sig, amp, chisqu_fit_value, x_motor, y_motor, peak_name, plot)
    print('Final fit report: \n', best_model.fit_report())
    
    if best_model is not None:
        integral_list, fwhm_list, peak_center_list = pf.get_values(best_model, sliced_q, sliced_I)
    else:
        return sample_name, x_motor, y_motor
    
    return [sample_name, x_motor, y_motor, integral_list, fwhm_list, peak_center_list, best_model, sliced_q]

# Sample info
sample_name = 'S1_LN_10psi_Ch10_0120922_map_02' #charged

plot = True
restart_run = False

# numper of centers to try
num_of_centers = 5

x_min, x_max = 0, 250
y_min, y_max = 0, 150

#Setup dataframe 
df_integrals = pd.DataFrame(columns=['Sample', 'file_name', 'x motor', 'y motor',  'Amplitude1', 'FWHM1', 'Center1',
                                     'Amplitude2', 'FWHM2', 'Center2', 'Amplitude3', 'FWHM3', 'Center3'])
# TO DO need to add in chi squared and look at how model feeds in

# path to all the tiff files
general_input_folder = r'D:\NSLS-II Winter 2023'
#general_input_folder = r'D:\NSLS-II June 2023'
input_folder = os.path.join(general_input_folder, sample_name, 'integration')

# Graphite/LiC12 only
q_range_dict = {'Graphite-LiC12':[1.75, 1.9, 500, 0.005, 5]} #Stage 2, 3, 4


# Make a list of all files names in folder
list_of_files = [files for files in listdir(input_folder) if isfile(join(input_folder, files))]


for element in q_range_dict.keys(): # for each peak defined in q_rage_dict

    # not part of the continue load feature
    df_integrals_temp = pd.DataFrame(columns=['Sample', 'file_name', 'x motor', 'y motor',  'Amplitude1', 'FWHM1', 'Center1',
                                     'Amplitude2', 'FWHM2', 'Center2', 'Amplitude3', 'FWHM3', 'Center3', 'Model Path', 'i_value'])
    
    q_min = q_range_dict.get(element)[0]
    q_max = q_range_dict.get(element)[1]
    sig = q_range_dict.get(element)[3]
    amp =q_range_dict.get(element)[4]
    
    chisqu_fit_value = q_range_dict.get(element)[2]
    print("Finding ", element, " peaks! Hold on to your socks!")
    print("qmax is " ,q_min, ", qmin is ", q_max)

    # loop through the list of files and append df_integrals --> Troubleshoot the peak fitting, getting weird numbers! 
    for i in range(len(list_of_files)):

        i_list =  [78] #26, 28, 32, 66, 68, 70, 72, 74, 75, 78, 80]
        
        if i in i_list:

            if 'mean_q' in list_of_files[i]:
                print('i', i, '\n')
                
                x, y = pf.get_xy_motor(list_of_files[i], input_folder, general_input_folder)
                if x >= x_min and x <= x_max:
                    if y >= y_min and y <= y_max:
                
                        #Call the master function to get the integral values for the specified peak
                        # returns [sample_name, x_motor, y_motor, integral_list, fwhm_list, peak_center_list, best_model]
                
                        get_integrals = master_function(list_of_files[i], num_of_centers, input_folder, q_min, q_max, 
                                                        sample_name, sig, amp, chisqu_fit_value, element, plot, general_input_folder)
                        
                    

                      
                        
                        
                        
                        
                        
###########################################################################################
                        
                        
def make_model(q_max, q_min, model_centers, sig, amp): #peak_name
    # if 'Graphite-LiC12' not in peak_name: 
    background = LinearModel(prefix=('b' + '_'))  
    pars = background.make_params()
    
    model = background
    
    # initial guesses     
    slope1 = 0 
    int1 = 50
    
    # For linear background
    pars = background.make_params()
    pars['b' + '_slope'].set(slope1)
    pars['b' + '_intercept'].set(int1)

    # else: 
    #     background = PolynomialModel(prefix=('b' + '_'))
    #     pars = background.make_params()
        
    #     model = background
        
    #     # initial guesses     
    #     a = 1
    #     b = 1
    #     c = 1
    #     pars = background.make_params()
    #     pars['b' + '_c0'].set(a)
    #     pars['b' + '_c1'].set(b)
    #     pars['b' + '_c2'].set(b)
    
      
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