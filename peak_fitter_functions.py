# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:44:46 2023

@author: Elizabeth Allan-Cole
"""

import user_fit_operations as ufo
import Specific_peak_fitting_functions as lpf
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
#from scipy import optimize
#from scipy import integrate
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
#from pathlib import Path
#from os import listdir, chdir
#from os.path import isfile, join
import regex as re
from lmfit import Model
from lmfit.models import LinearModel, GaussianModel, PolynomialModel, LorentzianModel, VoigtModel, PseudoVoigtModel
from lmfit.model import save_modelresult, load_modelresult
#import math
#import time
#import itertools as it
from joblib import Parallel, delayed

def make_dataframe(sample_name, data_path):

    file = open(os.path.join(data_path, sample_name))
    data = pd.read_csv(file, skiprows = 1, header = None, delim_whitespace=True)
    df = pd.DataFrame(data)
    df.columns = ['q','I']
        
    return df


def get_xy_motor(sample_name, data_path, general_input_folder):

    try:
        if 'Winter' in general_input_folder:
            # Find the x_motor position in the file title using Regex
            start_x = re.search('_x_', sample_name).end()
            end_x = re.search('mm_primary', sample_name).start() 
            x_motor = sample_name[start_x:end_x].replace(',', '.')
            x_motor = float(x_motor)
    
            # Find the y_motor position in the file title using Regex
            start_y = re.search('_y_', sample_name).end()
            end_y = re.search('mm_sample_x', sample_name).start()
            y_motor = sample_name[start_y:end_y].replace(',', '.')
            y_motor = float(y_motor)
        
        else:
            # Find the x_motor position in the file title using Regex
            start_x = re.search('_x_', sample_name).end()
            end_x = re.search('mm_sample', sample_name).start() 
            x_motor = sample_name[start_x:end_x].replace(',', '.')
            x_motor = float(x_motor)
    
            # Find the y_motor position in the file title using Regex
            start_y = re.search('_y_', sample_name).end()
            end_y = re.search('mm_primary', sample_name).start()
            y_motor = sample_name[start_y:end_y].replace(',', '.')
            y_motor = float(y_motor)
    
    except AttributeError:
        print('oh shit bra, the name changed! (function could not find x and y position in file name)')
        x_motor = input('Whats the x value?')
        x_motor = float(x_motor)
        
        y_motor = input('Whats the y value?')
        y_motor = float(y_motor)
        print("Groovie.")
    
    return x_motor, y_motor


# def normalize_data(df):
    
#     #Pull the intensity of the copper (111) peaks
#     q_min = 2.95
#     q_max = 3.06
    
#     df_Cu = df[(df['q'] >= q_min) & (df['q'] <= q_max)]
#     max_intensity = df_Cu['I'].max()
    
#     #The minumum intensity value is over the full data set 
#     min_intensity = df_Cu['I'].min()
    
#     #df_norm = pd.DataFrame(columns = ['q','I'])
#     columns = ['q','I']
#     values = []
    
#     for i in range(len(df)):
#         #calculate the normalized intensity 
#         #norm_intensity = (df['I'][i])
#         #print('Data is not being normalized, turn normalization back on')
#         norm_intensity = ((df['I'][i] - min_intensity) / (max_intensity - min_intensity))*1000        
        
#         #append the q and new normalized intensity to a list 
#         values.append([df['q'][i], norm_intensity])

#     # Put the list of q and normalized intensities into a dataframe
#     df_norm = pd.DataFrame(values, columns=columns)
    
#     return df_norm


def normalize_data(df, data_path, x_motor, y_motor):
    
    try:
        #set the folder location to the meta-data 
        os.chdir(data_path)
        os.chdir('../')
        new_path = os.getcwd()
        file_list = os.listdir(os.path.join(new_path, 'scalar_data'))
        
        # Make a dataframe of the meta data
        for file in file_list:
            meta_data = open(os.path.join(new_path,'scalar_data', file))
        
        df_meta = pd.read_csv(meta_data)
        #df = pd.DataFrame(meta_data)
            
        #The minumum and maxiumum detector counts for the data set 
        min_detector_count = df_meta['pe2_stats1_total'].min()
        max_detector_count = df_meta['pe2_stats1_total'].max()
        
        # Get the detector counts for the current x, y posistion
        for i in range(len(df_meta)):
            if df_meta['ss_stg2_y_user_setpoint'][i] == y_motor and df_meta['sample_x_user_setpoint'][i] == x_motor:
                current_detector_count = df_meta['pe2_stats1_total'][i]
        
        #df_norm = pd.DataFrame(columns = ['q','I'])
        columns = ['q','I']
        values = []
        
        for i in range(len(df)):
            #calculate a normalization factor
            scale_factor = max_detector_count/ current_detector_count        
            
            #append the q and new normalized intensity to a list 
            values.append([df['q'][i], df['I'][i] * scale_factor])
    
        print(scale_factor)
        # Put the list of q and normalized intensities into a dataframe
        df_norm = pd.DataFrame(values, columns=columns)
        
    except pd.errors.EmptyDataError:
        print('Could not Normalize, scalar data likely missing from folder, returning original data!')
        df_norm = df
    
    return df_norm


def get_points(df,q_min,q_max):
    
    ''' This function creates a condensed dataframe that isolates the deired peak
    Inputs: data set in data frame (df), lower q bound for peak(q_min), upper q bound for peak(q_max)
    Outputs: shortened dataframe (df_cut)'''
    df_cut = df[(df['q'] >= q_min) & (df['q'] <= q_max)]
    sliced_q = df_cut['q'].to_numpy()
    sliced_I = df_cut['I'].to_numpy()
    return sliced_q, sliced_I


def make_model(q_max, q_min, model_centers, sig_list, amp_list, peak_name):
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
    
      
    for index, center in enumerate(model_centers):
        # create prefex for each peak
        pref = 'v'+str(index)+'_'
        peak = PseudoVoigtModel(prefix=pref)  #Use for a Pseudo Voigt Model
        # peak = VoigtModel(prefix=pref) #Use for a Voigt Model
        # set the parimiters for each peak
        pars.update(peak.make_params())
        #pars[pref+'center'].set(value=center, min=q_min, max=q_max)
        pars[pref+'center'].set(value=center, min= center - 0.025, max= center + 0.025)
        pars[pref+'sigma'].set(value=sig_list[index], max = sig_list[index] * 2) #reduce this guess after scipy fixes! (1.2 maybe?)
        pars[pref+'amplitude'].set(value = amp_list[index], min = 0)
        #pars[pref+'gamma'].set(value=sig, vary=True, expr='', min = 0) #Use for a Voigt Model
        pars[pref+'fraction'].set(value=0.5, vary=True) #Use for a Pseudo Voigt Model
        
        model = model + peak
    return (model, pars)


def index_to_xdata(xdata, indices):
    "interpolate the values from signal.peak_widths to xdata"
    ind = np.arange(len(xdata))
    f = interp1d(ind,xdata)
    return f(indices)


def make_initial_guesses(sliced_q, sliced_I):
    peaks, properties = find_peaks(sliced_I, prominence = (1, None))
    #print('prom:', properties['prominences'])
    prom = properties['prominences']
    
    # Cacluate the width at half max of prominence
    widths_half, width_heights_half, left_ips_half, right_ips_half = peak_widths(sliced_I, peaks, rel_height= 0.5)
    
    widths_half = index_to_xdata(sliced_q, widths_half)
    left_ips_half = index_to_xdata(sliced_q, left_ips_half)
    right_ips_half = index_to_xdata(sliced_q, right_ips_half)
    
    sig_list = []
    for i in range(len(widths_half)):
        width_guess = right_ips_half[i] - left_ips_half[i]
        sig_guess = width_guess / 2.35 #Sigma is approximately sqrt(8*ln(2)) = 2.35
        sig_list.append(sig_guess)
        
    print('Sigma guess: ', sig_list)
    
    
    # Cacluate the width at the base
    widths_base, width_heights_base, left_ips_base, right_ips_base = peak_widths(sliced_I, peaks, rel_height = 0.9)
    
    widths_base = index_to_xdata(sliced_q, widths_base)
    left_ips_base = index_to_xdata(sliced_q, left_ips_base)
    right_ips_base = index_to_xdata(sliced_q, right_ips_base)
    
    amp_list = []
    #amplitude guess
    for i in range(len(widths_base)):
        width_guess = right_ips_base[i] - left_ips_base[i]
        amp_guess = 0.5 * prom[i] * width_guess * 1.5 # added a 50% increase for fudge factor on triangle area, amplitude does not quite match up
        amp_list.append(amp_guess)
        
    print('Amplitude guess: ', amp_list)

    return sig_list, amp_list
    

def get_prom_model_list(q_max, q_min, center_list, sig, amp, peak_name, sliced_q, sliced_I):
    
    model_list = []
    
    sig_list, amp_list = make_initial_guesses(sliced_q, sliced_I)
    
    if peak_name == 'Li':
        for centers in range(len(center_list)):
            model_list.append(lpf.make_Li_model(q_max, q_min, center_list[centers], sig, amp))
            #TODO Mkae sure to input lists for sig and amp and that these guesses are refined for Li!!! 
        return (model_list)
    # make a list of models for each center combination option
    
    for centers in range(len(center_list)):
        model_list.append(make_model(q_max, q_min, center_list[centers], sig_list, amp_list, peak_name))
    
    return(model_list)  


def run_model(sliced_q, sliced_I, model, pars):
    model_result = model.fit(sliced_I, pars, x = sliced_q, nan_policy = 'omit')
    return(model_result)

                      
def reduce_centers(center_dif, model_center_list, max_peak_allowed, q_max, q_min, sig, amp, sliced_q, sliced_I, x_motor, y_motor, peak_name, chisqu_fit_value, plot):
    
    max_peak_allowed = max_peak_allowed - 1
    center = (float(model_center_list[1]) + float(model_center_list[0])) / 2
    center = [center]
    
    (model, pars) = make_model(q_max, q_min, center, sig, amp, peak_name)
    
    best_model = run_model(sliced_q, sliced_I, model, pars)
    plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
    
    chisqr = best_model.chisqr
    
    print('Reduced the number of peaks, new chi sqrd: ' + str(chisqr))
    
    if chisqr > 2.5 * chisqu_fit_value:
        best_model = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot)
        plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
    
    chisqr = best_model.chisqr
    
    return best_model


def check_FWHM(sliced_q, best_model, peak_name):
    model_fwhm_list = []
    comps = best_model.eval_components(x=sliced_q)
    
    for prefex in comps.keys():
        if prefex != 'b_':
            model_fwhm_list.append(best_model.params[str(prefex)+'fwhm'].value)
    
    print('fwhm list: ', model_fwhm_list)
    
    if peak_name == 'Graphite-LiC12':
        while max(model_fwhm_list) > 0.03: 
            print('FWHM too big')
            
            return True
        
    if peak_name == 'LiC6':
        #If multiple borad peaks appear in the fit, go to user fit
        max_fwhm = 0.03
        if len(list(filter(lambda x: x >= max_fwhm, model_fwhm_list))) >= 2:
            print('potato')
            return True
        
        while max(model_fwhm_list) > 0.075: 
            
            print('FWHM too big')
            return True
        
    else:
        return False
        
    
def fit_data(sliced_q, sliced_I, q_max, q_min, num_of_centers, sig, amp, chisqu_fit_value, Li_q_max, Li_q_min, x_motor, y_motor, peak_name, plot):
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
        #TODO Make sure this number for prominences actually works
        print('prom:', properties['prominences'])
        #print('width:', properties['widths'])
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
        model_list = get_prom_model_list(q_max, q_min, new_center_list, sig, amp, peak_name, sliced_q, sliced_I)
        
        model_result_list = []
       
       # if you want you can change to -1!! might be good if the thing happens a lot
        model_result_list = Parallel(n_jobs=2)(delayed(run_model)(sliced_q, sliced_I, model[0], model[1])for model in model_list)
        
        # sort the model results for best chi squared value
        results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
        best_model = results_sorted[0]
        chisqr = best_model.chisqr
        print('number of peaks found', num_peaks)
        
        if peak_name == 'NMC-other' and chisqu_fit_value == 1:
            #Hit a 3 peak case where peak isn't fit well - define a new chi-squared value to solve and default to automated ufo 
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
                plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)

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
                        best_model = reduce_centers(center_dif, model_center_list, num_peaks, q_max, q_min, sig, amp, sliced_q, sliced_I, x_motor, y_motor, peak_name, chisqu_fit_value, plot)
                        
                        #Check FWHM of peak 
                        fwhm_too_big = check_FWHM(sliced_q, best_model, peak_name)
                        if fwhm_too_big == True:
                            best_model = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot)
                            
                        plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
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
                    model_list = get_prom_model_list(q_max, q_min, new_center_list, sig, amp, peak_name, sliced_q, sliced_I)
                    model_result_list = Parallel(n_jobs=2)(delayed(run_model)(sliced_q, sliced_I, model[0], model[1])for model in model_list)
                    
                    results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
                    best_model = results_sorted[0]
                    chisqr = best_model.chisqr
                    plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                    
                    fwhm_too_big = check_FWHM(sliced_q, best_model, peak_name)
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
                    model_list = get_prom_model_list(q_max, q_min, new_center_list, sig, amp, peak_name, sliced_q, sliced_I)
                    model_result_list = Parallel(n_jobs=2)(delayed(run_model)(sliced_q, sliced_I, model[0], model[1])for model in model_list)
                    
                    results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
                    best_model = results_sorted[0]
                    chisqr = best_model.chisqr

                    if chisqr <= chisqu_fit_value:
                        plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                        print('Final chi squared: ' + str(chisqr))
                        return best_model
        
        
        # While chi sdquared is too big, this will trigger the user fit on the next loop
        initial_fit = True
    
    # Resolve unconstrained solutions with too high FWHM
    fwhm_too_big = check_FWHM(sliced_q, best_model, peak_name)
    if fwhm_too_big == True:
        best_model = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot)    
            
        chisqr = best_model.chisqr
        print('Final chi squared: ' + str(chisqr))
        plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                
        return best_model
    
    plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
    chisqr = best_model.chisqr
    
    #print('\n\nFinal Fit Report: \n\n', best_model.fit_report())
    print('Final chi squared: ' + str(chisqr))
    
    return best_model


def get_values(best_model, sliced_q, sliced_I):
         
    # a list of tuples with 4 values. the peak data, fwhm, and center.
    # Looks like ((peak_data, fwhm, center, guess), (peak_data, fwhm, center, guess), ........)
    comps_list = []

    comps = best_model.eval_components(x=sliced_q)
    
    for prefex in comps.keys():
        if prefex != 'b_':
            comps_list.append(((comps[str(prefex)]), best_model.params[str(prefex)+'fwhm'].value, best_model.params[str(prefex)+'center'].value, best_model.params[str(prefex)+'amplitude'].value))
    
    integral_list = []
    fwhm_list = []
    peak_center_list = []
    
    for vals in comps_list:
        integral_list.append(vals[3])
        fwhm_list.append(vals[1])
        peak_center_list.append(vals[2])
        
    return integral_list, fwhm_list, peak_center_list

    
def plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot):
    if plot == True: 
        comps = best_model.eval_components(x=sliced_q)
        
        fig, ax = plt.subplots(1,1, figsize=(7,7))
        
        ax.scatter(sliced_q,sliced_I, label='Data', color='black')  
        ax.plot(sliced_q,best_model.best_fit, label='Model', color='gold')
        for prefix in comps.keys():
            ax.plot(sliced_q, comps[prefix], '--', label=str(prefix))
    
        ax.set_title(str(peak_name) + ' : (' + str(x_motor) + ',' + str(y_motor) + ')') 
        ax.set_xlabel('q [1/A]')
        ax.set_ylabel('I [au.]')
        ax.legend()
        plt.pause(1)
    
        if peak_name == 'Li':
            best_model.plot()
            plt.pause(1)


def master_function(read_sample_file, num_of_centers,  data_path, q_min, q_max,  sample_name, sig, amp, chisqu_fit_value, peak_name, Li_q_max, Li_q_min, plot, general_input_folder):
    
    # Make a dataframe of the entire XRD pattern
    df = make_dataframe(read_sample_file, data_path)
    
    # Get xy_motor positions
    x_motor, y_motor = get_xy_motor(read_sample_file, data_path, general_input_folder)
    
    #TODO Normalize data
    df_norm = normalize_data(df, data_path, x_motor, y_motor)
    #df_norm = df
    
    # Slice the dataframe to desired q range
    sliced_q, sliced_I = get_points(df_norm, q_min, q_max)

    # get the best fit for the data
    best_model = fit_data(sliced_q, sliced_I, q_max, q_min, num_of_centers, sig, amp, chisqu_fit_value, Li_q_max, Li_q_min, x_motor, y_motor, peak_name, plot)
    print('Final fit report: \n', best_model.fit_report())
    
    if best_model is not None:
        integral_list, fwhm_list, peak_center_list = get_values(best_model, sliced_q, sliced_I)
    else:
        return sample_name, x_motor, y_motor
    
    return [sample_name, x_motor, y_motor, integral_list, fwhm_list, peak_center_list, best_model, sliced_q]


def save_fits(savePath_gen, get_integrals, element, list_of_files, i, sample_name):
  
    # find the cordanets of the sample and get rid of the dots file paths don't like that
    coordinates = (str(get_integrals[1]) + '_' + str(get_integrals[2])).replace('.', '-')
    # make it a file path
    savePath = os.path.join(savePath_gen, sample_name, element, coordinates)
    
    # if that foulder dosn't exist make it exist
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # name the file
    #y = str(i)
    #file_name = str(list_of_files[i])
    #file_name = file_name.replace(",", "_")
    #file_name = file_name[:len(file_name) - 5]
    file_name = sample_name
    fig_path = os.path.join(savePath, file_name)
    # save the file! that wasn't at all convaluded was it?
    get_integrals[6].plot().savefig(fig_path)
    plt.close()
    
    #save the model fit
    os.chdir(savePath)
    best_model = get_integrals[6]
    save_modelresult(best_model, sample_name)
        
    return savePath

    