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
import xlsxwriter as xl

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
            
        elif 'Back' in general_input_folder:
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
            try: 
                if df_meta['ss_stg2_y_user_setpoint'][i] == y_motor and df_meta['sample_x_user_setpoint'][i] == x_motor:
                    current_detector_count = df_meta['pe2_stats1_total'][i]
                
            except KeyError: 
                if df_meta['sample_y_user_setpoint'][i] == y_motor and df_meta['sample_x_user_setpoint'][i] == x_motor:
                    current_detector_count = df_meta['pe2_stats1_total'][i]
        
        #df_norm = pd.DataFrame(columns = ['q','I'])
        columns = ['q','I']
        values = []
        
        for i in range(len(df)):
            #calculate a normalization factor
            scale_factor = max_detector_count/ current_detector_count        
            
            #append the q and new normalized intensity to a list 
            values.append([df['q'][i], df['I'][i] * scale_factor])
    
        #print(scale_factor)
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


def make_model(q_max, q_min, model_centers, sig_list, amp_list, peak_name, overlap):
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
        a, b, c, d = 20000, -40000, 20000, -5000
        pars = background.make_params()
        # if overlap == True:
        #     max_min_list = [40000, 0, -60000, 0, 40000, 0, -10000, 0]
        # else: 
        #max_min_list = [None, None, None, None]
            
        pars['b' + '_c0'].set(value = a) #, max = max_min_list[0], min = max_min_list[1])
        pars['b' + '_c1'].set(value = b) #, min = max_min_list[2], max = max_min_list[3])
        pars['b' + '_c2'].set(value = c) #, max = max_min_list[4], min = max_min_list[5])
        pars['b' + '_c3'].set(value = d) #, min = max_min_list[6], max = max_min_list[7])
    

    for index, center in enumerate(model_centers):
        # create prefex for each peak
        pref = 'v'+str(index)+'_'
        peak = PseudoVoigtModel(prefix=pref)  #Use for a Pseudo Voigt Model
        # peak = VoigtModel(prefix=pref) #Use for a Voigt Model
        # set the parimiters for each peak
        pars.update(peak.make_params())
        pars[pref+'center'].set(value=center, min= center - 0.01, max= center + 0.01)    
        pars[pref+'sigma'].set(value=sig_list[index], max = sig_list[index] * 1.25) #reduce this guess after scipy fixes! 
        pars[pref+'amplitude'].set(amp_list[index], min = 0, max = amp_list[index] * 2) #THIS IS APPARENTLY THE AREA
        #pars[pref+'gamma'].set(value=sig, vary=True, expr='', min = 0) #Use for a Voigt Model
        pars[pref+'fraction'].set(value=0.5, vary=True) #Use for a Pseudo Voigt Model
        
        model = model + peak
    return (model, pars)


def make_background_model(q_max, q_min, b_slope, peak_name):
    if 'Graphite-LiC12' in peak_name: 
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
    else: 
        background = LinearModel(prefix=('b' + '_'))  
        pars = background.make_params()
        
        model = background
        
        # initial guesses     
        slope1 = (float(b_slope[0]) + float(b_slope[1]))/2
        int1 = 50
        
        # For linear background
        pars = background.make_params()
        pars['b' + '_slope'].set(slope1, vary = True, min = float(b_slope[0]), max = float(b_slope[1]))
        pars['b' + '_intercept'].set(int1, vary = True)
    
    return (model, pars)

def index_to_xdata(xdata, indices):
    "interpolate the values from signal.peak_widths to xdata"
    ind = np.arange(len(xdata))
    f = interp1d(ind,xdata)
    return f(indices)


def make_initial_guesses(sliced_q, sliced_I):
    peaks, properties = find_peaks(sliced_I, prominence = (1, None), height = (1, None))
    #print('prom:', properties['prominences'])
    prom = properties['prominences']
    center_list = np.take(sliced_q, peaks)

    # Cacluate the width at half max of prominence
    widths_half, width_heights_half, left_ips_half, right_ips_half = peak_widths(sliced_I, peaks, rel_height= 0.5)
    
    widths_half = index_to_xdata(sliced_q, widths_half)
    left_ips_half = index_to_xdata(sliced_q, left_ips_half)
    right_ips_half = index_to_xdata(sliced_q, right_ips_half)
    
    
    sig_list = []
    fwhm_guess_list = []
    for i in range(len(widths_half)):
        width_guess = right_ips_half[i] - left_ips_half[i]
        sig_guess = width_guess / 2.35 #Sigma is approximately sqrt(8*ln(2)) = 2.35
        sig_list.append(sig_guess)
        fwhm_guess_list.append(width_guess)
        
    #print('Sigma guess: ', sig_list)
    
    # Cacluate the width at the base
    widths_base, width_heights_base, left_ips_base, right_ips_base = peak_widths(sliced_I, peaks, rel_height = 0.9)
    
    widths_base = index_to_xdata(sliced_q, widths_base)
    left_ips_base = index_to_xdata(sliced_q, left_ips_base)
    right_ips_base = index_to_xdata(sliced_q, right_ips_base)
    
    amp_list = []
    #amplitude guess
    for i in range(len(widths_base)):
        width_guess = right_ips_base[i] - left_ips_base[i]
        amp_guess = 0.5 * prom[i] * width_guess # consider adding a 50% increase for fudge factor on triangle area, amplitude does not quite match up
        amp_list.append(amp_guess)
        
    #print('Amplitude guess: ', amp_list)
    
    #height_list = properties['peak_heights']
    height_list = properties['peak_heights']

    return prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, height_list
    

def get_prom_model_list(q_max, q_min, peak_name, sliced_q, sliced_I, chisqu_fit_value):
    model_list = []
    
    prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, height_list = make_initial_guesses(sliced_q, sliced_I)
    overlap = False
    
    if peak_name == 'Graphite-LiC12':
    # Correct peaks peak overlap in one peak
        prom, amp_list, sig_list, overlap = lpf.fix_overlap_peaks(prom, center_list, amp_list, sig_list, height_list)
    
        if overlap == False: 
            # Peak slopes to find peaks together and only one peak was identified (no 2nd prominence found)
            center_list, amp_list, sig_list, prom, overlap = lpf.check_peak_slopes_test(left_ips_base, right_ips_base, prom, center_list, amp_list, sig_list, overlap)    
        
        if len(center_list) >= 3:
            center_list, amp_list, sig_list = lpf.Graphite_3_or_4_peaks(center_list, amp_list, sig_list)
    
    if peak_name == 'NMC-003': 
        sig_list, amp_list, chisqu_fit_value, center_list = lpf.NMC_003_peak_conditions(prom, center_list, sig_list, amp_list, chisqu_fit_value)
    
    if peak_name == 'NMC-other': 
        sig_list, amp_list, chisqu_fit_value, center_list = lpf.NMC_other_peak_conditions(prom, center_list, sig_list, amp_list, chisqu_fit_value)
    
    if peak_name == 'LiC6':
        center_list, amp_list, sig_list, chisqu_fit_value = lpf.LiC6_conditions(center_list, amp_list, sig_list, chisqu_fit_value)
    # if peak_name == 'Li':
    #     for i in range(len(center_list)):
    #         model_list.append(lpf.make_Li_model(q_max, q_min, center_list[i], sig_list[i], amp_list[i], peak_name))
    #         #TODO Mkae sure to input lists for sig and amp and that these guesses are refined for Li!!! 
    #     return (model_list)
    
    # make a list of models for each center combination option
    model, pars = make_model(q_max, q_min, center_list, sig_list, amp_list, peak_name, overlap)
    model_list = (model, pars)
    
    return model_list, prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, overlap, chisqu_fit_value

def background_parameters(peak_name, q_max, q_min): 
        if peak_name == 'Graphite-LiC12':
            b_slope = '-150, 0'
        elif peak_name == 'LiC6':
            b_slope = '-450, -20'
        elif peak_name == 'Li':
            b_slope = '0, 150'
        elif peak_name == 'NMC-003':
            b_slope = '0, 650'
        elif peak_name == 'NMC-other':
            b_slope = '0, 250'
        else: 
            print('------------IF YOU GET THIS MESSAGE, MAKE SURE THE BACKGROUND FUNCTION DID NOT BLOW UP -> BOOM ------------')
            b_slope = '-inf, inf'    
    
        b_slope = b_slope.split(',')
        background_model, pars = make_background_model(q_max, q_min, b_slope, peak_name)
        return (background_model, pars)

def run_model(sliced_q, sliced_I, model, pars):
    model_result = model.fit(sliced_I, pars, x = sliced_q, nan_policy = 'omit')
    return(model_result)


def check_FWHM(sliced_q, best_model, peak_name, overlap):
    model_fwhm_list = []
    comps = best_model.eval_components(x=sliced_q)
    
    if overlap == False:
        for prefex in comps.keys():
            if prefex != 'b_':
                model_fwhm_list.append(best_model.params[str(prefex)+'fwhm'].value)
        
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
        
    
def fit_data(sliced_q, sliced_I, q_max, q_min, num_of_centers, sig, amp, chisqu_fit_value, Li_q_max, Li_q_min, x_motor, y_motor, peak_name, plot, run_mode):
    chisqr = 1000000000
    initial_fit = False
    fwhm_too_big = False
    good_fit = True
    
    while chisqr >= chisqu_fit_value:

        #if more_peaks is True and num_peaks >= max_peak_allowed:
        if initial_fit == True:
            #print('Still Tomato???: ', sig, amp, chisqu_fit_value, center_list)
            #print("TURN THE USER FIT BACK ON")
            best_model, good_fit = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot, good_fit, run_mode)
            
            chisqr = best_model.chisqr
            # print('Final chi squared: ' + str(chisqr) + '\n\n')    
            return best_model, good_fit
        
        model_list, prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, overlap, chisqu_fit_value = get_prom_model_list(q_max, q_min, peak_name, sliced_q, sliced_I, chisqu_fit_value )
        
        # if peak_name == 'Graphite-LiC12':
        #     #sig_list, amp_list, chisqu_fit_value, center_list = lpf.Graphite_LiC12_peak_conditions(prom, center_list, sig_list, amp_list, chisqu_fit_value, fwhm_guess_list)
        #     model_list, prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, overlap, chisqu_fit_value = get_prom_model_list(q_max, q_min, peak_name, sliced_q, sliced_I, chisqu_fit_value)
            
        # if peak_name == 'NMC-003':
        #     #sig_list, amp_list, chisqu_fit_value, center_list = lpf.NMC_003_peak_conditions(prom, center_list, sig_list, amp_list, chisqu_fit_value)
        #     model_list, prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, overlap, chisqu_fit_value = get_prom_model_list(q_max, q_min, peak_name, sliced_q, sliced_I, chisqu_fit_value)
        
        # if peak_name == 'NMC-other': 
        #     sig_list, amp_list, chisqu_fit_value, center_list = lpf.NMC_other_peak_conditions(prom, center_list, sig_list, amp_list, chisqu_fit_value)
        #     model_list, prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, overlap = get_prom_model_list(q_max, q_min, peak_name, sliced_q, sliced_I, chisqu_fit_value)

        
        print('Centers: ', center_list)
        print('Prom: ', prom)
        print('Sig: ', sig_list)
        print('Amp: ', amp_list)
        num_peaks = len(center_list)
        #new_center_list = []    

        best_model = run_model(sliced_q, sliced_I, model_list[0], model_list[1])
   
        if peak_name == 'Graphite-LiC12':
            chisqu_fit_value = lpf.Graphite_LiC12_reset_chisqr(best_model, sliced_q, chisqu_fit_value, prom)
            best_model = lpf.clean_up_overlap(best_model, sliced_q, sliced_I, q_min, q_max, model_list, overlap, peak_name)
        
        chisqr = best_model.chisqr
        print('number of peaks found', num_peaks)
        
        # if peak_name == 'NMC-other' and chisqu_fit_value == 1:
        #     #Hit a 3 peak case where peak isn't fit well - define a new chi-squared value to solve and default to automated ufo 
        #     "Wierd 3 peak case - Figure out fitting paramters for this case!!!"
        #     chisqu_fit_value = 2000
        #     best_model, good_fit = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot, good_fit, run_mode)
            
        
        # If there are no peaks use prominemce to detemine and return a line
        if peak_name != 'Li':
            len_prominence = len(prom)
            if len_prominence != 0:
                val_promenence = max(prom) 
            
            if len_prominence == 0 or val_promenence < 2 or (max(fwhm_guess_list) > 0.02 and val_promenence <  3): #Previous value PROM: 1.15, 1.3
                print('Only background, no peaks found!')
                model_list= background_parameters(peak_name, q_max, q_min)
                best_model = run_model(sliced_q, sliced_I, model_list[0], model_list[1])
                chisqr = best_model.chisqr
                plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                return best_model, good_fit

        # There is a broad polymer peak overlapping with the LiC6 peak that we need to de-convolute
        # This will increase the centers guesses to find the LiC6 peak
        if peak_name =='LiC6':
            if chisqr >= chisqu_fit_value:
                if len(center_list) == 1:
                    model_list, prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, overlap, chisqu_fit_value = get_prom_model_list(q_max, q_min, peak_name, sliced_q, sliced_I, chisqu_fit_value )
                    best_model = run_model(sliced_q, sliced_I, model_list[0], model_list[1])
                    chisqr = best_model.chisqr
                    
        #             print('Attempting to increase number of peaks')
        #             center = float(center_list[0])
        #             center_list = [center - 0.01, center + 0.01]
                    
        #             new_center_list = []
                    
        #             # Creates target gueses close to the identified peaks (+/- 10% sigma away from center) 
        #             for center in range(len(center_list)):
        #                 new_center_list.append(ufo.make_center_list(center_list[center], sig))
                    
        #             #new_center_list = ufo.iterate_centers(new_center_list)
        #             model_list, prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, overlap, chisqu_fit_value = get_prom_model_list(q_max, q_min, peak_name, sliced_q, sliced_I, chisqu_fit_value)
        #             model_result_list = Parallel(n_jobs=2)(delayed(run_model)(sliced_q, sliced_I, model[0], model[1])for model in model_list)
                    
        #             results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
        #             best_model = results_sorted[0]
        #             chisqr = best_model.chisqr
        #             plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                    
        #             fwhm_too_big = check_FWHM(sliced_q, best_model, peak_name, overlap)
        #             if fwhm_too_big == True:
        #                 best_model, good_fit = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot, good_fit, run_mode) 
        #                 # print('Final chi squared: ' + str(chisqr) + '\n\n')
        #                 return best_model, good_fit
                    
        #             if chisqr <= chisqu_fit_value:
        #                 # print('Final chi squared: ' + str(chisqr) + '\n\n')
        #                 return best_model, good_fit
                    
        # If peak name is Li we need to call Li fitting fucntions to fit the voigt NMC peak
        if peak_name =='Li':
            
            # If the NMC peak is really big than we need to handle it differently
            last_I = sliced_I[-1]
            max_I = max(sliced_I)               
            if chisqr >= chisqu_fit_value:
                if max_I == last_I: 
                    #center_list = np.take(sliced_q, peaks)
                    new_center_list = np.append(center_list, last_I)
                    new_center_list = [new_center_list]
                    model_list, prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, overlap, chisqu_fit_value = get_prom_model_list(q_max, q_min, new_center_list, sig, amp, peak_name, sliced_q, sliced_I, chisqu_fit_value)
                    model_result_list = Parallel(n_jobs=2)(delayed(run_model)(sliced_q, sliced_I, model[0], model[1])for model in model_list)
                    
                    results_sorted = sorted(model_result_list, key=lambda model: model.chisqr)
                    best_model = results_sorted[0]
                    chisqr = best_model.chisqr

                    if chisqr <= chisqu_fit_value:
                        plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                        # print('Final chi squared: ' + str(chisqr))
                        return best_model, good_fit
        
        
        # While chi sdquared is too big, this will trigger the user fit on the next loop
        initial_fit = True
    
    # Resolve unconstrained solutions with too high FWHM
    fwhm_too_big = check_FWHM(sliced_q, best_model, peak_name, overlap)
    if fwhm_too_big == True:
        best_model, good_fit = ufo.user_model(best_model, sliced_q, sliced_I, sig, amp, q_max, q_min, chisqu_fit_value, x_motor, y_motor, peak_name, plot, good_fit, run_mode)    
            
        chisqr = best_model.chisqr
        # print('Final chi squared: ' + str(chisqr) + '\n\n')
        plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
                
        return best_model, good_fit
    
    plot_peaks(best_model, sliced_q, sliced_I, x_motor, y_motor, peak_name, plot)
    chisqr = best_model.chisqr
    
    #print('\n\nFinal Fit Report: \n\n', best_model.fit_report())
    # print('Final chi squared: ' + str(chisqr) + '\n\n')
    # print('Chi squared aim: ', chisqu_fit_value)
    
    return best_model, good_fit


def get_values(best_model, sliced_q, sliced_I):
         
    # a list of tuples with 4 values. the peak data, fwhm, and center and amplitude.
    # Looks like ((peak_data, fwhm, center, amplitude), (peak_data, fwhm, center, amplitude), ........)
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


def master_function(read_sample_file, num_of_centers,  data_path, q_min, q_max,  sample_name, sig, amp, chisqu_fit_value, peak_name, Li_q_max, Li_q_min, plot, general_input_folder, run_mode):
    
    # Make a dataframe of the entire XRD pattern
    df = make_dataframe(read_sample_file, data_path)
    
    # Get xy_motor positions
    x_motor, y_motor = get_xy_motor(read_sample_file, data_path, general_input_folder)
    
    #TODO Normalize data for all cases
    df_norm = normalize_data(df, data_path, x_motor, y_motor)
    #df_norm = df
    
    # Slice the dataframe to desired q range
    sliced_q, sliced_I = get_points(df_norm, q_min, q_max)

    # get the best fit for the data
    best_model, good_fit = fit_data(sliced_q, sliced_I, q_max, q_min, num_of_centers, sig, amp, chisqu_fit_value, Li_q_max, Li_q_min, x_motor, y_motor, peak_name, plot, run_mode)
    print('Final fit report: \n', best_model.fit_report(), '\n\n')
    print('Final chi squared: ' + str(best_model.chisqr))# + '\n\n')
    print('Good Fit: ', good_fit, '\n\n')
    if best_model is not None:
        integral_list, fwhm_list, peak_center_list = get_values(best_model, sliced_q, sliced_I)
    else:
        return sample_name, x_motor, y_motor
    
    return [sample_name, x_motor, y_motor, integral_list, fwhm_list, peak_center_list, best_model, sliced_q, sliced_I, good_fit]


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
    
    # now save the more etaled graph
    
    comps = get_integrals[6].eval_components(x=get_integrals[7])
    
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    
    ax.scatter(get_integrals[7],get_integrals[8], label='Data', color='black')  
    ax.plot(get_integrals[7],best_model.best_fit, label='Model', color='gold')
    for prefix in comps.keys():
        ax.plot(get_integrals[7], comps[prefix], '--', label=str(prefix))

    ax.set_title(str(element) + ' : (' + str(get_integrals[1]) + ',' + str(get_integrals[2]) + ')') 
    ax.set_xlabel('q [1/A]')
    ax.set_ylabel('I [au.]')
    ax.legend()
    plt.savefig(fig_path + '_detailed')
    # keeps the plot from being displayed
    plt.close(fig)
        
    return savePath

    

def write_data_with_graph(output_file, df_integrals_temp):
    
    workbook = xl.Workbook(output_file, {"nan_inf_to_errors": True})
    worksheet = workbook.add_worksheet()
    worksheet.set_default_row(240)
    worksheet.set_column(0, 0, 50)
    worksheet.set_row(0, 20)
    picture_file_paths = list(df_integrals_temp.Plot)
    
    #add the column names
    worksheet.write_row("A1", df_integrals_temp.columns)
    #[worksheet.insert_image("A"+str(row + 2), filepath, {"x_scale": 0.5, "y_scale": 0.5}) for row, filepath in enumerate(picture_file_paths)]
    for row, filepath in enumerate(picture_file_paths):
        worksheet.insert_image("A"+str(row + 2), to_raw(filepath), {"x_scale": 0.5, "y_scale": 0.5})
    
    #[worksheet.write_row("A"+str(row + 2), df_integrals_temp.loc[row, :].values.tolist()) for row in range(df_integrals_temp.shape[0])]
    for row in range(df_integrals_temp.shape[0]):
        cell = "A"+str(row + 2)
        data = df_integrals_temp.loc[row, :].values.tolist()
        worksheet.write_row(cell, data)
    workbook.close()

def to_raw(string):
    return fr"{string}"
