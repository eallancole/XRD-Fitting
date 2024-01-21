# -*- coding: utf-8 -*-
"""
Created on Mon May 15 20:16:28 2023

@author: Elizabeth Allan-Cole
"""
import peak_fitter_functions as pf
import numpy as np
from lmfit import Model
from lmfit.models import LinearModel, GaussianModel, PolynomialModel, LorentzianModel, VoigtModel, PseudoVoigtModel

def make_Li_model(q_max, q_min, model_centers, sig, amp):
    
    # Hard set to bound over the NMC electrode
    # x_min, x_max = 92, 102.5
    # y_min, y_max = 66, 70.5
    
    # if x_motor >= x_min and x_motor <= x_max:
    #     if y_motor >= y_min and y_motor <= y_max:
            
    linear = LinearModel(prefix=('b' + '_'))  
    pars = linear.make_params()
    
    model = linear
    
    # initial guesses     
    slope1 = 0 
    int1 = 50
    
    # For linear background
    pars = linear.make_params()
    pars['b' + '_slope'].set(slope1)
    pars['b' + '_intercept'].set(int1)
    
    # For NMC peak background
    pref_v = 'v_b_'
    voigt = VoigtModel(prefix = pref_v)
    pars.update(voigt.make_params())
    pars[pref_v+'center'].set(value=1.6, min=q_max, max=q_max + 0.2)
    pars[pref_v+'sigma'].set(value=0.005, max = sig * 2)
    pars[pref_v+'amplitude'].set(value = 5, min = 0)
    pars[pref_v+'gamma'].set(value=0.005, vary=True, expr='', min = 0)
    
    model = linear + voigt
    
    for peak, center in enumerate(model_centers):
        # create prefex for each peak
        pref = 'v'+str(peak)+'_'
        #peak = GaussianModel(prefix=pref)
        peak = VoigtModel(prefix=pref)
        # set the parimiters for each peak
        pars.update(peak.make_params())
        pars[pref+'center'].set(value=center, min=2.52, max=2.55)
        pars[pref+'sigma'].set(value=0.003, max = sig * 2)
        pars[pref+'amplitude'].set(value=0.2, min = 0)
        #FOR PSEUDO VOIGT pars[pref+'alpha'].set(value=sig, vary=True, expr='', min = 0)
        pars[pref+'gamma'].set(value=sig, vary=True, expr='', min = 0)
        #pars[pref+'height'].set(value=height, vary=True, expr='', min = 0, max = height * 2)
        #pars[pref+'fwhm'].set(value=fwhm, vary=True, expr='', min = 0.0000001, max = fwhm * 1.5)
                
        model = model + peak
    
        return (model, pars)
    
#Function to make better guesses for peaks that overlap when 2 peaks close together are found
def fix_overlap_peaks(prom, center_list, amp_list, sig_list, height_list):
    
    overlap = False
    
    for i in range(len(center_list)-1):
        j = i + 1

        height_diff = abs(height_list[i] - height_list[j])
        center_diff = abs(center_list[i] - center_list[j])
        
        if height_diff < 100 and center_diff < 0.03:
            # find which peak has the smaller prominence, we need to change the guesses for this peak 
            # print('Woohoo - time to troubleshoot and make sure this actually works!!')
            if prom[i] <= prom[j]:
                peak_to_change_index = i
                guess_index_to_copy = j
            else:
                peak_to_change_index = j
                guess_index_to_copy = i
            # print('Hit the right branch, Prom original: ', prom)
            # print('i and j: ', i, ', ', j)
            
            prom = np.delete(prom, peak_to_change_index)
            amp_list.pop(peak_to_change_index)
            sig_list.pop(peak_to_change_index)
            
            prom = np.insert(prom, peak_to_change_index, prom[guess_index_to_copy - 1])
            amp_list.insert(peak_to_change_index, amp_list[guess_index_to_copy - 1])
            sig_list.insert(peak_to_change_index, sig_list[guess_index_to_copy - 1])
            
            # print('Hit the right branch, Prom is now: ', prom)
            
            overlap = True
        
    return prom, amp_list, sig_list, overlap
    

# Check if one peak is actually two peaks 
def check_peak_slopes_test(left_ips_base, right_ips_base, prom, center_list, amp_list, sig_list, overlap):

    # Check approximate peak slopes to determine if it should be two peaks together
    for i in range(len(center_list)):
        rise = prom[i]
        center = center_list[i]

        #calculate the left slope of the peak
        run_left = center - left_ips_base[i] # should be a positive run and slope
        slope_left = abs(rise / run_left) 
        
        #calculate the right slope of the peak
        run_right = center - right_ips_base[i] # should be a negative run and slope
        slope_right = abs(rise / run_right)
        
        slope_diff = abs(slope_left - slope_right)
        # print('Left slope: ', slope_left, '      Right: ', slope_right)
        
        if (slope_diff > 2000 and prom[i] > 100 and prom[i] < 400) or (slope_diff > 1500 and prom[i] < 100): 
            #print('----------------Found 2 overlapping peaks----------------')    
                
            if slope_left <= slope_right: #left slope is smaller (less steep), so 2nd peak on left side
                # Add in the center to the center list    
                new_center_guess = center - 0.015
                center_list = np.append(center_list, new_center_guess)
                

            if slope_left >= slope_right: #right slope is smaller, so 2nd peak on right side
                # Add in the center to the center list    
                new_center_guess = center + 0.015
                center_list = np.append(center_list, new_center_guess)
                
            new_sig_guess = sig_list[i] * 1.5
            new_amp_guess = amp_list[i]
            sig_list.append(new_sig_guess)
            amp_list.append(new_amp_guess)
            prom = np.append(prom, prom[i])
            overlap = True

    return center_list, amp_list, sig_list, prom, overlap        
        
def Graphite_3_or_4_peaks(center_list, amp_list, sig_list):
    # had to have found 3 or 4 peaks with prominence for this to work! 
    amp_list = [max(amp_list)] * len(center_list)
    sig_list = [max(sig_list)] * len(center_list)
    
    return center_list, amp_list, sig_list

def get_comps(best_model, sliced_q):
    # Evaluate best_model
    comps_list = []
    comps = best_model.eval_components(x=sliced_q)
    
    # a list of tuples with 3 values. the fwhm, and center and height.
    # Looks like ((fwhm, center, height), (fwhm, center, height), ........)
    for prefex in comps.keys():
        if prefex != 'b_':
            comps_list.append((best_model.params[str(prefex)+'fwhm'].value, best_model.params[str(prefex)+'center'].value, best_model.params[str(prefex)+'height'].value))

    return comps_list

def Graphite_LiC12_reset_chisqr(best_model, sliced_q, chisqu_fit_value, prom):
    # variables we need for calculation from best_model
    # ((fwhm, center, height), (fwhm, center, height), ...)
    comps_list = get_comps(best_model, sliced_q)
    
    if len(prom) != 0:
        for i, peak in enumerate(comps_list):
            if prom[i] >= 50 and prom[i] <= 200: 
                center = peak[1]
                fwhm = peak[0]
                height = peak[2]
                if center >= 1.78 and center <= 1.825: # if the peak center is in range
                    if fwhm >= 0.015 and fwhm <= 0.03: # and (130 <= height <= 200): #if the FWHM and  
                        chisqu_fit_value = 1000 
    
    #Handle large graphite peak 
    if len(prom) != 0 and max(prom) > 1000:
        chisqu_fit_value = 5000
        
    if len(prom) != 0 and max(prom) >= 400 and max(prom) <= 1000:
        chisqu_fit_value = 1000
        
    # if overlap == True :
    #     chisqu_fit_value = 1000
    
    return chisqu_fit_value
    
def clean_up_overlap(best_model, sliced_q, sliced_I, q_min, q_max, model_list, overlap, peak_name):
    comps_list = get_comps(best_model, sliced_q)
    centers = [index[1] for index in comps_list]

    if len(centers) > 1:
        j = 0
        center_diff_list = []
        for i in range(len(centers) - 1):
            j = i + 1
            center_diff = abs(centers[j] - centers[i])
            center_diff_list.append(center_diff)
        print('Solved Center Diff List: ', center_diff_list)
        if min(center_diff_list) < 0.0075:
            prom, center_list, sig_list, amp_list, left_ips_base, right_ips_base, fwhm_guess_list, height_list = pf.make_initial_guesses(sliced_q, sliced_I)
            model, pars = pf.make_model(q_max, q_min, center_list, sig_list, amp_list, peak_name, overlap)
            model_list = (model, pars)
            best_model = pf.run_model(sliced_q, sliced_I, model_list[0], model_list[1])
    
    return best_model

def LiC6_conditions(center_list, amp_list, sig_list, chisqu_fit_value):
    # TODO, contrain by center and make sure that it is actually finfing LiC6
    # Make sure it it accurately fitting if there is no LiC6
    center_list = [1.68, 1.695]
    amp_list = [1.5, 0.5]
    sig_list = [0.04, 0.003]
    chisqu_fit_value = 200

    return center_list, amp_list, sig_list, chisqu_fit_value
   
def NMC_003_peak_conditions(prom, center_list, sig_list, amp_list, chisqu_fit_value):
    if len(center_list) > 1 and min(prom) < 2:
        index = int(np.where(prom == max(prom))[0])
        center_list = np.array([(center_list[index])]) #This needs to be the center list corresponding with max prom, not the max of the center list!!!
        sig_list = np.array([(sig_list[index])])
        amp_list = np.array([(amp_list[index])])
        prom = np.array([max(prom)])
    if len(prom) > 0:
        #print('Here are the prominences and center we are solving with: ', prom, center_list)
        if max(prom) > 350:
            print('BIG PEAK!')
            chisqu_fit_value = 10000
        elif max(prom) < 10:
            print('TINY PEAK!')
            chisqu_fit_value = 10
        elif 10 <= max(prom) <= 150:
            print('A small peak was found!')
            chisqu_fit_value = 1000
        elif 150 < max(prom) <= 350:
            print('A medium peak was found!')
            chisqu_fit_value = 2000

    return sig_list, amp_list, chisqu_fit_value, center_list

def NMC_other_peak_conditions(prom, center_list, sig_list, amp_list, chisqu_fit_value):
    if len(prom) > 0:
        #print('Here are the prominences and center we are solving with: ', prom, center_list)
        if len(prom) == 1:
            if 2.675 < float(center_list[0]) < 2.70:
                #print('Aluminum [111] peak')
                chisqu_fit_value = 5000
            if max(prom) > 1400:
                chisqu_fit_value = 10000
            
        elif len(prom) > 1:
            chisqu_fit_value = 6000
            
            # if len(prom) == 3 and min(prom) < 20:
            #     chisqu_fit_value = 3500
                # sig_list, amp_list = NMC_other_increase_guess(prom, center_list, sig_list, amp_list)
            # Covers two large peaks and a slightly larger 3rd peak than above
            if len(prom) == 3: #and min(prom) > 50
                chisqu_fit_value = 20000
                sig_list, amp_list = NMC_other_increase_guess(prom, center_list, sig_list, amp_list)
            
            if len(prom) == 2 and max(center_list) < 2.7:
                chisqu_fit_value = 10000
            # Covers the 2 large peak case for discharged
            if len(prom) == 2 and min(prom) > 200:
                chisqu_fit_value = 35000
            if len(prom) == 2 and max(prom) > 2000:
                chisqu_fit_value = 50000
            if len(prom) == 2 and min(center_list) < 2.575:
                center_list = [2.56, 2.57, max(center_list)]
                sig_list = [min(sig_list), min(sig_list), max(sig_list)]
                amp_list = [min(amp_list), min(amp_list), max(amp_list)]
                chisqu_fit_value = 5000
                #print('******* Hit the weird case! ********')
            
        else: 
            print('Add next case!!!')

    return sig_list, amp_list, chisqu_fit_value, center_list

def NMC_other_increase_guess(prom, center_list, sig_list, amp_list):            
    if len(center_list) == 3 and max(center_list) < 2.7:
        amp_list = [amp_list[0], 3, amp_list[2]]
        sig_list = [sig_list[0], 0.005, sig_list[2]]
    elif len(center_list) == 3 and max(center_list) > 2.7:
        amp_list = [amp_list[0], amp_list[1], 3]
        sig_list = [sig_list[0], sig_list[1], 0.007]

    return sig_list, amp_list
