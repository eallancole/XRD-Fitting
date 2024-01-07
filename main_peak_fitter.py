# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:43:36 2023

@author: Elizabeth Allan-Cole
"""
import peak_fitter_functions as pf
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
from lmfit.models import LinearModel, PolynomialModel, GaussianModel, VoigtModel, PseudoVoigtModel
from lmfit.model import save_modelresult, load_modelresult
import math
import time
import itertools as it
import xlsxwriter as xl



startTime = time.time()

# Sample info
#sample_name = 'S1_LN_10psi_Ch10_0120922_map_02' #discharged
sample_name = 'S1_LN_10psi_Ch10_0120922_map_01-4' #charged

plot = True
restart_run = False
run_mode = True
refit_mode = False

# numper of centers to try
num_of_centers = 5

# the range you want lmift to use for centers for a lithium peak. 
Li_q_max = 2.53
Li_q_min = 2.545

# Define x and y bounds of NMC 
#x_min, x_max = 92, 102.5
#y_min, y_max = 66, 70.5

x_min, x_max = 0, 250
y_min, y_max = 0, 150

#Setup dataframe 
df_integrals = pd.DataFrame() #columns=['Sample', 'file_name', 'x motor', 'y motor',  'Amplitude1', 'FWHM1', 'Center1', 'Amplitude2', 'FWHM2', 'Center2', 'Amplitude3', 'FWHM3', 'Center3'])

# TODO need to add in chi squared and look at how model feeds in

# path to all the tiff files
general_input_folder = r'D:\NSLS-II Winter 2023'
#general_input_folder = r'D:\NSLS-II June 2023'
input_folder = os.path.join(general_input_folder, sample_name, 'integration')

general_output_folder = r'C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Class-testing'
#general_output_folder = r'C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Summer 2023\Initial_Data'
output_folder = os.path.join(general_output_folder,  'Output',  sample_name)
plot_folder = os.path.join(general_output_folder, 'Plot Output')

# if that folder dosn't exist make it exist
if not os.path.exists(output_folder):
     os.makedirs(output_folder)

#Set isolated peak q range dict: [q_min, q_max, chi squared, sigma, amplitude]
#q_range_dict = {'Graphite-LixC6':[1.75, 1.9, 1000, 0.1, 150], 'NMC':[1.25, 1.36, 5000, 0.2, 500], 'Li': [2.49, 2.55, 10, 0.05, 1]}
#q_range_dict = {'Graphite-LiC12':[1.75, 1.9, 10, 0.1, 100],'LiC6':[1.6, 1.75, 50, 0.1, 100] ,'NMC':[1.25, 1.36, 1000, 0.1, 1000], 'Li': [2.49, 2.55, 2, 0.05, 1]}


# Graphite/LiC12 only
q_range_dict = {'Graphite-LiC12':[1.75, 1.9, 200, 0.005, 5]} #Stage 2, 3, 4

# LiC6 only
#q_range_dict = {'LiC6':[1.6, 1.75, 15, 0.1, 100]} #Stage 1

# nmc peaks only
#q_range_dict = {'NMC-003':[1.25, 1.36, 20, 0.1, 30]}
#q_range_dict = {'NMC-other':[2.55, 2.74, 5000, 0.1, 30]}

# Li peaks only
#q_range_dict = {'Li': [2.48, 2.58, 3, 0.05, 1]} #'Li-extended': [2.45, 2.58, 10, 0.05, 2]} #go to 2.605 ish 


# Normalizing the data with the Cu-111 peak
#q_range_dict = {'Cu-111':[2.95,3.06, 5000, 0.2, 500]}


# Make a list of all files names in folder
list_of_files = [files for files in listdir(input_folder) if isfile(join(input_folder, files))]


for element in q_range_dict.keys(): # for each peak defined in q_rage_dict
    
    # part of the continue load feature do not comment me out. I'll cry.
    last_i = None
    i_has_been_set = False
    
    # not part of the continue load feature
    df_integrals_temp = pd.DataFrame(columns=['Plot','Fit_quality','User_fit','i_value','Sample', 'file_name', 'x motor', 'y motor', 'Chi_Squared', 'Amplitude1', 'FWHM1', 'Center1',
                                     'Amplitude2', 'FWHM2', 'Center2', 'Amplitude3', 'FWHM3', 'Center3','Amplitude4', 'FWHM4', 'Center4', 'Fit_Model_Path'])
    
    if restart_run == True or refit_mode == True:
        # load the og file
        load_file = os.path.join(output_folder, (sample_name + '_' + element + '.xlsx'))
        df_integrals_temp = pd.read_excel(load_file)
        # save a copy of the og file
        df_integrals_temp.to_excel(load_file.replace('.xlsx', '_copy.xlsx'))
        if restart_run == True and refit_mode == False:
            # get the last file we ran and set the i value
            last_i = int(df_integrals_temp['i_value'].max()) + 1
            print('last i is ', last_i)
        if refit_mode == True:
            # make a list of bad fits
            #  df = df.drop(df[(df.score < 50) & (df.score > 20)].index)
            bad_fits = df_integrals_temp.drop(df_integrals_temp[(df_integrals_temp.Fit_quality == 'Good') & (df_integrals_temp.User_fit.isna())].index)
            bad_i_values = list(bad_fits.i_value)
            # save all the good values
            df_integrals_temp_good = df_integrals_temp.drop(df_integrals_temp[(df_integrals_temp.Fit_quality != 'Good') | (df_integrals_temp.User_fit.notna())].index)
            if restart_run == False:
                df_integrals_temp = pd.DataFrame(columns=['Plot','Fit_quality','User_fit','i_value','Sample', 'file_name', 'x motor', 'y motor', 'Chi_Squared', 'Amplitude1', 'FWHM1', 'Center1',
                                                 'Amplitude2', 'FWHM2', 'Center2', 'Amplitude3', 'FWHM3', 'Center3','Amplitude4', 'FWHM4', 'Center4', 'Fit_Model_Path'])
        if restart_run == True and refit_mode == True:
            load_file = os.path.join(output_folder, (sample_name + '_' + element + '_refit.xlsx'))
            df_integrals_temp = pd.read_excel(load_file)
            last_i = int(df_integrals_temp['i_value'].max()) + 1
            bad_i_values = list(filter(lambda i_val: i_val >= last_i, bad_i_values))
            if bad_i_values == []:
                print('NO REFITS TO BE DONE PROGRAM SHUTTING DOWN')
                raise AttributeError
            print('last i is ', last_i)
        
            
        
        
    # if resuming a run set restart run to True above.
    # if restart_run == True:
    #     load_file = os.path.join(output_folder, (sample_name + '_' + element + '.xlsx'))
    #     # MAKE SURE THAT THE YOU SAVE A BACKUP COPY OF THE FILE YOUR LOADING FROM!!!
    #     df_integrals_temp = pd.read_excel(load_file)
    #     df_integrals_temp.to_excel(load_file.replace('.xlsx', '_copy.xlsx'))
    #     last_i = int(df_integrals_temp['i_value'].max()) + 1
    #     print('last i is ', last_i)
    
    q_min = q_range_dict.get(element)[0] 
    q_max = q_range_dict.get(element)[1] 
    sig = q_range_dict.get(element)[3] 
    amp =q_range_dict.get(element)[4] 
    
    chisqu_fit_value = q_range_dict.get(element)[2]
    print("Finding ", element, " peaks! Hold on to your socks!")
    print("qmax is " ,q_min, ", qmin is ", q_max)

    n = 0

    if last_i is None:
        last_i = 0
    if refit_mode == True:
        i_list = bad_i_values
    else:
        i_list = range(last_i, len(list_of_files))
    # loop through the list of files and append df_integrals --> Troubleshoot the peak fitting, getting weird numbers! 
    for i in i_list:
                
        if 'mean_q' in list_of_files[i]:
            print('i', i, '\n')
            x, y = pf.get_xy_motor(list_of_files[i], input_folder, general_input_folder)
            if x >= x_min and x <= x_max:
                if y >= y_min and y <= y_max:
            
                    #Call the master function to get the integral values for the specified peak
                    # returns [sample_name, x_motor, y_motor, integral_list, fwhm_list, peak_center_list, best_model]
                    
                    get_integrals = pf.master_function(list_of_files[i], num_of_centers, input_folder, q_min, q_max, 
                                                    sample_name, sig, amp, chisqu_fit_value, element, Li_q_max, Li_q_min, plot, general_input_folder, run_mode)
                    
                    
                    # save the plots for the best fit if you want
                    savePath = pf.save_fits(plot_folder, get_integrals, element, list_of_files, i, sample_name)
                    
                    
                    # this just prints the number of files we've cronked through
                    print(n)
                    n += 1
                    
                    # zips the integral_list, fwhm_list, peak_center_list together to make a list of lists
                    # ie ((integral_1, fwhm_1, center_1), (integral_2, fwhm_2, center_2))
                    vals_list = list(zip(get_integrals[3], get_integrals[4], get_integrals[5]))
                    #print('peak centers: ', get_integrals[5])
                    
                    #flatten the list to just a list (integral_1, fwhm_1, center_1, integral_2, fwhm_2, center_2)
                    vals_list = [item for sublist in vals_list for item in sublist]
                    
                    if get_integrals[9] == False:
                        fit_quality = 'Bad' # for bad fit
                    else: 
                        fit_quality = 'Good' # for good fit
                        
                    plot_file = os.path.join(savePath, sample_name + '_detailed.png' )
                    plot_file = plot_file.replace("\\", "\\\\")
                    # add the sample information and position info: plot, fit_quality, -- ,i_value ,sample_name, file name, x_motor, y_motor, chi_squared
                    info_list = [plot_file, fit_quality, '', i, get_integrals[0], list_of_files[i], get_integrals[1], get_integrals[2], get_integrals[6].chisqr]
                    # add then together
                    info_list = info_list + vals_list
                    # Find the number of nan vales we add to make this list have 12 values so we can slap it in a dataframe
                    num_nans = df_integrals_temp.shape[1] - len(info_list)
                    
                    
                    # Add a bunch of nans
                    x = 0
                    while x < num_nans:
                        info_list.append(np.nan)
                        x += 1
                        
                    # find the last row in the df    
                    max_row = df_integrals_temp.shape[0]
        
                    # slap our list of values in the dataframe!
                    df_integrals_temp.loc[max_row] = info_list
                    #Add model path to dataframe to save path for fits later
                    df_integrals_temp.loc[max_row, 'Fit_Model_Path'] = savePath
                    
                    # after each fit is run save the data frame
                    # TODO find a way to move creation of file_name and output_file out of inner for loop.
                    if refit_mode == True:
                        file_name = str(get_integrals[0] + '_' + element + '_refit.xlsx')
                    else:
                        file_name = str(get_integrals[0] + '_' + element + '.xlsx')
                    output_file = os.path.join(output_folder, file_name)    
                    
                    if run_mode == True: 
                        pf.write_data_with_graph(output_file, df_integrals_temp)
                    else:
                        df_integrals_temp.to_excel(output_file, index = False)

    # if were refiting only bad fits we need to add the good fits backinto the final report
    if refit_mode == True:
        df_integrals_temp = pd.concat([df_integrals_temp_good, df_integrals_temp]).sort_values(by=['i_value']).reset_index(drop=True)
    # add data to the master data frame
    if df_integrals.empty:
        df_integrals = df_integrals_temp
    else:
        df_integrals = pd.concat([df_integrals, df_integrals_temp])

# save the master dataframe
file_name = str(get_integrals[0]) + '_all_data_test.xlsx'
output_file = os.path.join(output_folder, file_name)

pf.write_data_with_graph(output_file, df_integrals)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
