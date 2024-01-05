# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:19:37 2023

@author: Elizabeth Allan-Cole
"""

# from scipy.ndimage import zoom
# import numpy as np

# b = np.array([1, 2, 3])
# a = np.array([4, 5, 6, 7, 8, 9])

# zoom_rate = b.shape[0] / a.shape[0]
# print(zoom_rate)
# a = zoom(a, zoom_rate)
# print(a)




import peak_fitter_functions as pf
import pandas as pd
import os
from os import listdir, makedirs
from os.path import isfile, join
import numpy as np



# Sample info
sample_name = 'S1_LN_10psi_Ch10_0120922_map_02' #charged

# path to all the tiff files
general_input_folder = r'D:\NSLS-II Winter 2023'
input_folder = os.path.join(general_input_folder, sample_name, 'integration')

general_output_folder = r'C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Winter 2023\Processing\Initial_fit\Class-testing'
#general_output_folder = r'C:\Users\Elizabeth Allan-Cole\Desktop\XRD Data Processing\NSLS-II Summer 2023\Initial_Data'
output_folder = os.path.join(general_output_folder,  'Output',  sample_name)

# if that folder dosn't exist make it exist
if not os.path.exists(output_folder):
      os.makedirs(output_folder)

# Make a list of all files names in folder
list_of_files = [files for files in listdir(input_folder) if isfile(join(input_folder, files))]


for i in range(len(list_of_files)):
        
    i_list =  [78] #, 28, 32, 66, 68, 70, 72, 74, 75, 78, 80]
    if i in i_list:

        if 'mean_q' in list_of_files[i]:
            
            df = pf.make_dataframe(list_of_files[i], input_folder)
            
            q_min, q_max = 1.75, 1.9 
            df_cut = df[(df['q'] >= q_min) & (df['q'] <= q_max)]
            
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

def index_to_xdata(xdata, indices):
    "interpolate the values from signal.peak_widths to xdata"
    ind = np.arange(len(xdata))
    f = interp1d(ind,xdata)
    return f(indices)

x = np.array(df_cut.q)
y = np.array(df_cut.I)

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

sig_list, amp_list = make_initial_guesses(x, y)

plt.plot(x,y)
# plt.plot(x[peaks], y[peaks], "x")
# plt.hlines(width_heights_half, left_ips_half, right_ips_half, color='r')
# plt.hlines(width_heights_base, left_ips_base, right_ips_base, color='b')
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()
# peaks, properties = find_peaks(df_cut.I, prominence = (1, None), width = (0,None))
# widths, width_heights, left_ips, right_ips = peak_widths(df_cut.I, peaks)
# widths = index_to_xdata(df_cut.q, widths)
# print('new width', widths)
# df_cut.plot(x='q', y='I')
# print('prom:', properties['prominences'])
# print('width:', properties['widths'])
    