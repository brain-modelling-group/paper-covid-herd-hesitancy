#!/usr/bin/env python
# coding: utf-8

"""
Supp. Figure 01
Plot r_eff and derivative of reff timeseries

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=1.4)


import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--variant', default = 'delta', type = str)
parser.add_argument('--age_lb', default = 16, type = int)
parser.add_argument('--vax_cov', default = 1.0, type = float)
parser.add_argument('--vax_eff', default = 0.9, type = float)
parser.add_argument('--derivative', action = 'store_true')



def plot_reff_ts(args, ax=None):

    all_the_path_bits = ["../results", "csv-data", f"results-{args.variant}-{args.age_lb}", "sim-data-vax", "r_effs", f"qld_simple-vax_apply-tt_0020_iqf_0.00_vxstr_none_vxcov_{args.vax_cov:.{2}f}_vxeff_{args.vax_eff:.{2}f}_age-lb_{args.age_lb}_r_effTS.csv"] 

    # So it works on Linux, Mac and Windows
    results_folder = os.path.join(*all_the_path_bits)

    df = pd.read_csv(results_folder)
    time = df['Unnamed: 0']
    data_df = df.drop(columns = 'Unnamed: 0')

    if args.derivative:
        #time = time[1:]
        data_df = data_df.diff(axis = 0)

    mean = data_df.mean(axis = 1).values
    SEM = data_df.std(axis = 1) #/np.sqrt(len(data_df.columns))
    lb = mean - SEM
    ub = mean + SEM


    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = (12, 5))
    else:
        fig = None

    ax.plot(time, mean, color = '#153f65', label = 'mean $r_{eff}$(t)' if not args.derivative else r'mean $\frac{dr_{eff}}{dt}(t)$')
    ax.plot(time, lb, color = '#70bdf2', lw = 0.2)
    ax.plot(time, ub, color = '#70bdf2', lw = 0.2)
    ax.fill_between(time, lb, ub, color="#c1e3ff", alpha = 0.4)
    ax.vlines(30, 0, 1, color = 'k', linestyle = '--', lw = 1.0, transform=ax.get_xaxis_transform(), label = r'day 30')

    # Add some shading between 0-7
    ax.axvspan(0, 7, alpha=0.1, color='black')
    ax.axvspan(7, 21, alpha=0.05, color='black')


    if args.derivative:
        ax.hlines(0, 0, 100, color = (0.65,0.65,0.65), linestyle = '-', lw = 1.0, transform=ax.get_yaxis_transform(), label=r'$\frac{dr_{eff}}{dt}=0$')
    else:
        ax.hlines(1, 0, 100, color = (0.65,0.65,0.65), linestyle = '-', lw = 1.0, transform=ax.get_yaxis_transform(), label=r'$r_{eff}=1$')


    if args.derivative:
        ax.set_xlim([1, 90])
        ax.set_ylim([-1, 1])
    else:
        ax.set_xlim([0, 90])
        ax.set_ylim([0, 7])


    ax.set_ylabel('$r_{eff}(t)$' if not args.derivative else r'$\frac{dr_{eff}}{dt}(t)$', size = 20)

    if args.vax_cov < 0.5:
       ax.legend(loc="upper right")

    if args.derivative: 
        title_bit = r'$\frac{dr_{eff}}{dt}(t)$' 
    else: 
        title_bit = '$r_{eff}$(t)'


    ax.set_title(f'\nvaccine coverage= {args.vax_cov}')

    if args.vax_cov > 0.8:
        ax.set_xlabel('Time (days)')

    
    return fig, ax
    

if __name__ == "__main__":

   args = parser.parse_args()

   fig, ax = plt.subplots(3, 2, figsize = (12, 9))

   for row, vax_cov in enumerate([0.3, 0.7, 1.0]):
       args.vax_cov = vax_cov  
       for col, derivative in enumerate([False, True]):
            args.derivative = derivative
            _, ax_plot = plot_reff_ts(args, ax[row, col])


fig.tight_layout()
plt.show()    

  
   