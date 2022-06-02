#!/usr/bin/env python
# coding: utf-8
"""
Figures 05 and 06
Supp. Figures 04 and 06

Uses outputs of run_qld_brand_vaccine.py directly to create heatmaps of mean r_eff_30, 
or SEM r_eff_30. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplotcolors
import matplotlib.cm as mcm
import seaborn as sns
import os

sep = os.sep

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--age_lb',
                    default = 16,
                    type = int,
                    help = "Lower bound of age group eligible for vaccination.")

parser.add_argument('--variant',
                    default = 'delta',
                    type = str,
                    help = "Variant active in simulations.")

parser.add_argument('--metric',
                    default = 'Mean',
                    type = str,
                    help = "Mean, Median, SEM or 'STD'.")

parser.add_argument('--N',
                    default = 1000,
                    type = int,
                    help = "Number of trials.")

parser.add_argument('--cmap',
                    default = 'coolwarm',
                    type = str,
                    help = "Heatmap colourmap.")

args = parser.parse_args()

if not args.variant in ['delta', 'omicron']:

    raise ValueError("Variant type must be 'delta' or 'omicron'.")

if not args.age_lb in [12, 16]:

    raise ValueError("Age lower bound must be 12 or 16.")

if not args.metric in ['Mean', 'Median', 'SEM', 'STD']:

    raise ValueError("Metric must be 'Mean', 'Median', 'SEM', or 'STD'.")

def get_heatmap_df(vax_parts, vax_fulls, args):

    r_effs = np.empty((len(vax_fulls), len(vax_parts)))
    r_effs.fill(np.nan)

    options = {'Mean':0, 'Median':1, 'SEM':2, 'STD':2}

    for idx1, vax_full in enumerate(vax_fulls):
        for idx2, vax_part in enumerate(vax_parts):
            if vax_part >= vax_full:
                data = pd.read_csv(f'..{sep}results{sep}csv-data{sep}results-brand-vax{sep}{args.variant}-{args.age_lb}{sep}sim-data-vax{sep}summary-stats{sep}qld_pfizer-vax_0020_iqf_0.00_vxfull_{vax_full:.{2}f}_vxpart_{vax_part:.{2}f}.csv')
                r_effs[idx1, idx2] = data.loc[options[args.metric], 'r_eff_74'] # Simulations start 44 days before seed infection

    df = pd.DataFrame(r_effs, index = vax_fulls, columns = vax_parts)

    return df

def plot_heatmap(df, args):

    avg = args.metric == 'Mean' or args.metric == 'Median'

    if avg:
        vmin = 0.5
        vmax = 1.5
    elif args.metric == 'SEM':
        vmin = 0
        vmax = 0.04
    elif args.metric == 'SEM':
        vmin = 0
        vmax = 0.4
    elif args.metric == 'Variance':
        vmin = 0
        vmax = 1.3

    fig, ax = plt.subplots(figsize=(9, 10))
    sns.set(font_scale = 1.05 if not args.metric == 'SEM' else 1.0)
    sns.heatmap(df, annot=True, fmt = ".1f" if avg else ".2f", linewidths=.5, ax=ax, cmap=args.cmap, vmin=vmin, vmax=vmax, cbar = False)
    
    ax.set_xticklabels(df.columns, rotation = 0.0, size = 12)
    ax.xaxis.tick_top()
    ax.set_yticklabels(df.index, rotation = 0.0, size = 12)
    ax.set_ylabel('Proportion of Eligible Population \n Receiving Two Doses', size = 16)
    ax.set_xlabel('Proportion of Eligible Population Receiving at Least One Dose', size = 16)
    ax.xaxis.set_label_position('top')
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10

    ax.set_title(f'{args.variant[0].upper()}{args.variant[1:]} Variant, {args.age_lb}+ Eligible', size = 20, pad = 10)
    
    norm = mplotcolors.Normalize(vmin = vmin, vmax = vmax)
    sm = mcm.ScalarMappable(cmap = args.cmap, norm = norm)

    cax = fig.add_axes([0.91, 0.11, 0.04, 0.77])

    cbar = fig.colorbar(sm, cax = cax)
    cbar.set_label(label = '$r_{eff}^{30}$' if avg else '$r_{eff}^{30}$'+f' {args.metric}', size = 30)
    cbar.ax.tick_params(labelsize = 22)

    return fig, ax

def round_(x, ndec, base = 1):
    out = np.round(x*10**ndec/base)*base/10**ndec
    return out

def plot_box(ax, fst_dose_pct, scd_dose_pct, color, linestyle):

    fst_dose_frac = round_(fst_dose_pct/100, 2, 5)
    scd_dose_frac = round_(scd_dose_pct/100, 2, 5)
    x1 = int(round(fst_dose_frac/0.05))
    y1 = int(round(scd_dose_frac/0.05))

    line_segs = np.array([[[x1, x1+1],[y1, y1]], [[x1+1, x1+1],[y1, y1+1]], [[x1+1, x1],[y1+1, y1+1]], [[x1, x1],[y1+1, y1]]])

    for segment in line_segs:

        ax.plot(segment[0], segment[1], color = color, lw = 3, linestyle = linestyle)

    return ax




vax_parts = round_(np.linspace(0, 1, 21), 2)
vax_fulls = round_(np.linspace(0, 1, 21), 2)

df = get_heatmap_df(vax_parts, vax_fulls, args)

if args.metric == 'SEM':
    df /= np.sqrt(args.N) ## Find SEM from STD if plotting SEM

fig, ax = plot_heatmap(df, args)

original_metric = args.metric

args.metric = 'SEM'
SEM_df = get_heatmap_df(vax_fulls, vax_parts, args)
SEM_df /= np.sqrt(args.N)

args.metric = 'Mean'
mean_df = get_heatmap_df(vax_fulls, vax_parts, args)

mean_df_plus = (mean_df + SEM_df).values
mean_df_minus = (mean_df - SEM_df).values

L_indices = []
U_indices = []
used_idxs = []

for idx, (upper_bounds, lower_bounds) in enumerate(zip(mean_df_plus, mean_df_minus)):
    
    if any(upper_bounds < 1.0) and any(lower_bounds > 1.0):

        used_idxs.append(idx)
        #if idx >= 19:
            #import ipdb; ipdb.set_trace()
        lower_index = np.where(lower_bounds == np.nanmin(lower_bounds[lower_bounds > 1.0]))[0][0]
        upper_index = np.where(upper_bounds == np.nanmax(upper_bounds[upper_bounds < 1.0]))[0][0] + 1

        # Place rectangles
        points = np.array([[[upper_index, upper_index],[idx, idx + 1]], [[lower_index, lower_index], [idx + 1, idx]]])
        for point in points:

            ax.plot(point[0], point[1], color = 'k', lw = 3)

        if L_indices:

            points = np.array([[[upper_index, U_indices[-1]],[idx, idx]], [[lower_index, L_indices[-1]],[idx, idx]]])
            for point in points:
                ax.plot(point[0], point[1], color = 'k', lw = 3)

        L_indices.append(lower_index)
        U_indices.append(upper_index)

ax.plot([L_indices[0], U_indices[0]], [used_idxs[0], used_idxs[0]], color = 'k', lw = 3)
bottom_line = ax.plot([L_indices[-1], U_indices[-1]], [used_idxs[-1]+1, used_idxs[-1]+1], color = 'k', lw = 3)[0] #clip_on = False, markevery=1)
bottom_line.set_clip_on(False)

vax_data = pd.read_csv('vax_data_full_partial.csv')
keys = vax_data.T.keys()

for date in keys[:-2]:

    fst_dose_pct = vax_data.loc[date][f'{args.age_lb}+_fst-dose']
    scd_dose_pct = vax_data.loc[date][f'{args.age_lb}+_scd-dose']
    #ax = plot_box(ax, fst_dose_pct, scd_dose_pct, '#fff7a1', 'solid')

#plt.show()
plt.savefig(f'results_vax-heatmap_full-partial_{original_metric}_{args.variant}_{args.age_lb}yo-cs20-reff30_Feb2022.png', bbox_inches = 'tight')