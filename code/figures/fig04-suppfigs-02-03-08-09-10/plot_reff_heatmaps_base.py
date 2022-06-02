#!/usr/bin/env python
# coding: utf-8
"""
Figure 04 (mean reff30)
Supp. Figure 02 (sem reff30)
Supp. Figure 08 (mean reff30)
Supp. Figure 09 (sem reff30)

# Uses outputs of run_qld_simple_vaccine.py directly to create heatmaps of r_eff_30, or SEM reff30
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplotcolors
import matplotlib.cm as mcm
import seaborn as sns

import argparse
import os

sep = os.sep

parser = argparse.ArgumentParser()

parser.add_argument('--variant',
                     default = 'delta',
                     type = str,
                     help = "Variant type, options are 'alpha' or 'delta'.")

parser.add_argument('--age_lb',
                    default = 16,
                    type = int,
                    help = "Lower bound of age group eligible for vaccination.")

parser.add_argument('--metric',
                    default = 'Mean',
                    type = str)

parser.add_argument('--N',
                    default = 1000,
                    type = int,
                    help = "Number of trials.")

parser.add_argument('--cmap',
                    default = 'coolwarm',
                    type = str,
                    help = "Heatmap colourmap.")

args = parser.parse_args()

if not args.variant in ['alpha', 'delta', 'omicron']:

    raise ValueError("Variant type must be 'alpha', 'delta' or 'omicron'.")

if not args.metric in ['Mean', 'Median', 'SEM', 'STD', 'Variance']:

    raise ValueError("Metric must be 'Mean', 'Median', 'SEM', 'STD', or 'Variance'.")

if not args.age_lb in [0, 12, 16]:

    raise ValueError("Age lower bound must be 0, 12 or 16.")


def get_heatmap_df(vax_covs, vax_effs, args):

    r_effs = np.empty((len(vax_effs), len(vax_covs)))

    options = {'Mean':0, 'Median':1, 'SEM':2, 'STD':2, 'Variance':2}

    for idx1, vax_eff in enumerate(vax_effs):
        for idx2, vax_cov in enumerate(vax_covs):
            data = pd.read_csv(f'..{sep}../case1a{sep}results{sep}csv-data{sep}results-{args.variant}-{args.age_lb}{sep}sim-data-vax{sep}summary-stats{sep}qld_simple-vax_apply-tt_0020_iqf_0.00_vxstr_none_vxcov_{vax_cov:.{2}f}_vxeff_{vax_eff:.{2}f}_age-lb_{args.age_lb}.csv')
            r_effs[idx1, idx2] = data.loc[options[args.metric], 'r_eff_30'] #index 0 corresponds to mean value

    df = pd.DataFrame(r_effs, index = vax_effs, columns = vax_covs)

    return df

def plot_heatmap(df, total_coverage, args):

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
        
    fig, ax = plt.subplots(figsize=(9, 9) if len(df.columns) == 10 else (10, 10))
    sns.set(font_scale = 1.7 if len(df.columns) == 10 else 1.0)
    sns.heatmap(df, annot=True, fmt = ".1f" if avg else ".2f", linewidths=.5, ax=ax, cmap=args.cmap, vmin=vmin, vmax=vmax, cbar = False)

    ax.set_xticklabels(total_coverage, rotation = 0.0, size = 20 if len(df.columns) == 10 else 14)
    ax.set_yticklabels(df.index, rotation = 0.0, size = 20)
    ax.set_ylabel('Vaccine Effectiveness \n (Protection Against Infection)', size = 22)
    ax.set_xlabel('Coverage of Total Population' if not args.age_lb == 0 else 'Vaccine Coverage', size = 22)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10

    # Existing x-axis shows coverage of total QLD population, include coverage of eligible population only
    if not args.age_lb == 0:
        xax_top = ax.twiny()
        xax_top.grid(visible = False)
        xax_top.set_xlim(ax.get_xlim())
        xax_top.set_xticks(ax.get_xticks())
        xax_top.set_xticklabels(df.columns, size = 22)
        xax_top.set_xlabel('Coverage of Eligible Population')
        xax_top.xaxis.labelpad = 15
    else:
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

    #$r_{eff}^{30}$'+f' {args.metric} 
    ax.set_title(f'{args.variant[0].upper()}{args.variant[1:]} Variant, {args.age_lb}+ Eligible', size = 25, pad = 5)
    
    norm = mplotcolors.Normalize(vmin = vmin, vmax = vmax)
    sm = mcm.ScalarMappable(cmap = args.cmap, norm = norm)

    cax = fig.add_axes([0.91, 0.11, 0.04, 0.77])

    cbar = fig.colorbar(sm, cax = cax)
    cbar.set_label(label = '$r_{eff}^{30}$' if avg else '$r_{eff}^{30}$'+f' {args.metric}', size = 35, labelpad = 0 if avg else 10)
    cbar.ax.tick_params(labelsize = 25)

    return fig, ax

def round_(x, n):
    out = np.round(x*10**n)/10**n
    return out

increment_size = 0.05 if args.age_lb == 0 else 0.1
lowest_value = 0.05 if args.age_lb == 0 else 0.1
increments = 20 if args.age_lb == 0 else 10

vax_effs = round_(np.linspace(lowest_value, 1.0, increments), 2)
vax_covs = round_(np.linspace(lowest_value, 1.0, increments), 2)

if args.age_lb == 16:
    tf = 0.8 
elif args.age_lb == 12:
    tf = 0.85 # Population 12+ comprises 85% of the total population, 16+ comprises 80%
else:
    tf = 1.0

df = get_heatmap_df(vax_covs, vax_effs, args)

if args.metric == 'SEM':
    df /= np.sqrt(args.N) ## Find SEM from STD if plotting SEM
if args.metric == 'Variance':
    df = df**2

total_coverage = df.columns*tf
total_coverage = [round(f, 2) for f in total_coverage]
fig, ax = plot_heatmap(df, total_coverage, args)

## Following was adjusted from previous versions; TODO: Adjust args to get_heatmap_df()

original_metric = args.metric

args.metric = 'SEM'
SEM_df = get_heatmap_df(vax_covs, vax_effs, args)
SEM_df /= np.sqrt(args.N)

args.metric = 'Mean'
mean_df = get_heatmap_df(vax_covs, vax_effs, args)

mean_df_plus = (mean_df + SEM_df).values
mean_df_minus = (mean_df - SEM_df).values

L_indices = []
U_indices = []
used_idxs = []

for idx, (upper_bounds, lower_bounds) in enumerate(zip(mean_df_plus, mean_df_minus)):
    
    if upper_bounds[-1] < 1 and lower_bounds[-1] < 1:

        used_idxs.append(idx)
        lower_index = np.where(lower_bounds == np.min(lower_bounds[lower_bounds > 1.0]))[0][0]
        upper_index = np.where(upper_bounds == np.max(upper_bounds[upper_bounds < 1.0]))[0][0] + 1

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

#import pdb; pdb.set_trace()

#ax.set_clip_on(False)

plt.savefig(f'results_vax-heatmap_{original_metric}_{args.variant}_{args.age_lb}yo-cs20-reff30_Apr2022.png', bbox_inches = 'tight')