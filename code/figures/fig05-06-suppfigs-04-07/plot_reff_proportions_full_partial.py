# plot heatmaps with proportion of r_eff_30 values < 1

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

parser.add_argument('--variant', default = 'delta', type = str)
parser.add_argument('--age_lb', default = 16, type = int)

args = parser.parse_args()

def get_reff_heatmap_df(vax_fulls, vax_parts, metric, args):

    r_effs = np.empty((len(vax_fulls), len(vax_parts)))
    r_effs.fill(np.nan)

    options = {'Mean':0, 'Median':1, 'SEM':2, 'STD':2}

    for idx1, vax_full in enumerate(vax_fulls):
        for idx2, vax_part in enumerate(vax_parts):
            if vax_part >= vax_full:
                data = pd.read_csv(f'..{sep}results{sep}csv-data{sep}results-brand-vax{sep}{args.variant}-{args.age_lb}{sep}sim-data-vax{sep}summary-stats{sep}qld_pfizer-vax_0020_iqf_0.00_vxfull_{vax_full:.{2}f}_vxpart_{vax_part:.{2}f}.csv')
                r_effs[idx1, idx2] = data.loc[options[metric], 'r_eff_74'] # Simulations start 44 days before seed infection

    df = pd.DataFrame(r_effs, index = vax_fulls, columns = vax_parts)

    return df

def get_proportions_heatmap_df(vax_fulls, vax_parts, args):

    props = np.empty((len(vax_fulls), len(vax_parts)))
    props.fill(np.nan)

    for idx1, vax_full in enumerate(vax_fulls):
        for idx2, vax_part in enumerate(vax_parts):
            if vax_part >= vax_full:
                data = pd.read_csv(f'..\\results\\csv-data\\results-brand-vax\\{args.variant}-{args.age_lb}\\sim-data-vax\\r_effs\\qld_pfizer-vax_0020_iqf_0.00_vxfull_{vax_full:.{2}f}_vxpart_{vax_part:.{2}f}_r_effTS.csv')
                data.drop(columns = 'Unnamed: 0', inplace = True)
                props[idx1, idx2] = np.sum(data.iloc[73].values < 1)/len(data.iloc[73].values)

    df = pd.DataFrame(props, index = vax_fulls, columns = vax_parts)

    return df

def plot_heatmap(df, args, cmap_name = 'Purples'):

    vmin = 0.0
    vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set(font_scale = 1.0)
    sns.heatmap(df, annot=True, fmt = ".2f", linewidths=.5, ax=ax, cmap=cmap_name, vmin=vmin, vmax=vmax, cbar = False)
    
    ax.set_xticklabels(df.columns, rotation = 0.0, size = 14)
    ax.xaxis.tick_top()
    ax.set_yticklabels(df.index, rotation = 0.0, size = 15)
    ax.set_ylabel('Proportion of Eligible Population \n Receiving Two Doses', size = 25)
    ax.set_xlabel('Proportion of Eligible Population Receiving at Least One Dose', size = 25)
    ax.xaxis.set_label_position('top')
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10

    plt.rcParams['axes.titley'] = 1.1

    ax.set_title(f'{args.variant[0].upper()}{args.variant[1:]} Variant, {args.age_lb}+ Eligible', size =22)
    

    norm = mplotcolors.Normalize(vmin = vmin, vmax = vmax)
    sm = mcm.ScalarMappable(cmap = cmap_name, norm = norm)

    cax = fig.add_axes([0.91, 0.11, 0.04, 0.77])

    cbar = fig.colorbar(sm, cax = cax)
    cbar.set_label(label = 'Proportion of $r_{eff}^{30}$ Values < 1', size = 25)
    cbar.ax.tick_params(labelsize = 20)

    return fig, ax

def round_(x, n):
    out = np.round(x*10**n)/10**n
    return out

vax_fulls = round_(np.linspace(0.0, 1.0, int(1.0/0.05) + 1), 2)
vax_parts = round_(np.linspace(0.0, 1.0, int(1.0/0.05) + 1), 2)

df = get_proportions_heatmap_df(vax_fulls, vax_parts, args)
fig, ax = plot_heatmap(df, args)

SEM_df = get_reff_heatmap_df(vax_fulls, vax_parts, 'SEM', args)
SEM_df /= np.sqrt(1000)

mean_df = get_reff_heatmap_df(vax_fulls, vax_parts, 'Mean', args)

mean_df_plus = (mean_df + SEM_df).values
mean_df_minus = (mean_df - SEM_df).values

L_indices = []
U_indices = []
used_idxs = []

for idx, (upper_bounds, lower_bounds) in enumerate(zip(mean_df_plus, mean_df_minus)):
    
    if any(upper_bounds < 1.0) and any(lower_bounds > 1.0):

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
                ax.plot(point[0], point[1], color = 'k', lw = 3)[0]

        L_indices.append(lower_index)
        U_indices.append(upper_index)

ax.plot([L_indices[0], U_indices[0]], [used_idxs[0], used_idxs[0]], color = 'k', lw = 3)
bottom_line = ax.plot([L_indices[-1], U_indices[-1]], [used_idxs[-1]+1, used_idxs[-1]+1], color = 'k', lw = 3)[0] #clip_on = False, markevery=1)
bottom_line.set_clip_on(False)

plt.savefig(f'results_vax-heatmap-{args.variant}_{args.age_lb}yo-cs20-reff30_full_partial_proportion-under-1.png', bbox_inches = 'tight')
