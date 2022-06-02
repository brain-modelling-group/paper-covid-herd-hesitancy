# plot heatmaps with proportion of r_eff_30 values < 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplotcolors
import matplotlib.cm as mcm
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--variant', default = 'delta', type = str)
parser.add_argument('--age_lb', default = 16, type = int)

args = parser.parse_args()

def get_reff_heatmap_df(vax_covs, vax_effs, metric, args):

    r_effs = np.empty((len(vax_effs), len(vax_covs)))

    options = {'Mean':0, 'Median':1, 'SEM':2, 'STD':2}

    for idx1, vax_eff in enumerate(vax_effs):
        for idx2, vax_cov in enumerate(vax_covs):
            data = pd.read_csv(f'..\\results\\csv-data\\results-{args.variant}-{args.age_lb}\\sim-data-vax\\summary-stats\\qld_simple-vax_apply-tt_0020_iqf_0.00_vxstr_none_vxcov_{vax_cov:.{2}f}_vxeff_{vax_eff:.{2}f}_age-lb_{args.age_lb}.csv')
            r_effs[idx1, idx2] = data.loc[options[metric], 'r_eff_30'] #index 0 corresponds to mean value

    df = pd.DataFrame(r_effs, index = vax_effs, columns = vax_covs)

    return df

def get_proportions_heatmap_df(vax_covs, vax_effs, args):

    props = np.empty((len(vax_effs), len(vax_covs)))

    for idx1, vax_eff in enumerate(vax_effs):
        for idx2, vax_cov in enumerate(vax_covs):
            data = pd.read_csv(f'..\\results\\csv-data\\results-{args.variant}-{args.age_lb}\\sim-data-vax\\r_effs\\qld_simple-vax_apply-tt_0020_iqf_0.00_vxstr_none_vxcov_{vax_cov:.{2}f}_vxeff_{vax_eff:.{2}f}_age-lb_{args.age_lb}_r_effTS.csv')
            data.drop(columns = 'Unnamed: 0', inplace = True)
            props[idx1, idx2] = np.sum(data.iloc[29].values < 1)/len(data.iloc[29].values)

    df = pd.DataFrame(props, index = vax_effs, columns = vax_covs)

    return df

def plot_heatmap(df, total_coverage, args, cmap_name = 'Purples'):

    vmin = 0.0
    vmax = 1.0

    fig, ax = plt.subplots(figsize=(9, 9) if len(df.columns) == 10 else (10, 10))
    sns.set(font_scale = 1.7 if len(df.columns) == 10 else 1.0)
    sns.heatmap(df, annot=True, fmt = ".2f", linewidths=.5, ax=ax, cmap=cmap_name, vmin=vmin, vmax=vmax, cbar = False)
    
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


df = get_proportions_heatmap_df(vax_covs, vax_effs, args)
total_coverage = df.columns*tf
total_coverage = [round(f, 2) for f in total_coverage]
fig, ax = plot_heatmap(df, total_coverage, args)

SEM_df = get_reff_heatmap_df(vax_covs, vax_effs, 'SEM', args)
SEM_df /= np.sqrt(1000)

mean_df = get_reff_heatmap_df(vax_covs, vax_effs, 'Mean', args)

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
                ax.plot(point[0], point[1], color = 'k', lw = 3)[0]

        L_indices.append(lower_index)
        U_indices.append(upper_index)

ax.plot([L_indices[0], U_indices[0]], [used_idxs[0], used_idxs[0]], color = 'k', lw = 3)
bottom_line = ax.plot([L_indices[-1], U_indices[-1]], [used_idxs[-1]+1, used_idxs[-1]+1], color = 'k', lw = 3)[0] #clip_on = False, markevery=1)
bottom_line.set_clip_on(False)

plt.savefig(f'results_vax-heatmap-{args.variant}_{args.age_lb}yo-cs20-reff30_proportion-under-1.png', bbox_inches = 'tight')
