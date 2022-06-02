import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplotcolors
import matplotlib.cm as mcm
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--variant',
                     default = 'delta',
                     type = str,
                     help = "Variant type, options are 'alpha' or 'delta'.")

parser.add_argument('--metric',
                    default = 'Mean',
                    type = str,
                    help = "Mean or SEM.")

parser.add_argument('--N',
                    default = 100,
                    type = int,
                    help = "Number of trials.")

args = parser.parse_args()

if not args.variant in ['alpha', 'delta']:

    raise ValueError("Variant type must be 'alpha' or 'delta'.")

if not args.metric in ['Mean', 'Median', 'SEM']:

    raise ValueError("Metric must be 'Mean', 'Median', or 'SEM'.")

def get_heatmap_df(vax_covs, vax_effs, args):

    r_effs = np.empty((len(vax_effs), len(vax_covs)))

    options = {'Mean':0, 'Median':1, 'SEM':2}

    for idx1, vax_eff in enumerate(vax_effs):
        for idx2, vax_cov in enumerate(vax_covs):
            data = pd.read_csv(f'..\\results\\csv-data\\results-{args.variant}-0\\sim-data-vax\\summary-stats\\qld_simple-vax_apply-tt_0020_iqf_0.00_vxstr_none_vxcov_{vax_cov:.{2}f}_vxeff_{vax_eff:.{2}f}_age-lb_0.csv')
            r_effs[idx1, idx2] = data.loc[options[args.metric], 'r_eff_30']

    df = pd.DataFrame(r_effs, index = vax_effs, columns = vax_covs)

    return df

def plot_heatmap(df, args, cmap_name = 'coolwarm'):

    avg = args.metric == 'Mean' or args.metric == 'Median'

    if avg:
        vmin = 0.5
        vmax = 1.5
    else:
        vmin = 0
        vmax = 0.1

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.set(font_scale = 1.4)
    sns.heatmap(df, annot=True, fmt = ".1f" if avg else ".2f", linewidths=.5, ax=ax, cmap=cmap_name, vmin=vmin, vmax=vmax, cbar = False)
    
    ax.set_xticklabels(df.columns, rotation = 0.0, size = 15)
    ax.xaxis.tick_top()
    ax.set_yticklabels(df.index, rotation = 0.0, size = 15)
    ax.set_ylabel('Vaccine Efficacy \n (Infection Blocking)', size = 20)
    ax.set_xlabel('Coverage of Total Population', size = 20)
    ax.xaxis.set_label_position('top')
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10

    plt.rcParams['axes.titley'] = 1.1

    ax.set_title(f'{args.variant[0].upper()}{args.variant[1:]} Variant', size =22)
    

    norm = mplotcolors.Normalize(vmin = vmin, vmax = vmax)
    sm = mcm.ScalarMappable(cmap = cmap_name, norm = norm)

    cax = fig.add_axes([0.91, 0.11, 0.04, 0.77])

    cbar = fig.colorbar(sm, cax = cax)
    cbar.set_label(label = '$r_{eff}^{30}$', size = 25)
    cbar.ax.tick_params(labelsize = 20)
    

    return fig, ax

def round_(x, n):
    out = np.round(x*10**n)/10**n
    return out

vax_covs = round_(np.linspace(0.05, 1, 20), 2)
vax_effs = round_(np.linspace(0.05, 1, 20), 2)

df = get_heatmap_df(vax_covs, vax_effs, args)
fig, ax = plot_heatmap(df, args)

if args.variant == 'delta':
    
    trans = transforms.blended_transform_factory(ax.transData, ax.transData)

    ax.add_patch(patches.Rectangle(
                                (3, 15),   # (x,y)
                                    2,          # width
                                    4,          # height
                                    fill=False,
                                    transform=trans,
                                    zorder=4,
                                    alpha=1.0,
                                    ec="black",
                                    lw=6
                                    ))

    ax.add_patch(patches.Rectangle(
            (3, 15),   # (x,y)
            2,          # width
            4,          # height
            fill=False,
            transform=trans,
            zorder=4,
            alpha=1.0,
            ec="#f46d43",
            lw=2
            ))


    ax.add_patch(patches.Rectangle(
            (11, 15),   # (x,y)
            2,          # width
            4,          # height
            fill=False,
            transform=trans,
            zorder=4,
            ec="black",
            lw=6
            ))

    ax.add_patch(patches.Rectangle(
            (11, 15),   # (x,y)
            2,          # width
            4,          # height
            fill=False,
            transform=trans,
            zorder=4,
            ec="#ffffbf",
            lw=2
            ))


    ax.add_patch(patches.Rectangle(
            (13, 15),   # (x,y)
            1,          # width
            4,          # height
            fill=False,
            transform=trans,
            zorder=4,
            ec="black",
            lw=6
            ))

    ax.add_patch(patches.Rectangle(
            (13, 15),   # (x,y)
            1,          # width
            4,          # height
            fill=False,
            transform=trans,
            zorder=4,
            ec="#91cf60",
            lw=2
            ))



plt.savefig(f'results_vax-heatmap-{args.variant}_0yo-cs20-reff30_Jan2022.png', bbox_inches = 'tight')