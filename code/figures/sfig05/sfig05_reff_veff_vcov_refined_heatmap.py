import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

sns.set_context("paper", font_scale=2.1)


def get_subframe(df, cluster_size=20, par_to_display_name="r_eff_30", var1="vax_efficacy", var2="vax_coverage"):
    # Filter to get the right subframe then convert to 2D array layout
    df_sub = df[(df["cluster_size"] == cluster_size)] 
    # Round for visualization purposes, only column of interest
    df_sub[par_to_display_name] = df[par_to_display_name].apply(lambda x: round(x, 2))
    df_sub = df_sub[[par_to_display_name, var2, var1]].drop_duplicates(subset=[par_to_display_name, var2, var1], keep='last').reset_index()
    df_map = df_sub.pivot(values=par_to_display_name, columns=[var2], index=var1)
    return df_map



def plot_heatmaps(df_map, cmap_name='coolwarm', fig_label=''):
    #f, ax = plt.subplots(figsize=(19.5, 17.5))
    f, ax = plt.subplots(figsize=(17.5, 17.5))
    sns.heatmap(df_map, annot=True, fmt=".1f", linewidths=.5, ax=ax, cmap=cmap_name, vmin=0.5, vmax=1.5, cbar_kws={'label': '$r_{eff}^{30}$'})
    
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

    plt.yticks(rotation=0) 
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel('vaccine efficacy \n (tranmission blocking)')
    #ax.set_xlabel('B.1.1.7\n vaccine coverage \n(fraction of vaccinated people aged 0-104 years old)\n')
    ax.set_xlabel('B.1.617.2\n vaccine coverage \n(fraction of vaccinated people aged 0-104 years old)\n')
    #ax.annotate(fig_label, xy=(0.02, 0.9125), xycoords='figure fraction', fontsize=32)
    #ax.annotate('X', xy=(0.2, 0.8), xycoords='data', fontsize=32)
    f.tight_layout()
    return


df  = pd.read_csv('../results/csv-data/results_vax_simple-vaccine-apply-testing-tracing_nruns-0100-b16172-00yo-mean-vals.csv')
dsf = get_subframe(df)
plot_heatmaps(dsf)
plt.show()