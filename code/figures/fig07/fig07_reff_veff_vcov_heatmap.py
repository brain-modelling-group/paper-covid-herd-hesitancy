import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("paper", font_scale=2.6)


def get_subframe(df, cluster_size=16, par_to_display_name="r_eff_30", var1="vax_efficacy", var2="vax_coverage"):
    # Filter to get the right subframe then convert to 2D array layout
    df_sub = df[(df["cluster_size"] == cluster_size)] 
    # Round for visualization purposesm only column of interest
    df_sub[par_to_display_name] = df[par_to_display_name].apply(lambda x: round(x, 2))
    df_sub = df_sub[[par_to_display_name, var2, var1]].drop_duplicates(subset=[par_to_display_name, var2, var1], keep='last').reset_index()
    df_map = df_sub.pivot(values=par_to_display_name, columns=[var2], index=var1)
    return df_map


def plot_heatmaps(df_map, cmap_name='coolwarm', fig_label=''):
    f, ax = plt.subplots(figsize=(14.5, 12.5))
    sns.heatmap(df_map, annot=True, fmt=".1f", linewidths=.5, ax=ax, cmap=cmap_name, vmin=0.5, vmax=1.5, cbar_kws={'label': '$r_{eff}^{30}$'})
    plt.yticks(rotation=0) 
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_ylabel('vaccine efficacy \n (tranmission blocking)')


    ax.annotate(fig_label, xy=(0.02, 0.9125), xycoords='figure fraction', fontsize=32)
    f.tight_layout()

    #cv.savefig(fig_name, dpi=300)
    return f



filenames = ['../results/csv-data/results_vax_simple-vaccine-apply-testing-tracing_nruns-0100-b117-12yo-mean-vals.csv',
             '../results/csv-data/results_vax_simple-vaccine-apply-testing-tracing_nruns-0100-b117-16yo-mean-vals.csv',
             '../results/csv-data/results_vax_simple-vaccine-apply-testing-tracing_nruns-0100-b16172-12yo-mean-vals.csv',
             '../results/csv-data/results_vax_simple-vaccine-apply-testing-tracing_nruns-0100-b16172-16yo-mean-vals.csv']


xlabels = ['B.1.1.7\n vaccine coverage \n(fraction of vaccinated people aged 12-104 years old)\n',
           'B.1.1.7\n vaccine coverage \n(fraction of vaccinated people aged 16-104 years old)\n',
           'B.1.617.2\n vaccine coverage \n(fraction of vaccinated people aged 12-104 years old)\n',
           'B.1.617.2\n vaccine coverage \n(fraction of vaccinated people aged 16-104 years old)\n']

squares = [[4, 5, 8, 9], [6, 7, 8, 9], [6, 7, 8, 9], [8, 9, 8, 9]]

for (this_file, this_xlabel, points) in zip(filenames, xlabels, squares):
    df  = pd.read_csv(this_file)
    dsf = get_subframe(df)
    this_fig = plot_heatmaps(dsf)
    ax = plt.gca()
    ax.set_xlabel(this_xlabel)
    this_fig.tight_layout()
    ax.plot([points[0], points[1]], [points[2], points[2]], lw=3, color="black")
    ax.plot([points[0], points[1]], [points[3], points[3]], lw=3, color="black")
    ax.plot([points[0], points[0]], [points[2], points[3]], lw=3, color="black")
    ax.plot([points[1], points[1]], [points[2], points[3]], lw=3, color="black")
    plt.show()
