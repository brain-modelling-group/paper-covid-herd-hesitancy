import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""
This script plots 
    + Figs 04 & 05
    + Supp. Figs S1 S2 S3 & S4

Paula Sanz-Leon, August 2021, QIMR
"""
sns.set_theme(style="ticks", font_scale=1.2)




filenames = ['results_vax-16078594_simple-vaccine-apply-testing-tracing_nruns-0100-b117-16yo_facetgrid_cum-deaths-reff-30.csv',
             'results_vax-16015911_simple-vaccine-apply-testing-tracing_nruns-0100-b16172-16yo_facetgrid_cum-deaths-reff-30.csv',
             'results_vax-16078596_simple-vaccine-apply-testing-tracing_nruns-0100-b117-12yo_facetgrid_cum-deaths-reff-30.csv',
             'results_vax-16116702_simple-vaccine-apply-testing-tracing_nruns-0100-b16172-12yo_facetgrid_cum-deaths-reff-30.csv',
             'results_vax-16116730_simple-vaccine-apply-testing-tracing_nruns-0100-b117-00yo_facetgrid_cum-deaths-reff-30.csv',
             'results_vax-16116703_simple-vaccine-apply-testing-tracing_nruns-0100-b16172-00yo_facetgrid_cum-deaths-reff-30.csv']


for this_filename in filenames:
    df = pd.read_csv(this_filename)

    # Initialize a grid of plots with an Axes for cluster
    grid = sns.FacetGrid(df, row="vp", col="ve", palette="coolwarm_r", height=1.2, aspect=2, margin_titles=True)

    # Draw a horizontal line 
    grid.map(plt.axhline, y=1, ls=":", c="0.5")

    # Draw a line plot to show the trajectory of each random walk
    grid.map(plt.plot, "cluster_size", "r_eff_30", marker="o")
    grid.map(plt.plot, "cluster_size", "reff", marker="o", markerfacecolor="red")

    # Adjust the tick positions and labels
    grid.set(xticks=np.arange(1, 20, 2), xlim=(0.5, 20.5), ylabel="$r_{eff}$ \n day 30", ylim=(0, 3.0), xlabel="cluster size")

    grid.set(yticks=[0, 1, 2])
    grid.set(xticks=[1, 3, 5, 7, 10, 15, 20])

    #grid.set(xlabel=[])
    # Adjust the arrangement of the plots
    grid.figure.subplots_adjust(wspace=.08, hspace=.08)
    grid.set_axis_labels(clear_inner=True)
    #grid.fig.tight_layout(w_pad=1)

    f = plt.gcf()
    plt.show()