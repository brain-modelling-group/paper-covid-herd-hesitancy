import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("paper", font_scale=1.5)


def plot_fun(df):
    f, ax = plt.subplots(figsize=(6,6))
    cs = 20
    df_sub = df[df["cluster_size"] == cs]
    df_sub = df_sub[df_sub["descriptor"] == "mean"]


    vp = "vax_coverage"
    ve = "vax_efficacy"
    what= "cum_deaths"
    df_sub = df_sub[df_sub[ve] > 0.00]
    df_sub = df_sub[df_sub[vp] > 0.00]

    df_sub_  = df_sub[(df_sub["r_eff_30"] < 1.0)]
    df_sub__ = df_sub[(df_sub["r_eff_30"] > 1.0)]

    # Background
    s1 = plt.scatter(df_sub_[vp],  df_sub_[ve], s=(np.ceil(df_sub_[what]))*15, c='blue', alpha=1.0)
    s2 = plt.scatter(df_sub__[vp], df_sub__[ve], s=(np.ceil(df_sub__[what]))*15, c='red', alpha=1.0)

    return f, ax
    # produce a legend with a cross section of sizes from the scatter
    #handles, labels = s1.legend_elements(prop="sizes", alpha=1)
    #legend2 = ax.legend(handles, labels, mode="expand", title="Sizes")



filenames = ['results_vax_simple-vaccine-apply-testing-tracing_nruns-0100-b117-16yo-mean-vals.csv',
             'results_vax_simple-vaccine-apply-testing-tracing_nruns-0100-b117-12yo-mean-vals.csv',
             'results_vax_simple-vaccine-apply-testing-tracing_nruns-0100-b16172-16yo-mean-vals.csv',
             'results_vax_simple-vaccine-apply-testing-tracing_nruns-0100-b16172-12yo-mean-vals.csv']

xlabels = ['B.1.1.7\n vaccine coverage \n(fraction of vaccinated people aged 16 and over)',
           'B.1.1.7\n vaccine coverage \n(fraction of vaccinated people aged 12 and over)',
           'B.1.617.2 \n vaccine coverage \n(fraction of vaccinated people aged 16 and over)',
           'B.1.617.2 \n vaccine coverage \n(fraction of vaccinated people aged 12 and over)']

for (this_file, this_xlabel) in zip(filenames, xlabels):
    df2 = pd.read_csv(this_file)
    f, ax = plot_fun(df2)
    ax.set_ylabel('vaccine efficacy \n(transmission blocking)')
    ax.set_xlabel('proportion vaccinated (ages 0-104)')
    ax.set_xlim([0.38 , 1.01])
    ax.set_ylim([0.38 , 1.01]) 
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.invert_yaxis()
    ax.set_xlabel(this_xlabel)

f.tight_layout()
plt.show()
