import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("paper", font_scale=1.5)

f, ax = plt.subplots(figsize=(6,6))

def plot_fun(df):
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


    # produce a legend with a cross section of sizes from the scatter
    #handles, labels = s1.legend_elements(prop="sizes", alpha=1)
    #legend2 = ax.legend(handles, labels, mode="expand", title="Sizes")


#df = pd.read_csv('../results/csv-data/results_vax-15915586_simple-vaccine-apply-testing-tracing_nruns-0100_facetgrid_cum-deaths-reff-30.csv')
#df1 = pd.read_csv('../results/csv-data/results_vax-16126820.csv')
#plot_fun(df1)
df2 = pd.read_csv('../results/csv-data/results_vax_simple-vaccine-apply-testing-tracing_nruns-0100-b16172-16yo-mean-vals.csv')
plot_fun(df2)


#import pdb; pdb.set_trace()
ax.set_ylabel('vaccine efficacy \n(transmission blocking)')
#ax.set_xlabel('proportion vaccinated (ages 0-104)')
ax.set_xlim([0.38 , 1.01])
ax.set_ylim([0.38 , 1.01]) 
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 
ax.invert_yaxis()
ax.set_xlabel('B.1.617.2 \n vaccine coverage \n(fraction of vaccinated people aged 16 and over)')
#ax.set_xlabel('B.1.1.7\n vaccine coverage \n(fraction of vaccinated people aged 12 and over)')
#(~72% of total population)


f.tight_layout()
plt.show()
