import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import pdb

# Choose variant and specific or non-specific
variants = ['delta','omicron']
# agespec = 'NonSpec'
agespecs = ['','NonSpec']

# Need to convert from probability to number of people
popsize = 5.1e6 # QLD Population size

# Name of different scenarios
Sim_Names = [ "January2021","April2021","April_12_2021","LowHesitancy"]

Fig_Titles = ["16+ Eligible \n January 2021 Hesitancy",
              "16+ Eligible \n April 2021 Hesitancy",
              "12+ Eligible \n April 2021 Hesitancy",
              "12+ Eligible \n Low Hesitancy"]

#%% Boxplot code

# Setting up plot parameters
swarmsize = 4

fsize_title = 22
fsize_legend = 18
fsize_labels= 22
fsize_numbers = 22

plt.rc('font', size=fsize_numbers)          # controls default text sizes
plt.rc('axes', titlesize=fsize_labels)     # fontsize of the axes title
plt.rc('axes', labelsize=fsize_labels)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsize_legend)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fsize_legend)    # fontsize of the tick labels
plt.rc('legend', fontsize=fsize_legend)    # legend fontsize
plt.rc('figure', titlesize=fsize_title)  # fontsize of the figure title

# Colour schemes
col_infected = np.multiply([150/255,250/255,150/255],0.95)
col_critical = np.multiply([220/255,20/255,60/255],0.9)
col_dead = [0.5,0.5,0.5]
# Vaccinated vs. unvaccinated
alph1 = 0.5
alph2 = 0.9

# Violin widths
width_bars = 0.75
# Reading dictionary names
dict_labels = ["0-14","15-24","25-34","35-44","45-54","55-64","65+"]
AgeNames = ['0-14','15-24','25-34','35-44','45-54','55-64','65+']
# Setting up dictionary
R_eff_30 = {}

agespec = agespecs[0] # Only age-specific data. 

for variant in variants:
    for ii in range(np.size(Sim_Names)):    
        #Load file 
        fname = glob.glob('../models/Results/' + Sim_Names[ii] + variant + agespec + '_Dynamics_ClusterSize_20' + '*.csv' )
        N = np.size(fname) # Find number of matching files
        print(Sim_Names[ii] + variant + agespec)
        print(N)

        R_effs = np.zeros((N))
        for jk in range(N):
            Data = pd.read_csv(fname[jk])
            # r_eff 30 is day 51 (21 + 30) in the simulation
            R_effs[jk] = Data['r_eff'][51]               

        R_eff_30[Sim_Names[ii] + variant + agespec] = R_effs           
        print("Median r_eff_30 = " + str(np.median(R_effs)))
        print("Mean r_eff_30 = " + str(np.mean(R_effs))+ " +- " +  str(np.std(R_effs)))


pdb.set_trace()

# Fix different sized arrays - data error
R_eff_30 = pd.DataFrame(R_eff_30)
R_eff_30.to_csv('AgeSpecific_R_eff_30.csv')

# Making the figure
fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(11,11))
# Box plot
plot_I = sns.boxplot(data = R_eff_30 ,ax = ax[0])
fig.savefig('All_Sims_reff30.jpg',dpi=300)        

