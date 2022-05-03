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
Median = {}

for agespec in agespecs:
    for variant in variants:
        for ii in range(np.size(Sim_Names)):    
            #Load file 
            fname = glob.glob('../models/Results/' + Sim_Names[ii] + variant + agespec + '_AgeSpec_ClusterSize_20' + '*.csv' )
            N = np.size(fname) # Find number of matching files
    
            # Prepare and clear data
            Infectious_N = np.zeros((np.size(AgeNames),N))
            Critical_N =  np.zeros((np.size(AgeNames),N))
            Dead_N =  np.zeros((np.size(AgeNames),N))
            Infectious_Vacc_N = np.zeros((np.size(AgeNames),N))
            Critical_Vacc_N =  np.zeros((np.size(AgeNames),N))
            Dead_Vacc_N =  np.zeros((np.size(AgeNames),N))   
            for jk in range(N):
                Data = pd.read_csv(fname[jk])
                # Move through ages             
                # Get values - NOTE: stored as % of population
                Infectious_N[:,jk] = np.multiply(np.array(Data['Infected']),popsize)
                Critical_N[:,jk] = np.multiply(np.array(Data['Critical']),popsize)
                Dead_N[:,jk] =  np.multiply(np.array(Data['Dead']),popsize)
                Infectious_Vacc_N[:,jk] = np.multiply(np.array(Data['Infected_Vaccinated']),popsize)
                Critical_Vacc_N[:,jk] = np.multiply(np.array(Data['Critical_Vaccinated']),popsize)
                Dead_Vacc_N[:,jk] =  np.multiply(np.array(Data['Dead_Vaccinated']),popsize)
                    
                            
            # Create dictionary for violin plots               
            
            Infectious_Data = pd.DataFrame(data = Infectious_N, index = dict_labels).T
            Critical_Data = pd.DataFrame(data = Critical_N, index = dict_labels).T
            Dead_Data = pd.DataFrame(data = Dead_N, index = dict_labels).T   
            Infectious_Vacc_Data = pd.DataFrame(data = Infectious_Vacc_N, index = dict_labels).T
            Critical_Vacc_Data = pd.DataFrame(data = Critical_Vacc_N, index = dict_labels).T
            Dead_Vacc_Data = pd.DataFrame(data = Dead_Vacc_N, index = dict_labels).T            
         
            # Output Medians
        
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+ 'Infections'] = Infectious_Data.median()           
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+ 'Infections and Vaccinated'] = Infectious_Vacc_Data.median()
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+'Critical'] = Critical_Data.median()
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+ 'Critical and Vaccinated'] = Critical_Vacc_Data.median()
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+ 'Dead'] = Dead_Data.median()
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+'Dead and Vaccinated'] = Dead_Vacc_Data.median()

            # Making the figure
            fig, ax = plt.subplots(ncols = 1, nrows = 3, figsize=(11,11))
            # Box plot
            plot_I = sns.boxplot(data = Infectious_Data ,ax = ax[0], color=col_infected)        
            plot_C = sns.boxplot(data = Critical_Data ,ax = ax[1], color=col_critical)      
            plot_D =  sns.boxplot(data = Dead_Data ,ax = ax[2], color=col_dead)      

            plot_I = sns.boxplot(data = Infectious_Vacc_Data ,ax = ax[0], color=np.multiply(col_infected,alph1))        
            plot_C = sns.boxplot(data = Critical_Vacc_Data ,ax = ax[1], color=np.multiply(col_critical,alph1))      
            plot_D =  sns.boxplot(data = Dead_Vacc_Data ,ax = ax[2], color=np.multiply(col_dead,alph1)) 
            # Scatter on top
            plot_I = sns.stripplot(data = Infectious_Data ,ax = ax[0], color=col_infected,size= swarmsize)        
            plot_C = sns.stripplot(data = Critical_Data ,ax = ax[1], color=col_critical,size= swarmsize)      
            plot_D =  sns.stripplot(data = Dead_Data ,ax = ax[2], color=col_dead,size= swarmsize)      

            plot_I = sns.stripplot(data = Infectious_Vacc_Data ,ax = ax[0], color=np.multiply(col_infected,alph1),size= swarmsize)        
            plot_C = sns.stripplot(data = Critical_Vacc_Data ,ax = ax[1], color=np.multiply(col_critical,alph1),size= swarmsize)      
            plot_D =  sns.stripplot(data = Dead_Vacc_Data ,ax = ax[2], color=np.multiply(col_dead,alph1),size= swarmsize) 

            # "Artificial" plots to get colours for legend
            bwidth = 10
            ax[0].bar(-10,1, color=col_infected,label = 'Infectious')
            ax[0].bar(-10,1, color=np.multiply(col_infected,alph1),label = 'Infectious and Vaccinated')
            ax[0].legend(loc = 'upper center')
            ax[1].bar(-10,1, color=col_critical,label = 'Critical')
            ax[1].bar(-10,1, color=np.multiply(col_critical,alph1),label = 'Critical and Vaccinated')
            ax[1].legend(loc = 'upper center')
            ax[2].bar(-10,1, color=col_dead,label = 'Dead')
            ax[2].bar(-10,1, color=np.multiply(col_dead,alph1),label = 'Dead and Vaccinated')
            ax[2].legend(loc = 'upper center')


            # Make neat
            if agespec == 'NonSpec':
            
                ax[0].set_title(Fig_Titles[ii]+' (Non-Specific)')
            else:
                ax[0].set_title(Fig_Titles[ii])
            # Set axis for all
            for jj in range(3):
                ax[jj].set_xticks([])  
                ax[jj].set_xlim([-1,np.size(AgeNames)]) 
                if jj == 2:
                    ax[jj].set_xticks([x for x in range(np.size(AgeNames))])
                    ax[jj].tick_params('x',rotation = 45)    
                    ax[jj].set_xticklabels([AgeNames[x] for x in range(np.size(AgeNames))])
        
            # ax[0].set_ylim([0,Infectious_Data.max().max()*1.5]) 
            # ax[1].set_ylim([0,Critical_Data.max().max()*1.5]) 
            # ax[2].set_ylim([0,Dead_Data.max().max()*1.5]) 
            if variant=='delta':                        
                ax[0].set_ylim([0,125000]) 
                ax[1].set_ylim([0,8000]) 
                ax[2].set_ylim([0,4000]) 
            if variant=='omicron':                        
                ax[0].set_ylim([0,800000]) 
                ax[1].set_ylim([0,30000]) 
                ax[2].set_ylim([0,15000]) 

            ax[2].set_xlabel('Age-Group') 
        
            ax[0].set_ylabel('Number of People') 
            ax[1].set_ylabel('Number of People')
            ax[2].set_ylabel('Number of People')
                # ax[0,ii].legend(loc = "upper left", ncol =3 ) 
                # ax[1,ii].legend(loc = "upper left", ncol =3 ) 
                # ax[2,ii].legend(loc = "upper left", ncol =3 ) 

            fig.savefig(Sim_Names[ii] +'_'+variant+'_'+agespec+'_'+'BoxPlot_AgeSpec_Day_90.jpg',dpi=300)

Median = pd.DataFrame(Median)
Median.to_csv('AgeSpecific_Medians.csv')