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

Fig_Titles = ["16+ Eligible \n High Hesitancy (20%, January 2021)",
              "16+ Eligible \n Low Hesitancy (14%, April 2021)",
              "12+ Eligible \n Low Hesitancy (14%, April 2021)",
              "12+ Eligible \n Very Low Hesitancy (8%)"]

#%% boxplot code

# Setting up plot parameters
swarmsize = 2

fsize_title = 30
fsize_legend = 30
fsize_labels= 30
fsize_numbers = 25

plt.rc('font', size=fsize_numbers)          # controls default text sizes
plt.rc('axes', titlesize=fsize_labels)     # fontsize of the axes title
plt.rc('axes', labelsize=fsize_labels)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsize_numbers)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fsize_numbers)    # fontsize of the tick labels
plt.rc('legend', fontsize=fsize_legend)    # legend fontsize
plt.rc('figure', titlesize=fsize_title)  # fontsize of the figure title

# Colour schemes
col_infected = np.multiply([150/255,250/255,150/255],0.95)
col_critical = np.multiply([220/255,20/255,60/255],0.9)
col_dead = [0.5,0.5,0.5]
# Vaccinated vs. unvaccinated
alph1 = 0.5
alph2 = 0.9
alph_swarm = 0.2

# Violin widths
width_bars = 0.75
bwidth = 0.5
# Reading dictionary names
dict_labels = ["0-14","15-24","25-34","35-44","45-54","55-64","65+"]
AgeNames = ['0-14','15-24','25-34','35-44','45-54','55-64','65+']
# Setting up dictionary
Median = {}

for agespec in agespecs:
    for variant in variants:
        for ii in range(np.size(Sim_Names)):                      
                            
            # Load dictionary for violin plots               
            Infectious_Data = pd.read_csv('../models/combined_results/' + Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_Infectious.csv',index_col=0) 
            Critical_Data= pd.read_csv('../models/combined_results/' +Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_Critical.csv',index_col=0) 
            Dead_Data = pd.read_csv('../models/combined_results/' +Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_Dead.csv',index_col=0) 
            Infectious_Vacc_Data = pd.read_csv('../models/combined_results/' +Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_InfectiousVacc.csv',index_col=0) 
            Critical_Vacc_Data = pd.read_csv('../models/combined_results/' +Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_CriticalVacc.csv',index_col=0) 
            Dead_Vacc_Data = pd.read_csv('../models/combined_results/' +Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_DeadVacc.csv',index_col=0) 
            
             # Give appropriate x-axis
            x = np.linspace(0,18,7)  
            x_vacc = np.add(x,1)       
            X = np.repeat(x,np.shape(Infectious_Data)[0])
            X_Vacc = np.repeat(x_vacc,np.shape(Infectious_Data)[0])
            X_axis = np.zeros((2,np.size(x)))
            X_axis[0] = x
            X_axis[1] = x_vacc
            Infectious_Data_Vec = pd.DataFrame(dict(x=X,y =np.reshape(np.array(Infectious_Data).T,(np.size(Infectious_Data)))))
            Infectious_Vacc_Data_Vec = pd.DataFrame(dict(x=X_Vacc,y =np.reshape(np.array(Infectious_Vacc_Data).T,(np.size(Infectious_Vacc_Data)))))
            Critical_Data_Vec = pd.DataFrame(dict(x=X,y =np.reshape(np.array(Critical_Data).T,(np.size(Critical_Data)))))
            Critical_Vacc_Data_Vec = pd.DataFrame(dict(x=X_Vacc,y =np.reshape(np.array(Critical_Vacc_Data).T,(np.size(Critical_Vacc_Data)))))
            Dead_Data_Vec = pd.DataFrame(dict(x=X,y =np.reshape(np.array(Dead_Data).T,(np.size(Dead_Data)))))
            Dead_Vacc_Data_Vec = pd.DataFrame(dict(x=X_Vacc,y =np.reshape(np.array(Dead_Vacc_Data).T,(np.size(Dead_Vacc_Data)))))

            # Making the figure
            fig, ax = plt.subplots(ncols = 1, nrows = 3, figsize=(15,15))
            # Box plot
            plot_I = sns.boxplot(x ="x", y = "y",data = Infectious_Data_Vec,ax = ax[0], color=col_infected,order =np.arange(np.max(x_vacc)+1))        
            plot_C = sns.boxplot(x ="x", y = "y",data = Critical_Data_Vec,ax = ax[1], color=col_critical,order =np.arange(np.max(x_vacc)+1))     
            plot_D = sns.boxplot(x ="x", y = "y",data = Dead_Data_Vec,ax = ax[2], color=col_dead,order =np.arange(np.max(x_vacc)+1))     
         
            plot_I = sns.boxplot(x = "x",y = "y",data= Infectious_Vacc_Data_Vec,ax = ax[0], color=np.multiply(col_infected,alph1),order =np.arange(np.max(x_vacc)+1))        
            plot_C = sns.boxplot(x ="x", y = "y",data = Critical_Vacc_Data_Vec,ax = ax[1],color=np.multiply(col_critical,alph1),order =np.arange(np.max(x_vacc)+1))     
            plot_D = sns.boxplot(x ="x", y = "y",data = Dead_Vacc_Data_Vec,ax = ax[2], color=np.multiply(col_dead,alph1),order =np.arange(np.max(x_vacc)+1)) 
            # Scatter on top
            plot_I = sns.stripplot(x ="x", y = "y",data = Infectious_Data_Vec,ax = ax[0], color=col_infected,order =np.arange(np.max(x_vacc)+1),size=swarmsize,alpha = alph_swarm)        
            plot_C = sns.stripplot(x ="x", y = "y",data = Critical_Data_Vec,ax = ax[1], color=col_critical,order =np.arange(np.max(x_vacc)+1),size=swarmsize,alpha = alph_swarm)     
            plot_D = sns.stripplot(x ="x", y = "y",data = Dead_Data_Vec,ax = ax[2], color=col_dead,order =np.arange(np.max(x_vacc)+1),size=swarmsize,alpha = alph_swarm)     
         
            plot_I = sns.stripplot(x = "x",y = "y",data= Infectious_Vacc_Data_Vec,ax = ax[0], color=np.multiply(col_infected,alph1),order =np.arange(np.max(x_vacc)+1),size=swarmsize,alpha = alph_swarm)        
            plot_C = sns.stripplot(x ="x", y = "y",data = Critical_Vacc_Data_Vec,ax = ax[1], color=np.multiply(col_critical,alph1),order =np.arange(np.max(x_vacc)+1),size=swarmsize,alpha = alph_swarm)     
            plot_D = sns.stripplot(x ="x", y = "y",data = Dead_Vacc_Data_Vec,ax = ax[2], color=np.multiply(col_dead,alph1),order =np.arange(np.max(x_vacc)+1),size=swarmsize,alpha = alph_swarm) 

            # "Artificial" plots to get colours for legend
           
            ax[0].bar(-10,1, color=col_infected,label = 'Infected')
            ax[0].bar(-10,1, color=np.multiply(col_infected,alph1),label = 'Infected and Vaccinated')
            ax[0].legend(loc = 'upper center')
            ax[0].ticklabel_format(axis = 'y',style= 'sci')
            ax[1].bar(-10,1, color=col_critical,label = 'Critical')
            ax[1].bar(-10,1, color=np.multiply(col_critical,alph1),label = 'Critical and Vaccinated')
            ax[1].legend(loc = 'upper left')
            ax[2].bar(-10,1, color=col_dead,label = 'Dead')
            ax[2].bar(-10,1, color=np.multiply(col_dead,alph1),label = 'Dead and Vaccinated')
            ax[2].legend(loc = 'upper left')

            # Make neat
            if agespec == 'NonSpec':
            
                ax[0].set_title(Fig_Titles[ii]+' (Non-Specific)')
            else:
                ax[0].set_title(Fig_Titles[ii])
            # Set axis for all
            for jj in range(3):
                
                ax[jj].set_xticks(np.mean(X_axis,0))  
                ax[jj].set_xlabel('') 
                ax[jj].set_ylabel('') 
                ax[jj].tick_params(axis='x', which='major', width=2.5, length=10)
                ax[jj].set_xlim([np.min(X_axis)-0.5,np.max(X_axis)+0.5]) 
                if jj == 2:                    
                    ax[jj].set_xticks(np.mean(X_axis,0))
                    ax[jj].tick_params('x',rotation = 45)    
                    ax[jj].set_xticklabels([AgeNames[x] for x in range(np.size(AgeNames))])
        
            # ax[0].set_ylim([0,Infectious_Data.max().max()*1.5]) 
            # ax[1].set_ylim([0,Critical_Data.max().max()*1.5]) 
            # ax[2].set_ylim([0,Dead_Data.max().max()*1.5]) 
            if variant=='delta':        
                
                if ii < 2:
                    ax[0].set_ylim([0,110000]) 
                    ax[1].set_ylim([0,6000]) 
                    ax[2].set_ylim([0,3000])    
                else:
                    ax[0].set_ylim([0,55000]) 
                    ax[1].set_ylim([0,2000]) 
                    ax[2].set_ylim([0,1500])   


            if variant=='omicron':                        
                ax[0].set_ylim([0,1000000]) 
                ax[1].set_ylim([0,25000]) 
                ax[2].set_ylim([0,10000]) 

            ax[2].set_xlabel('Age-Group') 
        
            # ax[0].set_ylabel('Number of People') 
            ax[1].set_ylabel('Number of People')
            # ax[2].set_ylabel('Number of People')
                # ax[0,ii].legend(loc = "upper left", ncol =3 ) 
                # ax[1,ii].legend(loc = "upper left", ncol =3 ) 
                # ax[2,ii].legend(loc = "upper left", ncol =3 ) 
            # fig.show()
            # pdb.set_trace()
            fig.tight_layout()
            fig.savefig(Sim_Names[ii] +'_'+variant+'_'+agespec+'_'+'boxplot_AgeSpec_Day_90.jpg',dpi=300)


