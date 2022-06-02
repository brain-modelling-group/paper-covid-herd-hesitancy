import numpy as np
import pandas as pd
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

            # save to file
            Infectious_Data.to_csv(Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_Infectious.csv') 
            Critical_Data.to_csv(Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_Critical.csv') 
            Dead_Data.to_csv(Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_Dead.csv') 
            Infectious_Vacc_Data.to_csv(Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_InfectiousVacc.csv') 
            Critical_Vacc_Data.to_csv(Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_CriticalVacc.csv') 
            Dead_Vacc_Data.to_csv(Sim_Names[ii] + '_' +variant +'_' +agespec+'_All_DeadVacc.csv') 
         
            # Output Medians
        
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+ 'Infections'] = Infectious_Data.median()           
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+ 'Infections and Vaccinated'] = Infectious_Vacc_Data.median()
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+'Critical'] = Critical_Data.median()
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+ 'Critical and Vaccinated'] = Critical_Vacc_Data.median()
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+ 'Dead'] = Dead_Data.median()
            Median[Sim_Names[ii] +' '+ variant +' ' + agespec +' '+'Dead and Vaccinated'] = Dead_Vacc_Data.median()     

Median = pd.DataFrame(Median)
Median.to_csv('AgeSpecific_Medians.csv')