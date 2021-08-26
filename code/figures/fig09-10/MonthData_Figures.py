# Extract and average the data from all the single hesitancy runs of the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved data and create figures
DataMonths = ['January','April']
DataTypes = ['AgeSpec','Dynamics']
HesitancyTypes = ['','NonSpec'] # Chose age-specific or non-specific hesitancy
HTypeLabel = ['','Non-Specific ']
Array_Nums = np.linspace(0,99,100) # Number for repititions
Cluster_Sizes = np.linspace(1,20,20) # Imported infections

fsize_title = 22
fsize_legend = 13
fsize_labels= 16
fsize_numbers = 11

plt.rc('font', size=fsize_numbers)          # controls default text sizes
plt.rc('axes', titlesize=fsize_labels)     # fontsize of the axes title
plt.rc('axes', labelsize=fsize_labels)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsize_legend)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fsize_legend)    # fontsize of the tick labels
plt.rc('legend', fontsize=fsize_legend)    # legend fontsize
plt.rc('figure', titlesize=fsize_title)  # fontsize of the figure title


Cluster_Size = 20
pop_size = 5.1e6

AgeLabels = ['0-17','18-24','25-34','35-44','45-54','55-64','65+']
                 
# TODO: 4 loops here, month, DType, HType, Cluster_Size
         
# Start with month 1 and age-specific
filename_agespec = 'Combined' + DataMonths[0] + HesitancyTypes[1] +'Delta_' + DataTypes[0] + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'

AgeSpecificData = pd.read_csv('Data/' + filename_agespec,index_col=0)

scale = 1
# PlotAgeSpecificData(AgeSpecificData) 
# # Data of Infected, Infected and Vacc, Dead, Dead and Vacc

data1_25_34  = np.multiply([AgeSpecificData['25-34']['Mean_Infected']                
                 ],scale)

data1_45_54  = np.multiply([AgeSpecificData['45-54']['Mean_Infected']                
                 ],scale)

data1_65_plus  = np.multiply([AgeSpecificData['65+']['Mean_Infected']            
                 ],scale)


Vdata1_25_34  = np.multiply([AgeSpecificData['25-34']['Mean_Infected_Vaccinated'],                
                 ],scale)

Vdata1_45_54  = np.multiply([AgeSpecificData['45-54']['Mean_Infected_Vaccinated'],                
                 ],scale)

Vdata1_65_plus  = np.multiply([AgeSpecificData['65+']['Mean_Infected_Vaccinated']              
                 ],scale)

# # Data of Diagnosed, Diagnosed and Vacc, Critical, Critical and Vacc

data2_25_34  = np.multiply([AgeSpecificData['25-34']['Mean_Critical'],                             
                 AgeSpecificData['25-34']['Mean_Dead'], 
                 ],scale)

data2_45_54  = np.multiply([AgeSpecificData['45-54']['Mean_Critical'], 
                 AgeSpecificData['45-54']['Mean_Dead'],
                 ],scale)


data2_65_plus   = np.multiply([AgeSpecificData['65+']['Mean_Critical'],                                                                
                 AgeSpecificData['65+']['Mean_Dead'],
                 ],scale)        

Vdata2_25_34  = np.multiply([AgeSpecificData['25-34']['Mean_Critical_Vaccinated'],
                 AgeSpecificData['25-34']['Mean_Dead_Vaccinated'], 
                 ],scale)

Vdata2_45_54  = np.multiply([AgeSpecificData['45-54']['Mean_Critical_Vaccinated'],
                 AgeSpecificData['45-54']['Mean_Dead_Vaccinated'], 
                 ],scale)

Vdata2_65_plus   = np.multiply([AgeSpecificData['65+']['Mean_Critical_Vaccinated'], 
                 AgeSpecificData['65+']['Mean_Dead_Vaccinated'], 
                 ],scale)       

# Month 2, age-specific
filename_agespec = 'Combined' + DataMonths[1] + HesitancyTypes[1] +'Delta_' + DataTypes[0] + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'

AgeSpecificData = pd.read_csv('Data/' + filename_agespec,index_col=0)

scale = 1
# PlotAgeSpecificData(AgeSpecificData) 
# # Data of Infected, Infected and Vacc, Dead, Dead and Vacc

data1_25_34 = [data1_25_34,np.multiply([AgeSpecificData['25-34']['Mean_Infected']                
                 ],scale)]

data1_45_54 = [data1_45_54,np.multiply([AgeSpecificData['45-54']['Mean_Infected']                
                 ],scale)]

data1_65_plus = [data1_65_plus,np.multiply([AgeSpecificData['65+']['Mean_Infected']            
                 ],scale)]


Vdata1_25_34 = [Vdata1_25_34,np.multiply([AgeSpecificData['25-34']['Mean_Infected_Vaccinated'],                
                 ],scale)]

Vdata1_45_54 = [Vdata1_45_54,np.multiply([AgeSpecificData['45-54']['Mean_Infected_Vaccinated'],                
                 ],scale)]

Vdata1_65_plus = [Vdata1_65_plus,np.multiply([AgeSpecificData['65+']['Mean_Infected_Vaccinated']              
                 ],scale)]

# # Data of Diagnosed, Diagnosed and Vacc, Critical, Critical and Vacc

data2_25_34 = [data2_25_34,np.multiply([AgeSpecificData['25-34']['Mean_Critical'],                             
                 AgeSpecificData['25-34']['Mean_Dead'], 
                 ],scale)]

data2_45_54 = [data2_45_54,np.multiply([AgeSpecificData['45-54']['Mean_Critical'], 
                 AgeSpecificData['45-54']['Mean_Dead'],
                 ],scale)]


data2_65_plus = [data2_65_plus,np.multiply([AgeSpecificData['65+']['Mean_Critical'],                                                                
                 AgeSpecificData['65+']['Mean_Dead'],
                 ],scale)]     

Vdata2_25_34 = [Vdata2_25_34,np.multiply([AgeSpecificData['25-34']['Mean_Critical_Vaccinated'],
                 AgeSpecificData['25-34']['Mean_Dead_Vaccinated'], 
                 ],scale)]

Vdata2_45_54 = [Vdata2_45_54,np.multiply([AgeSpecificData['45-54']['Mean_Critical_Vaccinated'],
                 AgeSpecificData['45-54']['Mean_Dead_Vaccinated'], 
                 ],scale)]

Vdata2_65_plus = [Vdata2_65_plus, np.multiply([AgeSpecificData['65+']['Mean_Critical_Vaccinated'], 
                 AgeSpecificData['65+']['Mean_Dead_Vaccinated'], 
                 ],scale)       ]

            
data_labels1 = ["Infected \n and Vaccinated"]
data_labels2 = ["Critical \n and Vaccinated", "Dead \n and Vaccinated"]
age_labels = ['18-24','25-34','35-44','45-54','55-64','65+'] 

# Sort out colours here
alph1 = 0.5
alph2 = 0.9
col_grad = np.linspace(0.6,1,np.size(age_labels))
col_infected = [150/255,250/255,150/255]
col_critical = [220/255,20/255,60/255]
col_dead = [0.5,0.5,0.5]
        
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,9))
xticks1 = 0
   
width = 0.16  # the width of the bars
width_bars = 0.14
width_sep = width_bars + 0.005
w_adj = (np.sqrt(5)-1)/2

x_pos = [xticks1-3.5*width, xticks1-2.5*width,xticks1-1/2*width,
         xticks1+1/2*width, xticks1+1.5*width,xticks1+3.5*width]
x_pos = [0,xticks1-width,0,xticks1+width,0,xticks1+3*width ]

d12 = ax[0].bar(np.add(x_pos[1],-1/2*width_sep),  data1_25_34[0] ,   width_bars, color=np.multiply(col_grad[1],col_infected),alpha=alph1)
d14 = ax[0].bar(np.add(x_pos[3],-1/2*width_sep),  data1_45_54[0]   , width_bars, color=np.multiply(col_grad[3],col_infected),  alpha=alph1)
d16 = ax[0].bar(np.add(x_pos[5],-1/2*width_sep),  data1_65_plus[0]  , width_bars, color=np.multiply(col_grad[5],col_infected), alpha=alph1,label='Infected')
         
ax[0].bar(np.add(x_pos[1],-1/2*width_sep),  Vdata1_25_34[0] ,   width_bars*w_adj,color=np.multiply(col_grad[1],col_infected),alpha=alph2)
ax[0].bar(np.add(x_pos[3],-1/2*width_sep),  Vdata1_45_54[0]   , width_bars*w_adj, color=np.multiply(col_grad[3],col_infected),alpha=alph2)
ax[0].bar(np.add(x_pos[5],-1/2*width_sep),  Vdata1_65_plus[0]  , width_bars*w_adj, color=np.multiply(col_grad[5],col_infected),alpha=alph2,label = 'Infected \n and Vaccinated') 

d22 = ax[0].bar(np.add(x_pos[1],+1/2*width_sep),  data1_25_34[1] ,   width_bars, color=np.multiply(col_grad[1],col_infected),alpha=alph1)
d24 = ax[0].bar(np.add(x_pos[3],+1/2*width_sep),  data1_45_54[1]   , width_bars, color=np.multiply(col_grad[3],col_infected),  alpha=alph1)
d26 = ax[0].bar(np.add(x_pos[5],+1/2*width_sep),  data1_65_plus[1]  , width_bars, color=np.multiply(col_grad[5],col_infected), alpha=alph1)
         
ax[0].bar(np.add(x_pos[1],+1/2*width_sep),  Vdata1_25_34[1] ,   width_bars*w_adj,color=np.multiply(col_grad[1],col_infected),alpha=alph2)
ax[0].bar(np.add(x_pos[3],+1/2*width_sep),  Vdata1_45_54[1]   , width_bars*w_adj, color=np.multiply(col_grad[3],col_infected),alpha=alph2)
ax[0].bar(np.add(x_pos[5],+1/2*width_sep),  Vdata1_65_plus[1]  , width_bars*w_adj, color=np.multiply(col_grad[5],col_infected),alpha=alph2) 



# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0].set_ylabel('Number of People')
ax[0].set_xlabel('Age Category')
# ax[0].set_title('Comparing Months')
x_pos_label = [x_pos[1],x_pos[3],x_pos[5]]
age_use_label = [age_labels[1],age_labels[3],age_labels[5]]
ax[0].set_xticks(x_pos_label)
ax[0].set_xticklabels(age_use_label)
ax[0].set_ylim([0, 25000])
ax[0].legend(loc="upper right", ncol = 2)

ax[0].bar_label(d12, labels =['January \n' + str(round(data1_25_34[0][0])) ],size = 16)
ax[0].bar_label(d22, labels =['April \n' + str(round(data1_25_34[1][0]))],size = 16)
ax[0].bar_label(d14, labels =[str(round(data1_45_54[0][0])) ],size = 16)
ax[0].bar_label(d24, labels =[str(round(data1_45_54[1][0])) ],size = 16)
ax[0].bar_label(d16, labels =[str(round(data1_65_plus[0][0])) ],size = 16)
ax[0].bar_label(d26, labels =[str(round(data1_65_plus[1][0])) ],size = 16)
# ax[0].bar_label(d2, labels = np.round(data1_25_34,2), padding=3)
# ax[0].bar_label(d3, labels = np.round(data1_35_44,2), padding=3)
# ax[0].bar_label(d4, labels = np.round(data1_45_54,2), padding=3)
# ax[0].bar_label(d5, labels = np.round(data1_55_64,2), padding=3)
# ax[0].bar_label(d6, labels = np.round(data1_65_plus,2), padding=3)

xticks2 = 0 # Critical
dcj1 = ax[1].bar(x_pos[1]-1/2*width_sep,  data2_25_34[0][0],   width_bars, color=np.multiply(col_grad[1],col_critical), alpha=alph1)
dcj2=ax[1].bar(x_pos[3]-1/2*width_sep,  data2_45_54[0][0], width_bars, color=np.multiply(col_grad[3],col_critical),  alpha=alph1)
dcj3=ax[1].bar(x_pos[5]-1/2*width_sep,  data2_65_plus[0][0], width_bars, color=np.multiply(col_grad[5],col_critical), alpha=alph1, label = 'Critical')
           
ax[1].bar(x_pos[1]-1/2*width_sep,  Vdata2_25_34[0][0],   width_bars*w_adj,color=np.multiply(col_grad[1],col_critical),alpha=alph2)
ax[1].bar(x_pos[3]-1/2*width_sep,  Vdata2_45_54[0][0], width_bars*w_adj, color=np.multiply(col_grad[3],col_critical),alpha=alph2)
ax[1].bar(x_pos[5]-1/2*width_sep,  Vdata2_65_plus[0][0], width_bars*w_adj, color=np.multiply(col_grad[5],col_critical),alpha=alph2,label = 'Critical \n and Vaccinated') 
       
xticks2 = 1 # Deaths
ddj1 = ax[1].bar(x_pos[1]+xticks2 -1/2*width_sep,  data2_25_34[0][1],   width_bars, color=np.multiply(col_grad[1],col_dead), alpha=alph1)
ddj2 = ax[1].bar(x_pos[3]+xticks2 -1/2*width_sep,  data2_45_54[0][1], width_bars, color=np.multiply(col_grad[3],col_dead),  alpha=alph1)
ddj3 = ax[1].bar(x_pos[5]+xticks2 -1/2*width_sep,  data2_65_plus[0][1], width_bars, color=np.multiply(col_grad[5],col_dead), alpha=alph1, label = 'Dead')

ax[1].bar(x_pos[1]+xticks2 -1/2*width_sep,  Vdata2_25_34[0][1],   width_bars*w_adj,color=np.multiply(col_grad[1],col_dead),alpha=alph2)
ax[1].bar(x_pos[3]+xticks2   -1/2*width_sep,  Vdata2_45_54[0][1], width_bars*w_adj, color=np.multiply(col_grad[3],col_dead),alpha=alph2)
ax[1].bar(x_pos[5]+xticks2 -1/2*width_sep,  Vdata2_65_plus[0][1], width_bars*w_adj, color=np.multiply(col_grad[5],col_dead),alpha=alph2, label = 'Dead \n and Vaccinated') 
  

xticks2 = 0 # Critical
dca1 = ax[1].bar(x_pos[1] +1/2*width_sep,  data2_25_34[1][0],   width_bars, color=np.multiply(col_grad[1],col_critical), alpha=alph1)
dca2 = ax[1].bar(x_pos[3] +1/2*width_sep,  data2_45_54[1][0], width_bars, color=np.multiply(col_grad[3],col_critical),  alpha=alph1)
dca3 = ax[1].bar(x_pos[5] +1/2*width_sep,  data2_65_plus[1][0], width_bars, color=np.multiply(col_grad[5],col_critical), alpha=alph1)
           
ax[1].bar(x_pos[1] +1/2*width_sep,  Vdata2_25_34[1][0],   width_bars*w_adj,color=np.multiply(col_grad[1],col_critical),alpha=alph2)
ax[1].bar(x_pos[3] +1/2*width_sep,  Vdata2_45_54[1][0], width_bars*w_adj, color=np.multiply(col_grad[3],col_critical),alpha=alph2)
ax[1].bar(x_pos[5] +1/2*width_sep,  Vdata2_65_plus[1][0], width_bars*w_adj, color=np.multiply(col_grad[5],col_critical),alpha=alph2) 


xticks2 = 1 # Deaths

dda1 = ax[1].bar(x_pos[1] +xticks2 +1/2*width_sep,  data2_25_34[1][1],   width_bars, color=np.multiply(col_grad[1],col_dead), alpha=alph1)
dda2 = ax[1].bar(x_pos[3] +xticks2 +1/2*width_sep,  data2_45_54[1][1], width_bars, color=np.multiply(col_grad[3],col_dead),  alpha=alph1)
dda3 = ax[1].bar(x_pos[5] +xticks2 +1/2*width_sep,  data2_65_plus[1][1], width_bars, color=np.multiply(col_grad[5],col_dead), alpha=alph1)

ax[1].bar(x_pos[1]+xticks2 +1/2*width_sep,  Vdata2_25_34[1][1],   width_bars*w_adj,color=np.multiply(col_grad[1],col_dead),alpha=alph2)
ax[1].bar(x_pos[3]+xticks2 +1/2*width_sep,  Vdata2_45_54[1][1], width_bars*w_adj, color=np.multiply(col_grad[3],col_dead),alpha=alph2)
ax[1].bar(x_pos[5]+xticks2 +1/2*width_sep,  Vdata2_65_plus[1][1], width_bars*w_adj, color=np.multiply(col_grad[5],col_dead),alpha=alph2) 




       
ax[1].bar_label(dcj1, labels =['Jan. \n ' + str(round(data2_25_34[0][0]))],size = 16)
ax[1].bar_label(ddj1, labels =['Jan. \n ' + str(round(data2_25_34[0][1]))],size = 16)
ax[1].bar_label(dca1, labels =['Apr. \n ' + str(round(data2_25_34[1][0]))],size = 16)
ax[1].bar_label(dda1, labels =['Apr. \n ' + str(round(data2_25_34[1][1]))],size = 16)
ax[1].bar_label(dcj2, labels =[ str(round(data2_45_54[0][0]))],size = 16)
ax[1].bar_label(ddj2, labels =[ str(round(data2_45_54[0][1]))],size = 16)
ax[1].bar_label(dca2, labels =[str(round(data2_45_54[1][0]))],size = 16)
ax[1].bar_label(dda2, labels =[str(round(data2_45_54[1][1]))],size = 16)
ax[1].bar_label(dcj3, labels =[ str(round(data2_65_plus[0][0]))],size = 16)
ax[1].bar_label(ddj3, labels =[ str(round(data2_65_plus[0][1]))],size = 16)
ax[1].bar_label(dca3, labels =[str(round(data2_65_plus[1][0]))],size = 16)
ax[1].bar_label(dda3, labels =[str(round(data2_65_plus[1][1]))],size = 16)


# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1].set_ylabel('Number of People')        
ax[1].set_xlabel('Age Category')
x_pos1 = [x + xticks2 for x in x_pos_label]
ticks = x_pos_label+x_pos1
ax[1].set_xticks(ticks)
ax[1].set_xticklabels(age_use_label+age_use_label)          
ax[1].legend(loc="upper left", ncol = 4)
# ax[1].legend()

# ax[1].bar_label(ax[1].patches[0], labels = data2_18_24.astype(int), padding=3)
# ax[1].bar_label(d12, labels = data2_25_34.astype(int), padding=3)
# ax[1].bar_label(d13, labels = data2_35_44.astype(int), padding=3)
# ax[1].bar_label(d14, labels = data2_45_54.astype(int), padding=3)
# ax[1].bar_label(d15, labels = data2_55_64.astype(int), padding=3)
# ax[1].bar_label(d16, labels = data2_65_plus.astype(int), padding=3)

# ax[1].bar_label(d21, labels = data2_18_24.astype(int)[1], padding=3)
# ax[1].bar_label(d22, labels = data2_25_34.astype(int)[1], padding=3)
# ax[1].bar_label(d23, labels = data2_35_44.astype(int)[1], padding=3)
# ax[1].bar_label(d24, labels = data2_45_54.astype(int)[1], padding=3)
# ax[1].bar_label(d25, labels = data2_55_64.astype(int)[1], padding=3)
# ax[1].bar_label(d26, labels = data2_65_plus.astype(int)[1], padding=3)
ax[1].set_yscale('log')
ax[1].set_ylim([1, 5e5])
# ax[1].set_ylim([1, 18000])
fig.savefig('Figures/AgeSpecResults_CompareMonths.pdf')

############################# Dynamics #####################################


fsize_title = 22
fsize_legend = 16
fsize_labels= 22
fsize_numbers = 16


plt.rc('font', size=fsize_numbers)          # controls default text sizes
plt.rc('axes', titlesize=fsize_labels)     # fontsize of the axes title
plt.rc('axes', labelsize=fsize_labels)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsize_legend)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fsize_legend)    # fontsize of the tick labels
plt.rc('legend', fontsize=fsize_legend)    # legend fontsize
plt.rc('figure', titlesize=fsize_title)  # fontsize of the figure title




# Load the saved data and create figures
DataMonths = ['January','April']
DataTypes = ['AgeSpec','Dynamics']
HesitancyTypes = ['','NonSpec'] # Chose age-specific or non-specific hesitancy
HTypeLabel = ['Age-Specific','Non-Specific ']
Array_Nums = np.linspace(0,99,100) # Number for repititions
Cluster_Sizes = np.linspace(1,20,20) # Imported infections
TitleLabel = ['January 2021', 'April 2021']
pop_size = 5.1e6

 

DType = DataTypes[1] # Chose datatype to combine

Cluster_Size = int(20) # Pick values to use    

cumsum = 0
  # set 0 for raw, 1 for cum_sum


col_crit = np.multiply([255/255,165/255,0],1)
# col_crit1 = np.multiply([255/255,165/255,0],1)
col_crit2 =  np.multiply([255/255,135/255,0],1)
col_dead = np.multiply([0.5,0.5,0.5],1)
col_dead2 = np.multiply([0.5,0.5,0.5],0.6)
  
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))
# Critical Condition: Age- and Non-Spec



# Load two files: Age-specific and non-specific
filename_agespec = 'Combined' + DataMonths[0] + HesitancyTypes[0] +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'  
DynamicsData_AgeSpec = pd.read_csv('Data/' + filename_agespec,index_col=0)
    
filename_nonspec = 'Combined' + DataMonths[0] + HesitancyTypes[1] +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'  
DynamicsData_NonSpec = pd.read_csv('Data/' + filename_nonspec,index_col=0)

print(np.cumsum(DynamicsData_AgeSpec['Mean_Infections'])  )
print(np.cumsum(DynamicsData_AgeSpec['Mean_Critical'])  )
print(np.cumsum(DynamicsData_AgeSpec['Mean_Deaths'])  )

# If cumsum==1, convert values to cumsum
if cumsum==1:
    DynamicsData_AgeSpec['Mean_Critical'] = np.cumsum(DynamicsData_AgeSpec['Mean_Critical'])
    DynamicsData_AgeSpec['Mean_Deaths'] = np.cumsum(DynamicsData_AgeSpec['Mean_Deaths'])
    DynamicsData_AgeSpec['SD_Critical'] = np.cumsum(DynamicsData_AgeSpec['SD_Critical'])
    DynamicsData_AgeSpec['SD_Deaths'] = np.cumsum(DynamicsData_AgeSpec['SD_Deaths'])

# Create time-series
t_steps = np.size(DynamicsData_AgeSpec['Mean_Infections'])      
t = np.linspace(1,t_steps,t_steps)# Plot infections and deaths


t_use = 90
# # Critical
# ax[0].plot(t[0:t_use], DynamicsData_AgeSpec['Mean_Critical'][0:t_use], color = col_crit,label = HTypeLabel[0])
# ax[0].fill_between(t[0:t_use], 
#                   DynamicsData_AgeSpec['Mean_Critical'][0:t_use]-DynamicsData_AgeSpec['SD_Critical'][0:t_use], 
#                   DynamicsData_AgeSpec['Mean_Critical'][0:t_use]+DynamicsData_AgeSpec['SD_Critical'][0:t_use],
#                   color =  col_crit,
#                   alpha=0.5)      

      
# ax[0].axhline(y = 351, color = 'r', linestyle = '--',label='ICU Capacity QLD')    

# ax[0].legend(loc="upper left")

# if cumsum==1:
#     ax[0].set_ylabel('Cumulative Number of Critical Conditions')
#     ax[0].set_ylim([0, 1500])
# else:
#     ax[0].set_ylabel('Number of New Critical Conditions')
#     ax[0].set_ylim([0, 1500])
# # ax[0].set_xlabel('Time [days]')   
# ax[0].set_xlim([1, 90])
# ax[0].set_title(TitleLabel[0])

#  # Next month in list
# filename_agespec = 'Combined' + DataMonths[1] + HesitancyTypes[0] +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'  
# DynamicsData_AgeSpec = pd.read_csv('Data/' + filename_agespec,index_col=0)
    

# if cumsum==1:
#     DynamicsData_AgeSpec['Mean_Critical'] = np.cumsum(DynamicsData_AgeSpec['Mean_Critical'])
#     DynamicsData_AgeSpec['Mean_Deaths'] = np.cumsum(DynamicsData_AgeSpec['Mean_Deaths'])
#     DynamicsData_AgeSpec['SD_Critical'] = np.cumsum(DynamicsData_AgeSpec['SD_Critical'])
#     DynamicsData_AgeSpec['SD_Deaths'] = np.cumsum(DynamicsData_AgeSpec['SD_Deaths'])


# # Critical
# ax[1].plot(t[0:t_use], DynamicsData_AgeSpec['Mean_Critical'][0:t_use], color = col_crit,label = HTypeLabel[0])
# ax[1].fill_between(t[0:t_use], 
#                   DynamicsData_AgeSpec['Mean_Critical'][0:t_use]-DynamicsData_AgeSpec['SD_Critical'][0:t_use], 
#                   DynamicsData_AgeSpec['Mean_Critical'][0:t_use]+DynamicsData_AgeSpec['SD_Critical'][0:t_use],
#                   color =  col_crit,
#                   alpha=0.5)      

      
# ax[1].axhline(y = 351, color = 'r', linestyle = '--',label='ICU Capacity QLD')    
# ax[1].legend(loc="upper left")
# ax[1].set_xlabel('Time [days]')   
# ax[1].set_title(TitleLabel[1])
# if cumsum==1:
#     ax[1].set_ylim([0, 1500])
# else:
#     ax[1].set_ylim([0, 1500])
# ax[0].set_xlim([1, t_use])
# ax[1].set_xlim([1, t_use])


# ax[0].set_xlim([1, t_use])

# fig.savefig('Figures/CompareCriticalDynamics_'  +'Delta_ClusterSize_' + str(int(Cluster_Size))+'.pdf')

  # Critical

ax.fill_between(t[0:t_use], 
                  DynamicsData_AgeSpec['Mean_Critical'][0:t_use]-DynamicsData_AgeSpec['SD_Critical'][0:t_use], 
                  DynamicsData_AgeSpec['Mean_Critical'][0:t_use]+DynamicsData_AgeSpec['SD_Critical'][0:t_use],
                  color =  col_crit,
                  alpha=0.25)      
ax.plot(t[0:t_use], DynamicsData_AgeSpec['Mean_Critical'][0:t_use], color = col_crit,
        label = 'January 2021',
        linewidth=4)
      
ax.axhline(y = 351, color = 'r', linestyle = '--')    

ax.legend(loc="upper left")

if cumsum==1:
    ax.set_ylabel('Cumulative Number of Critical Conditions')
    ax.set_ylim([0, 1500])
else:
    ax.set_ylabel('Number of New Critical Conditions')
    ax.set_ylim([0, 1500])
# ax[0].set_xlabel('Time [days]')   
ax.set_xlim([1, 90])
# ax.set_title(TitleLabel[0])

  # Next month in list
filename_agespec = 'Combined' + DataMonths[1] + HesitancyTypes[0] +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'  
DynamicsData_AgeSpec = pd.read_csv('Data/' + filename_agespec,index_col=0)
    
print(np.cumsum(DynamicsData_AgeSpec['Mean_Infections'])  )
print(np.cumsum(DynamicsData_AgeSpec['Mean_Critical'])  )
print(np.cumsum(DynamicsData_AgeSpec['Mean_Deaths'])  )


if cumsum==1:
    DynamicsData_AgeSpec['Mean_Critical'] = np.cumsum(DynamicsData_AgeSpec['Mean_Critical'])
    DynamicsData_AgeSpec['Mean_Deaths'] = np.cumsum(DynamicsData_AgeSpec['Mean_Deaths'])
    DynamicsData_AgeSpec['SD_Critical'] = np.cumsum(DynamicsData_AgeSpec['SD_Critical'])
    DynamicsData_AgeSpec['SD_Deaths'] = np.cumsum(DynamicsData_AgeSpec['SD_Deaths'])


# Critical

ax.fill_between(t[0:t_use], 
                  DynamicsData_AgeSpec['Mean_Critical'][0:t_use]-DynamicsData_AgeSpec['SD_Critical'][0:t_use], 
                  DynamicsData_AgeSpec['Mean_Critical'][0:t_use]+DynamicsData_AgeSpec['SD_Critical'][0:t_use],
                  color =  col_crit2,
                  alpha=0.25)      
ax.plot(t[0:t_use], DynamicsData_AgeSpec['Mean_Critical'][0:t_use], color = col_crit2,label = 'April 2021',linewidth=4)
      

ax.axhline(y = 351, color = 'r', linestyle = '--',label='ICU Capacity QLD')    
ax.legend(loc="upper left")
ax.set_xlabel('Time [days]')   
# ax.set_title(TitleLabel[1])
if cumsum==1:
    ax.set_ylim([0, 1500])
else:
    ax.set_ylim([0, 1000])
ax.set_xlim([1, t_use])

ax.set_xlim([1, t_use])

fig.savefig('Figures/CompareCriticalDynamics_'  +'Delta_ClusterSize_' + str(int(Cluster_Size))+'.pdf')





## Deaths
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))
# Critical Condition: Age- and Non-Spec

# Load two files: Age-specific and non-specific
filename_agespec = 'Combined' + DataMonths[0] + HesitancyTypes[0] +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'  
DynamicsData_AgeSpec = pd.read_csv('Data/' + filename_agespec,index_col=0)
    

# Create time-series
t_steps = np.size(DynamicsData_AgeSpec['Mean_Deaths'])      
t = np.linspace(1,t_steps,t_steps)# Plot infections and deaths


t_use = 90
ax.fill_between(t[0:t_use], 
                  DynamicsData_AgeSpec['Mean_Deaths'][0:t_use]-DynamicsData_AgeSpec['SD_Deaths'][0:t_use], 
                  DynamicsData_AgeSpec['Mean_Deaths'][0:t_use]+DynamicsData_AgeSpec['SD_Deaths'][0:t_use],
                  color =  col_dead,
                  alpha=0.25)      
ax.plot(t[0:t_use], DynamicsData_AgeSpec['Mean_Deaths'][0:t_use], color = col_dead,
        label = 'January 2021',
        linewidth=4)
ax.legend(loc="upper left")

if cumsum==1:
    ax.set_ylabel('Cumulative Number of Deaths')
    ax.set_ylim([0, 1500])
else:
    ax.set_ylabel('Number of New Deaths')
    ax.set_ylim([0, 1500])
# ax[0].set_xlabel('Time [days]')   
ax.set_xlim([1, 90])
# ax.set_title(TitleLabel[0])

  # Next month in list
filename_agespec = 'Combined' + DataMonths[1] + HesitancyTypes[0] +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'  
DynamicsData_AgeSpec = pd.read_csv('Data/' + filename_agespec,index_col=0)
    

if cumsum==1:
    DynamicsData_AgeSpec['Mean_Critical'] = np.cumsum(DynamicsData_AgeSpec['Mean_Critical'])
    DynamicsData_AgeSpec['Mean_Deaths'] = np.cumsum(DynamicsData_AgeSpec['Mean_Deaths'])
    DynamicsData_AgeSpec['SD_Critical'] = np.cumsum(DynamicsData_AgeSpec['SD_Critical'])
    DynamicsData_AgeSpec['SD_Deaths'] = np.cumsum(DynamicsData_AgeSpec['SD_Deaths'])


# Critical

ax.fill_between(t[0:t_use], 
                  DynamicsData_AgeSpec['Mean_Deaths'][0:t_use]-DynamicsData_AgeSpec['SD_Deaths'][0:t_use], 
                  DynamicsData_AgeSpec['Mean_Deaths'][0:t_use]+DynamicsData_AgeSpec['SD_Deaths'][0:t_use],
                  color =  col_dead2,
                  alpha=0.25)      
ax.plot(t[0:t_use], DynamicsData_AgeSpec['Mean_Deaths'][0:t_use], color = col_dead2,label = 'April 2021',linewidth=4)
      
 
ax.legend(loc="upper left")
ax.set_xlabel('Time [days]')   
# ax.set_title(TitleLabel[1])
if cumsum==1:
    ax.set_ylim([0, 40])
else:
    ax.set_ylim([0, 40])
ax.set_xlim([1, t_use])

ax.set_xlim([1, t_use])
    
fig.savefig('Figures/CompareDeathDynamics_'  +'Delta_ClusterSize_' + str(int(Cluster_Size))+'.pdf')
            
      
# filename_agespec = 'Combined' + DataMonths[0] + HesitancyTypes[0] +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'  
# DynamicsData_AgeSpec = pd.read_csv('Data/' + filename_agespec,index_col=0)
# plt.fill_between(t[1:t_use], 
#                   DynamicsData_AgeSpec['Mean_Infections'][1:t_use]-DynamicsData_AgeSpec['SD_Infections'][1:t_use][0:t_use], 
#                   DynamicsData_AgeSpec['Mean_Infections'][1:t_use]+DynamicsData_AgeSpec['SD_Infections'][1:t_use][0:t_use],
#                   color =  col_dead,
#                   alpha=0.25)      
# plt.plot(t[1:t_use], DynamicsData_AgeSpec['Mean_Infections'][1:t_use], color = col_dead,label = 'January 2021',linewidth=4)
# filename_agespec = 'Combined' + DataMonths[1] + HesitancyTypes[0] +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'  
# DynamicsData_AgeSpec = pd.read_csv('Data/' + filename_agespec,index_col=0)
# plt.fill_between(t[1:t_use], 
#                   DynamicsData_AgeSpec['Mean_Infections'][1:t_use]-DynamicsData_AgeSpec['SD_Infections'][1:t_use][0:t_use], 
#                   DynamicsData_AgeSpec['Mean_Infections'][1:t_use]+DynamicsData_AgeSpec['SD_Infections'][1:t_use][0:t_use],
#                   color =  col_dead2,
#                   alpha=0.25)      
# plt.plot(t[1:t_use], DynamicsData_AgeSpec['Mean_Infections'][1:t_use], color = col_dead2,label = 'April 2021',linewidth=4)


