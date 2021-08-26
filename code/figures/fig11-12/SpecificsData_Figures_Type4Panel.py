# Extract and average the data from all the single hesitancy runs of the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
# font = {'family' : 'normal',
#         'size'   : 12}
# matplotlib.rc('font', **font)

# Load the saved data and create figures
DataMonths = ['CurrentVaccinated','NowHes','NoYouthHes']
TitleLabel = ['Current Vaccinated Population ','Current Hesitant Population','No Hesitant Population <34 ']
# ONLY USE TWO TO COMPARE
# DataMonths = ['August','January']
DataTypes = ['AgeSpec','Dynamics']
HesitancyTypes = ['',''] # Chose age-specific or non-specific hesitancy
HTypeLabel = ['','']
Array_Nums = np.linspace(0,99,100) # Number for repititions
Cluster_Sizes = np.linspace(1,20,20) # Imported infections


pop_size = 5.1e6

AgeLabels = ['0-17','18-24','25-34','35-44','45-54','55-64','65+']

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
                 
# # TODO: 4 loops here, month, DType, HType, Cluster_Size
for id_Months in range(np.size(DataMonths)):   
    Month = DataMonths[id_Months] # Chose a month to combine
    DType = DataTypes[0] # Chose datatype to combine
    HType = HesitancyTypes[0] # Chose hesitancy type to combine

    Cluster_Size = int(20) # Pick values to use
    if DType == 'AgeSpec':           
            
        filename_agespec = 'Combined' + Month + HType +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'
        
        AgeSpecificData = pd.read_csv('Data/' + filename_agespec,index_col=0)
        scale = 1
        # PlotAgeSpecificData(AgeSpecificData) 
        # # Data of Infected, Infected and Vacc, Dead, Dead and Vacc
        data1_18_24   = np.multiply([AgeSpecificData['18-24']['Mean_Infected']              
                          ],scale)
        
        data1_25_34  = np.multiply([AgeSpecificData['25-34']['Mean_Infected']                
                          ],scale)
        
        data1_35_44  = np.multiply([AgeSpecificData['35-44']['Mean_Infected']                
                          ],scale)
        
        data1_45_54  = np.multiply([AgeSpecificData['45-54']['Mean_Infected']                
                          ],scale)
        
        data1_55_64  = np.multiply([AgeSpecificData['55-64']['Mean_Infected']              
                          ],scale)
        
        data1_65_plus  = np.multiply([AgeSpecificData['65+']['Mean_Infected']            
                          ],scale)
        
        Vdata1_18_24   = np.multiply([AgeSpecificData['18-24']['Mean_Infected_Vaccinated'],              
                          ],scale)
        
        Vdata1_25_34  = np.multiply([AgeSpecificData['25-34']['Mean_Infected_Vaccinated'],                
                          ],scale)
        
        Vdata1_35_44  = np.multiply([AgeSpecificData['35-44']['Mean_Infected_Vaccinated'],                
                          ],scale)
        
        Vdata1_45_54  = np.multiply([AgeSpecificData['45-54']['Mean_Infected_Vaccinated'],                
                          ],scale)
        
        Vdata1_55_64  = np.multiply([AgeSpecificData['55-64']['Mean_Infected_Vaccinated'],              
                          ],scale)
        
        Vdata1_65_plus  = np.multiply([AgeSpecificData['65+']['Mean_Infected_Vaccinated']              
                          ],scale)
        
        # # Data of Diagnosed, Diagnosed and Vacc, Critical, Critical and Vacc
        data2_18_24   = np.multiply([AgeSpecificData['18-24']['Mean_Critical'],                            
                          AgeSpecificData['18-24']['Mean_Dead'], 
                          ],scale)
        
        data2_25_34  = np.multiply([AgeSpecificData['25-34']['Mean_Critical'],                             
                          AgeSpecificData['25-34']['Mean_Dead'], 
                          ],scale)
        
        data2_35_44  = np.multiply([AgeSpecificData['35-44']['Mean_Critical'],                              
                          AgeSpecificData['35-44']['Mean_Dead'],
                          ],scale)
        
        data2_45_54  = np.multiply([AgeSpecificData['45-54']['Mean_Critical'], 
                          AgeSpecificData['45-54']['Mean_Dead'],
                          ],scale)
        data2_55_64  = np.multiply([AgeSpecificData['55-64']['Mean_Critical'],                                                         
                          AgeSpecificData['55-64']['Mean_Dead'], 
                          ],scale)
        
        data2_65_plus   = np.multiply([AgeSpecificData['65+']['Mean_Critical'],                                                                
                          AgeSpecificData['65+']['Mean_Dead'],
                          ],scale)        
        
        Vdata2_18_24   = np.multiply([AgeSpecificData['18-24']['Mean_Critical_Vaccinated'],
                          AgeSpecificData['18-24']['Mean_Dead_Vaccinated'],   
                          ],scale)
        
        Vdata2_25_34  = np.multiply([AgeSpecificData['25-34']['Mean_Critical_Vaccinated'],
                          AgeSpecificData['25-34']['Mean_Dead_Vaccinated'], 
                          ],scale)
        
        Vdata2_35_44  = np.multiply([AgeSpecificData['35-44']['Mean_Critical_Vaccinated'],
                          AgeSpecificData['35-44']['Mean_Dead_Vaccinated'], 
                          ],scale)
        
        Vdata2_45_54  = np.multiply([AgeSpecificData['45-54']['Mean_Critical_Vaccinated'],
                          AgeSpecificData['45-54']['Mean_Dead_Vaccinated'], 
                          ],scale)
        Vdata2_55_64  = np.multiply([AgeSpecificData['55-64']['Mean_Critical_Vaccinated'],
                          AgeSpecificData['55-64']['Mean_Dead_Vaccinated'],  
                          ],scale)
        
        Vdata2_65_plus   = np.multiply([AgeSpecificData['65+']['Mean_Critical_Vaccinated'], 
                          AgeSpecificData['65+']['Mean_Dead_Vaccinated'], 
                          ],scale)       
                    
        data_labels1 = ["Infected \n and Vaccinated"]
        data_labels2 = ["Critical \n and Vaccinated", "Dead \n and Vaccinated"]
        age_labels = ['18-24','25-34','35-44','45-54','55-64','65+'] 
        
        # Sort out colours here
        alph1 = 0.5
        alph2 = 0.9
        col_grad = np.linspace(0.7,1,np.size(age_labels))
        col_infected = [150/255,250/255,150/255]
        col_critical = [220/255,20/255,60/255]
        col_dead = [0.5,0.5,0.5]
                  
                
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,9))
        xticks1 = 0
   
        width = 0.15  # the width of the bars
        width_bars = 0.14
        w_adj = (np.sqrt(5)-1)/2
        x_pos = [xticks1-2.5*width, xticks1-1.5*width,xticks1-1/2*width,
                  xticks1+1/2*width, xticks1+1.5*width,xticks1+2.5*width]
        
        d1 = ax[0].bar(x_pos[0], data1_18_24   , width_bars, color=np.multiply(col_grad[0],col_infected), alpha=alph1)
        d2 = ax[0].bar(x_pos[1],  data1_25_34 ,   width_bars, color=np.multiply(col_grad[1],col_infected),alpha=alph1)
        d3 = ax[0].bar(x_pos[2],  data1_35_44  , width_bars, color=np.multiply(col_grad[2],col_infected), alpha=alph1)
        d4 = ax[0].bar(x_pos[3],  data1_45_54   , width_bars, color=np.multiply(col_grad[3],col_infected),  alpha=alph1)
        d5 = ax[0].bar(x_pos[4],  data1_55_64 ,   width_bars, color=np.multiply(col_grad[4],col_infected), alpha=alph1)
        d6 = ax[0].bar(x_pos[5],  data1_65_plus  , width_bars, color=np.multiply(col_grad[5],col_infected), alpha=alph1,label='Infected')
                 
        d1 = ax[0].bar(x_pos[0], Vdata1_18_24   , width_bars*w_adj, color=np.multiply(col_grad[0],col_infected),alpha=alph2)
        d2 = ax[0].bar(x_pos[1],  Vdata1_25_34 ,   width_bars*w_adj,color=np.multiply(col_grad[1],col_infected),alpha=alph2)
        d3 = ax[0].bar(x_pos[2],  Vdata1_35_44  , width_bars*w_adj, color=np.multiply(col_grad[2],col_infected), alpha=alph2)
        d4 = ax[0].bar(x_pos[3],  Vdata1_45_54   , width_bars*w_adj, color=np.multiply(col_grad[3],col_infected),alpha=alph2)
        d5 = ax[0].bar(x_pos[4],  Vdata1_55_64 ,   width_bars*w_adj, color=np.multiply(col_grad[4],col_infected),alpha=alph2)
        d6 = ax[0].bar(x_pos[5],  Vdata1_65_plus  , width_bars*w_adj, color=np.multiply(col_grad[5],col_infected),alpha=alph2,label = 'Infected \n and Vaccinated') 
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[0].set_ylabel('Number of People')
        ax[0].set_xlabel('Age Category')
        ax[0].set_title(TitleLabel[id_Months],size = fsize_title)
        ax[0].set_xticks(x_pos)
        ax[0].set_xticklabels(age_labels)
       
        ax[0].legend(loc="upper left", ncol = 2)
        
        # ax[0].bar_label(d1, labels = np.round(data1_18_24,2), padding=3)
        # ax[0].bar_label(d2, labels = np.round(data1_25_34,2), padding=3)
        # ax[0].bar_label(d3, labels = np.round(data1_35_44,2), padding=3)
        # ax[0].bar_label(d4, labels = np.round(data1_45_54,2), padding=3)
        # ax[0].bar_label(d5, labels = np.round(data1_55_64,2), padding=3)
        # ax[0].bar_label(d6, labels = np.round(data1_65_plus,2), padding=3)
        xticks2 = 0 # Critical
        d11 = ax[1].bar(xticks2-2.5*width,  data2_18_24[0], width_bars, color=np.multiply(col_grad[0],col_critical),  alpha=alph1)
        d12 = ax[1].bar(xticks2-1.5*width,  data2_25_34[0],   width_bars, color=np.multiply(col_grad[1],col_critical), alpha=alph1)
        d13 = ax[1].bar(xticks2-1/2*width,  data2_35_44[0], width_bars, color=np.multiply(col_grad[2],col_critical), alpha=alph1)
        d14 = ax[1].bar(xticks2+1/2*width,  data2_45_54[0], width_bars, color=np.multiply(col_grad[3],col_critical),  alpha=alph1)
        d15 = ax[1].bar(xticks2+1.5*width,  data2_55_64[0],   width_bars, color=np.multiply(col_grad[4],col_critical),alpha=alph1)
        d16 = ax[1].bar(xticks2+2.5*width,  data2_65_plus[0], width_bars, color=np.multiply(col_grad[5],col_critical), alpha=alph1, label = 'Critical')
                   
        ax[1].bar(xticks2-2.5*width,  Vdata2_18_24[0], width_bars*w_adj, color=np.multiply(col_grad[0],col_critical),alpha=alph2)
        ax[1].bar(xticks2-1.5*width,  Vdata2_25_34[0],   width_bars*w_adj,color=np.multiply(col_grad[1],col_critical),alpha=alph2)
        ax[1].bar(xticks2-1/2*width,  Vdata2_35_44[0], width_bars*w_adj, color=np.multiply(col_grad[2],col_critical), alpha=alph2)
        ax[1].bar(xticks2+1/2*width,  Vdata2_45_54[0], width_bars*w_adj, color=np.multiply(col_grad[3],col_critical),alpha=alph2)
        ax[1].bar(xticks2+1.5*width,  Vdata2_55_64[0],   width_bars*w_adj, color=np.multiply(col_grad[4],col_critical),alpha=alph2)
        ax[1].bar(xticks2+2.5*width,  Vdata2_65_plus[0], width_bars*w_adj, color=np.multiply(col_grad[5],col_critical),alpha=alph2,label = 'Critical \n and Vaccinated') 
               
        xticks2 = 1 # Deaths
        d21 = ax[1].bar(xticks2-2.5*width,  data2_18_24[1], width_bars, color=np.multiply(col_grad[0],col_dead),  alpha=alph1)
        d22 = ax[1].bar(xticks2-1.5*width,  data2_25_34[1],   width_bars, color=np.multiply(col_grad[1],col_dead), alpha=alph1)
        d23 = ax[1].bar(xticks2-1/2*width,  data2_35_44[1], width_bars, color=np.multiply(col_grad[2],col_dead), alpha=alph1)
        d24 = ax[1].bar(xticks2+1/2*width,  data2_45_54[1], width_bars, color=np.multiply(col_grad[3],col_dead),  alpha=alph1)
        d25 = ax[1].bar(xticks2+1.5*width,  data2_55_64[1],   width_bars, color=np.multiply(col_grad[4],col_dead), alpha=alph1)
        d26 = ax[1].bar(xticks2+2.5*width,  data2_65_plus[1], width_bars, color=np.multiply(col_grad[5],col_dead), alpha=alph1, label = 'Dead')

        ax[1].bar(xticks2-2.5*width,  Vdata2_18_24[1], width_bars*w_adj, color=np.multiply(col_grad[0],col_dead),alpha=alph2)
        ax[1].bar(xticks2-1.5*width,  Vdata2_25_34[1],   width_bars*w_adj,color=np.multiply(col_grad[1],col_dead),alpha=alph2)
        ax[1].bar(xticks2-1/2*width,  Vdata2_35_44[1], width_bars*w_adj, color=np.multiply(col_grad[2],col_dead), alpha=alph2)
        ax[1].bar(xticks2+1/2*width,  Vdata2_45_54[1], width_bars*w_adj, color=np.multiply(col_grad[3],col_dead),alpha=alph2)
        ax[1].bar(xticks2+1.5*width,  Vdata2_55_64[1],   width_bars*w_adj, color=np.multiply(col_grad[4],col_dead),alpha=alph2)
        ax[1].bar(xticks2+2.5*width,  Vdata2_65_plus[1], width_bars*w_adj, color=np.multiply(col_grad[5],col_dead),alpha=alph2, label = 'Dead \n and Vaccinated') 
      
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[1].set_ylabel('Number of People')        
        ax[1].set_xlabel('Age Category')
        x_pos1 = [x+1 for x in x_pos]
        ticks = x_pos+x_pos1
        ax[1].set_xticks(ticks)
        ax[1].set_xticklabels(age_labels+age_labels)          
        ax[1].legend(loc="upper left", ncol = 4)
        # ax[1].legend()
        
        ax[1].bar_label(d11)#, labels = data2_18_24.astype(int), padding=3)
        ax[1].bar_label(d12)#, labels = data2_25_34.astype(int), padding=3)
        ax[1].bar_label(d13)#, labels = data2_35_44.astype(int), padding=3)
        ax[1].bar_label(d14)#, labels = data2_45_54.astype(int), padding=3)
        ax[1].bar_label(d15)#, labels = data2_55_64.astype(int), padding=3)
        ax[1].bar_label(d16)#, labels = data2_65_plus.astype(int), padding=3)
        
        ax[1].bar_label(d21)#, labels = data2_18_24.astype(int), padding=3)
        ax[1].bar_label(d22)#, labels = data2_25_34.astype(int), padding=3)
        ax[1].bar_label(d23)#, labels = data2_35_44.astype(int)[1], padding=3)
        ax[1].bar_label(d24)#, labels = data2_45_54.astype(int)[1], padding=3)
        ax[1].bar_label(d25)#, labels = data2_55_64.astype(int)[1], padding=3)
        ax[1].bar_label(d26)#, labels = data2_65_plus.astype(int)[1], padding=3)
        ax[1].set_yscale('log')
      
        
        if id_Months == 0:
            ax[0].set_ylim([0, 600000])
            ax[1].set_ylim([1, 5e6])
        else:
            ax[0].set_ylim([0, 30000])
            ax[1].set_ylim([1, 5e5])
        
        
        fig.savefig('Figures/SpecificsCombined' + Month + HType +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.pdf')


############################# Dynamics #####################################

# Load the saved data and create figures
DataMonths = ['CurrentVaccinated','NowHes','NoYouthHes']
TitleLabel = ['Current Vaccinated Population ','Current Hesitant Population','No Hesitant Population <34 ']
HesitancyTypes = ['','NonSpec'] # Chose age-specific or non-specific hesitancy
HTypeLabel = ['Age-Specific','Non-Specific ']
Array_Nums = np.linspace(0,99,100) # Number for repititions
Cluster_Sizes = np.linspace(1,20,20) # Imported infections
pop_size = 5.1e6

DType = DataTypes[1] # Chose datatype to combine

Cluster_Size = int(20) # Pick values to use    

cumsum = 0
 # set 0 for raw, 1 for cum_sum


col_crit = [255/255,165/255,0]
col_dead = [0.5,0.5,0.5]
col_crit = np.multiply([255/255,165/255,0],1)
col_crit2 =  np.multiply([255/255,135/255,0],1)
col_dead = np.multiply([0.5,0.5,0.5],1)
col_dead2 = np.multiply([0.5,0.5,0.5],0.6)  

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,8.5))
# Critical Condition: Age- and Non-Spec
for ii in range(3):
    if ii == 0:
        jj = 0
    else:
        jj = 1
        
    # Load two files: Age-specific and non-specific
    filename_agespec = 'Combined' + DataMonths[ii] + HesitancyTypes[0] +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'  
    DynamicsData_AgeSpec = pd.read_csv('Data/' + filename_agespec,index_col=0)
 
    # print(np.cumsum(DynamicsData_AgeSpec['Mean_Infections'])  )
    # print(np.cumsum(DynamicsData_AgeSpec['Mean_Critical'])  )
    # print(np.cumsum(DynamicsData_AgeSpec['Mean_Deaths'])  )

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
    # Critical
    if ii == 2:
        col_crit = col_crit2
   
    ax[jj].fill_between(t[0:t_use], 
                      DynamicsData_AgeSpec['Mean_Critical'][0:t_use]-DynamicsData_AgeSpec['SD_Critical'][0:t_use], 
                      DynamicsData_AgeSpec['Mean_Critical'][0:t_use]+DynamicsData_AgeSpec['SD_Critical'][0:t_use],
                      color =  col_crit,
                      alpha=0.25)      
    ax[jj].plot(t[0:t_use], DynamicsData_AgeSpec['Mean_Critical'][0:t_use], color = col_crit,label = TitleLabel[ii],linewidth = 4)
    if ii == 0 | ii == 2 :
        ax[jj].axhline(y = 351, color = 'r', linestyle = '--',label='ICU Capacity QLD')
    else:
        ax[jj].axhline(y = 351, color = 'r', linestyle = '--')  
    ax[jj].legend(loc="upper left") 
    
    ax[jj].set_xlabel('Time [days]')       
    
if cumsum==1:
    ax[1].set_ylabel('Cumulative Number of Critical Conditions',labelpad=10)
    ax[0].set_ylim([0, 2000])
    ax[1].set_ylim([0, 800])
    
else:
    ax[1].set_ylabel('Number of New Critical Conditions',labelpad=10)
    ax[0].set_ylim([0, 8000])
    ax[1].set_ylim([0, 800])

ax[0].set_xlim([1, 90])
ax[1].set_xlim([1, 90])
fig.savefig('Figures/SpecificsCriticalDynamics_'  +'Delta_ClusterSize_' + str(int(Cluster_Size))+'.pdf')

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,8.5))
# Critical Condition: Age- and Non-Spec
for ii in range(3):
    if ii == 0:
        jj = 0
    else:
        jj = 1
        
    # Load two files: Age-specific and non-specific
    filename_agespec = 'Combined' + DataMonths[ii] + HesitancyTypes[0] +'Delta_' + DType + '_ClusterSize_' + str(int(Cluster_Size))+'.csv'  
    DynamicsData_AgeSpec = pd.read_csv('Data/' + filename_agespec,index_col=0)
 
    # print(np.cumsum(DynamicsData_AgeSpec['Mean_Infections'])  )
    # print(np.cumsum(DynamicsData_AgeSpec['Mean_Critical'])  )
    # print(np.cumsum(DynamicsData_AgeSpec['Mean_Deaths'])  )

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
    # Critical
    if ii == 2:
        col_dead = col_dead2
   
    ax[jj].fill_between(t[0:t_use], 
                      DynamicsData_AgeSpec['Mean_Deaths'][0:t_use]-DynamicsData_AgeSpec['SD_Deaths'][0:t_use], 
                      DynamicsData_AgeSpec['Mean_Deaths'][0:t_use]+DynamicsData_AgeSpec['SD_Deaths'][0:t_use],
                      color =  col_dead,
                      alpha=0.25)      
    ax[jj].plot(t[0:t_use], DynamicsData_AgeSpec['Mean_Deaths'][0:t_use], color = col_dead,label = TitleLabel[ii],linewidth = 4)

     
    ax[jj].set_xlabel('Time [days]')       
    ax[jj].legend(loc="upper left") 
if cumsum==1:
    ax[1].set_ylabel('Cumulative Number of Deaths',labelpad=10)
    ax[0].set_ylim([0, 300])
    ax[1].set_ylim([0, 50])
    
else:
    ax[1].set_ylabel('Number of New Deaths',labelpad=10)
    ax[0].set_ylim([0, 300])
    ax[1].set_ylim([0, 50])

ax[0].set_xlim([1, 90])
ax[1].set_xlim([1, 90])
fig.savefig('Figures/SpecificsDeathDynamics_'  +'Delta_ClusterSize_' + str(int(Cluster_Size))+'.pdf')