# CombinedDataFigures: create figures for the combined data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import colors
import seaborn as sns
# Read if variables missing

Vars = ['Alpha','Delta']
Title = ['B.1.1.7','B.1.617.2']

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

for jk in range(np.size(Vars)):
    
    if jk == 0:
        location = 'Alpha_Variant/ClusterSize_Combined_Data/'
    else:
        location = 'Delta_Variant/ClusterSize_Combined_Data/'
        
    # Choose testing and tracing for 1
    tt = 1 
    ftemplates = ['Hesitancy_','FracVac_']
    prefixes = ['Hes','FracVac'] # File name prefixes
    specifics = ['Age-', 'Non-']
    
    # Choose directory and style here    
    
    # Different file names are labelled by their input data, which are these vectors
    ClusterSize_Vector = np.linspace(1,20,20)   # ClusterSize
    H_Pow_Vector = np.linspace(-1.85,1.85,19)   # Hesitancy values
    N_CS = np.size(ClusterSize_Vector) # Size of data
    N_HP = np.size(H_Pow_Vector)
    
    # Load data
    id_directory = 0 # Change this to choose age-specific (0) or non-specific (1)
    ftemplate = ftemplates[id_directory] 
    prefix = prefixes[id_directory]
    specific = specifics[id_directory]
    file_name = location+str(prefix) + 'Combined_ClusterSize_PostVaccine_Data.csv'
    
    df = pd.read_csv(file_name)
    
    # Reshape to make colour plots
    
    ClusterSize = np.array(df['ClusterSize'])
    ID_Use = ((ClusterSize == 5)+  (ClusterSize == 10)  +   (ClusterSize == 15)  + (ClusterSize == 20))
    ClusterSize = ClusterSize[ID_Use]
    Mean_FracVaccinated = np.array(df['Mean_FracVaccinated'])[ID_Use]
       
    Mean_Infections = np.array(df['Mean_Infections'])[ID_Use]
    Mean_Deaths = np.array(df['Mean_Deaths'])[ID_Use]
    Contain_Day = np.array(df['Contain_Day'])[ID_Use]
    r_eff = np.array(df['reff'])[ID_Use]
    SD_r_eff = np.array(df['SD_reff'])[ID_Use]
    r_eff30 = np.array(df['reff30'])[ID_Use]
    SD_r_eff30 = np.array(df['SD_reff30'])[ID_Use]    
    
    id_directory = 1 # Change this to choose age-specific (0) or non-specific (1)
    ftemplate = ftemplates[id_directory] 
    prefix = prefixes[id_directory]
    specific = specifics[id_directory]
    file_name = location+str(prefix) + 'Combined_ClusterSize_PostVaccine_Data.csv'
    
    df = pd.read_csv(file_name)
    
    # Reshape to make colour plots
    
    Mean_FracVaccinated_FV = np.array(df['Mean_FracVaccinated'])[ID_Use]
    ClusterSize_FV = np.array(df['ClusterSize'])[ID_Use]
    Mean_Infections_FV = np.array(df['Mean_Infections'])[ID_Use]
    Mean_Deaths_FV = np.array(df['Mean_Deaths'])[ID_Use]
    Contain_Day_FV = np.array(df['Contain_Day'])[ID_Use]
    r_eff_FV = np.array(df['reff'])[ID_Use]
    SD_r_eff_FV = np.array(df['SD_reff'])[ID_Use]
    r_eff30_FV = np.array(df['reff30'])[ID_Use]
    SD_r_eff30_FV = np.array(df['SD_reff30'])[ID_Use]   
            
    plt.scatter(Mean_FracVaccinated,  r_eff30, c = ClusterSize, cmap = 'Reds')
    clb=plt.colorbar()
    clb.ax.get_yaxis().set_ticks([5,10,15,20])
    clb.ax.set_yticklabels(['5','10','15','20'])
    plt.scatter(Mean_FracVaccinated_FV, r_eff30_FV, c = ClusterSize,cmap = 'Blues')
    clb=plt.colorbar()
    clb.ax.get_yaxis().set_ticks([5,10,15,20])
    clb.ax.set_yticklabels(['5','10','15','20'])
    clb.set_label('Initial Infections', rotation=270,labelpad = 15)
    plt.scatter(np.nan,np.nan,color = 'r', label = 'Age-Specific')
    plt.scatter(np.nan,np.nan,color = 'b', label = 'Non-Specific')
    plt.hlines(1,xmin = np.min(Mean_FracVaccinated),xmax = np.max(Mean_FracVaccinated),linestyle = '--',colors = 'k',label = 'Herd-Immunity',linewidths = 3)
    plt.legend()
    plt.xlabel('Fraction of Population Vaccinated')
    plt.ylabel('Effective Reproduction Rate $r_{eff}$')
    plt.title(Title[jk])
    plt.ylim([0.25,2.5])
        
    plt.savefig(Vars[jk] + 'r_eff30_Scatter.pdf')
    plt.show()
    
    CS = [5,10,15,20]
    Rcols = sns.color_palette("Reds", np.size(CS)+1)
    Bcols = sns.color_palette("Blues", np.size(CS)+1)
    lnwidth = 3
    
    for ii in range(np.size(CS)):
    
        plt.plot(Mean_FracVaccinated[ClusterSize == CS[ii]],  Mean_Deaths[ClusterSize == CS[ii]], color = Rcols[ii+1],linewidth = lnwidth)
        # clb1=plt.colorbar()
       
        plt.plot(Mean_FracVaccinated_FV[ClusterSize == CS[ii]],  Mean_Deaths_FV[ClusterSize == CS[ii]],color = Bcols[ii+1],linewidth = lnwidth)
        # clb2=plt.colorbar()
    sm = plt.cm.ScalarMappable(cmap="Reds",norm = cm.colors.Normalize(vmin=0,vmax=20))
    clb = plt.colorbar(sm)
    clb.ax.get_yaxis().set_ticks([5,10,15,20])
    clb.ax.set_yticklabels(['5','10','15','20'])
    clb.set_label('Initial Infections', rotation=270,labelpad = 15)
    sm = plt.cm.ScalarMappable(cmap="Blues",norm = cm.colors.Normalize(vmin=0,vmax=20))
    clb = plt.colorbar(sm)
    clb.ax.get_yaxis().set_ticks([5,10,15,20])
    clb.ax.set_yticklabels(['5','10','15','20'])
    # clb.set_label('Initial Infections', rotation=270,labelpad = 15)
    # clb=plt.colorbar()    
    
    plt.scatter(np.nan,np.nan,color = 'r', label = 'Age-Specific')
    plt.scatter(np.nan,np.nan,color = 'b', label = 'Non-Specific')
    plt.legend()
    plt.xlabel('Fraction of Population Vaccinated')
    plt.ylabel('Mean Number of Deaths')
    plt.title(Title[jk])
    plt.savefig(Vars[jk] + 'Num_Deaths_Lines.pdf')
    plt.show()  
    
    ################################## Map#################################
    
    fsize_title = 22
    fsize_legend = 13
    fsize_labels= 14
    fsize_numbers = 11    
    
    plt.rc('font', size=fsize_numbers)          # controls default text sizes
    plt.rc('axes', titlesize=fsize_labels)     # fontsize of the axes title
    plt.rc('axes', labelsize=fsize_labels)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fsize_legend)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fsize_legend)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fsize_legend)    # legend fontsize
    plt.rc('figure', titlesize=fsize_title)  # fontsize of the figure title
    
    # Load data
    id_directory = 0 # Change this to choose age-specific (0) or non-specific (1)
    ftemplate = ftemplates[id_directory] 
    prefix = prefixes[id_directory]
    specific = specifics[id_directory]
    file_name = location+str(prefix) + 'Combined_ClusterSize_PostVaccine_Data.csv'
    
    df = pd.read_csv(file_name)
    
    # Reshape to make colour plots
    ClusterSize_Vector = np.linspace(5,20,16) 
    ClusterSize = df['ClusterSize']
    ID_Use = (ClusterSize >= 5)
    N_CS = np.size(ClusterSize_Vector) # Size of data
    N_HP = np.size(H_Pow_Vector)    
    
    ClusterSize = np.reshape(np.array(df['ClusterSize'][ID_Use]), (N_CS, N_HP))
    Mean_FracVaccinated = np.reshape(np.array(df['Mean_FracVaccinated'][ID_Use]), (N_CS, N_HP))
    r_eff30 = np.reshape(np.array(df['reff30'][ID_Use]), (N_CS, N_HP))
    SD_r_eff30 = np.reshape(np.array(df['SD_reff30'][ID_Use]), (N_CS, N_HP))        
    
    divnorm=colors.TwoSlopeNorm(vmin=np.min(r_eff30), vcenter=1, vmax=np.max(r_eff30))
    plt.pcolor(Mean_FracVaccinated,ClusterSize,r_eff30, cmap=cm.coolwarm,shading='auto',norm=divnorm)
    cbar = plt.colorbar()
    cbar.set_label('Effective Reproduction Rate $r_{eff}$', rotation=270,labelpad=15)
    plt.xlabel('Fraction of Population Vaccinated (' + specific +'Specific)')
    plt.ylabel('Number of Initial Infected Agents')
    plt.title(Title[jk])
    plt.yticks([5,10,15,20])
    # plt.yticklabels(['1','5','10','15','20'])
    plt.savefig(Vars[jk] + str(prefix) + 'r_eff_Heatmap.pdf')
 
    plt.show()