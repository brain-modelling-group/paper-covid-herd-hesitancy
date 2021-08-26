import covasim as cv
import pandas as pd
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt
# For local testruns
AgeGroups = [17,24,34,44,54,64,200]    
# Hesitancy = (1,1,1,1,1,1)
# Hesitancy = (0,0,0,0,0,0)
H_Pow = 0
Cluster_Size = 20
tt = 1
Variant = ['Alpha', 'Delta']
kk = 1 # 0 = alpha, 1 = delta
pop_size = 200000
N_Sims = 24
N_sims = N_Sims
CPU_NUM = 24
# Hesitancy values from Edwards et al., 2021
Hesitancy_Aug20 = (0.092, 0.171,0.117,0.163, 0.104,0.08)
Hesitancy_Jan21 = (0.145, 0.298,0.247,0.270, 0.121,0.094)



Hesitancy = Hesitancy_Jan21
Hesitancy = (0,0,0,0,0,0)
# data,msim = qld_clustersize_hesitancy_postvaccine_function(Hesitancy,AgeGroups,N_sims,H_Pow,Cluster_Size,tt,CPU_NUM)
    ##using Paula's code for the rollout
    
def testing_and_tracing_interventions(sim,start_simulation_date, end_simulation_date, label, num_tests=7501):
  
  ntpts = sim.day(end_simulation_date)-sim.day(start_simulation_date)

  if label == 'tt': 
    # Testing
    sim.pars['interventions'].append(cv.test_num(daily_tests=[num_tests]*ntpts, 
                                                 start_day=sim.day(start_simulation_date), 
                                                 end_day=sim.day(end_simulation_date)+5, 
                                                 symp_test=100.0, test_delay=0))

    # Tracing
    trace_probs = {'h': 1.00, 's': 0.95, 
                   'w': 0.80, 'c': 0.05, 
                   'church': 0.50, 
                   'pSport': 0.80, 
                   'cSport': 0.50,
                   'entertainment': 0.10, 
                   'cafe_restaurant': 0.70, 
                   'pub_bar': 0.50, 
                   'transport': 0.50, 
                   'public_parks': 0.00, 
                   'large_events': 0.05, 
                   'social': 0.90}

    trace_time = {'h': 1, 's': 2, 
                  'w': 2, 'c': 14, 
                  'church': 5, 
                  'pSport': 3, 
                  'cSport': 3, 
                  'entertainment': 7,
                  'cafe_restaurant': 7, 
                  'pub_bar': 7, 
                  'transport': 14, 
                  'public_parks': 21,  
                  'large_events': 21,
                  'social': 3}

    sim.pars['interventions'].append(cv.contact_tracing(trace_probs=trace_probs,  
                                                        trace_time=trace_time, 
                                                        start_day=0, do_plot=False))
  
    return sim
def vaccine_phase_subtargets_instantaneous(sim):
    
    sim_pop_size = len(sim.people)

    FullVacc_age_group = cv.true((sim.people.age >= 18))
    FullVacc_subpop_size = int(np.size(FullVacc_age_group))
    # Make an array of size pop_size
    FullVacc_subpop = np.zeros((sim_pop_size, ), dtype=bool)
    # Select phase1a_pop_size agents within the age-bracket specified  
    FullVacc_agents = np.random.choice(sim.people.uid[FullVacc_age_group], FullVacc_subpop_size, replace=False)
    FullVacc_subpop[FullVacc_agents] = True
    vaccine_rollout_phases = {'Full_Vaccination': {'start_date': sim.day('2020-12-10'),
                                          'subpop_inds': FullVacc_subpop,
                                          'prob': 1.0,
                                          'vaccine_brand': 'pfizer'},
                       }   
    return vaccine_rollout_phases


def get_rollout_phase_data(sim, vaccin_rollout_dict, phase_label):
    '''
    This function will select the data for the vaccination phase 
    specified in `phase_label`
    '''
    sim_pop_size = len(sim.people)
    # Indices with people that fall within this age group 
    vals = np.zeros((sim_pop_size )) 
    vals[vaccin_rollout_dict[phase_label]['subpop_inds']] =  vaccin_rollout_dict[phase_label]['prob']# probability for people in this subpopulation
    subtarget_dict = dict(inds = sim.people.uid,
                          vals = vals)
    phase_start_date = vaccin_rollout_dict[phase_label]['start_date']
    vaccine_brand = vaccin_rollout_dict[phase_label]['vaccine_brand']
    
    return vaccine_brand, phase_start_date, subtarget_dict   

# Get the unique ids for the hesitant fraction of each age group
def get_hesitancy_uids(sim,hes,ages):

    hesitancy_uids = np.array(0)
    hesitancy_uids = sim.people.uid[cv.true(sim.people.age <= 18)]
    
    for ii in range(len(hes)):
        # Find the index for the people in this age group
        this_age_group = cv.true(((sim.people.age > ages[ii] )*(sim.people.age <= ages[ii+1])))
        this_groupsize = int(np.size(this_age_group)*hes[ii]) # Find the number of hesitant people 
        # Get the ID of the hesitant people      
  
        
        this_agent_uids = np.random.choice(sim.people.uid[this_age_group], this_groupsize, replace=False)
        hesitancy_uids = np.append(hesitancy_uids,this_agent_uids) # Add the IDs to the output vector 
    
    hesitancy_uids = hesitancy_uids[1:-1] # Remove the first element - it's a placeholder
 
    return hesitancy_uids


def get_individual_traces(key, sims, convolve=False, num_days=3):
# Returns an array of shape tpts x nruns 
# Values correspond to covasim output defined in 'key' 

    ys = []
    for this_sim in sims:
        ys.append(this_sim.results[key].values)
    yarr = np.array(ys)

    if convolve:
        for idx in range(yarr.shape[0]):
            yarr[idx, :] = np.convolve(yarr[idx, :], np.ones((num_days, ))/num_days, mode='same')
            yarr = np.array(yarr).T

    return yarr       

    # Add a small script for averaging a window in python
#----------------------------------------------------------------------------
def smooth_curve(data,width):
    N = np.size(data)
    smooth_data = np.zeros(np.shape(data))    
    for ii in range(N):
        if (ii >= width and ii < N-width):            
            smooth_data[ii] = np.mean(data[ii-width:ii+width])            
        elif ii < width: # Force edges to be flat    
            smooth_data[ii] = np.mean(data[1:width])            
        else:
            smooth_data[ii] = np.mean(data[-1-width:-1])
            
    return smooth_data
#----------------------------------------------------------------------------

#################################################################################
  



start_date = '2020-12-10'
end_date = '2021-06-30' 
iq_factor = 10.0 # high compliance of quarantine and isolation

# Importing infections   
# When infections are imported
import_dates = ['2021-01-01'] # Day 1

# Define list with dates and number of imports and virus variant
these_dates = [cv.day(day, start_day=start_date) for day in import_dates]
if Variant[kk] == 'Alpha':
    imported_infections = cv.variant('uk', days=these_dates, n_imports=Cluster_Size, rescale=False) 
elif Variant[kk] == 'Delta':
    b16172 = dict(
                rel_beta        = 2.2, # Estimated to be 1.25-1.6-fold more transmissible than B117: https://www.researchsquare.com/article/rs-637724/v1
                rel_symp_prob   = 1.0,
                rel_severe_prob = 3.2, # 2x more transmissible than alpha from https://mobile.twitter.com/dgurdasani1/status/1403293582279294983
                rel_crit_prob   = 1.0,
                rel_death_prob  = 1.0,
            )
    imported_infections = cv.variant(variant = b16172, days=these_dates, n_imports=Cluster_Size, rescale=False)
    

all_layers = ['h', 's', 'w', 'c', 
              'church', 
              'pSport', 
              'cSport', 
              'entertainment', 
              'cafe_restaurant', 
              'pub_bar', 
              'transport', 
              'public_parks', 
              'large_events', 
              'social']

dynam_layers = ['c', 'entertainment', 'cafe_restaurant', 'pub_bar', 'transport', 'public_parks', 'large_events']

qld_pop = {
    '0-4':  314602,
    '5-9':  339247,
  '10-14': 345205,
  '15-19': 319014,
  '20-24': 338824,
  '25-29': 370468,
  '30-34': 362541,
  '35-39': 354219,
  '40-44': 325208,
  '45-49': 348003,
  '50-54': 321168,
  '55-59': 317489,
  '60-64': 288317,
  '65-69': 254114,
  '70-74': 226033,
  '75-79': 156776,
  '80-84': 100692,
  '85-89': 57073,
  '90-94': 28134,
  '95-99': 7931,
  '100-104':1128
  }


# Update with qld population stats
cv.data.country_age_data.data['Australia'] = qld_pop

# Simulation and population parameters
pars = {
        'pop_size':  pop_size,
        'pop_scale': 5100000/pop_size,    # Population scales to 5.1M ppl in QLD
        'pop_infected': Cluster_Size,    # Original population infedcted
        'rescale': False,      # Population dynamics rescaling
        'dynam_layer': pd.Series([0.0,     0.0,    0.0,    1.0,   0.0,    0.0,    0.0,      1.0,    1.0,    1.0,    1.0,     1.0,        1.0,    0.0], index=all_layers).to_dict(),
                                  # H        S       W       C   church   psport  csport    ent     cafe    pub     trans    park        event    soc
        'contacts':    pd.Series([4.0,    21.0,    5.0,    1.0,   20.0,   40.0,    30.0,    25.0,   19.00,  30.00,   25.00,   10.00,     50.00,   6.0], index=all_layers).to_dict(),
        'beta_layer':  pd.Series([1.0,     0.3,    0.2,    0.1,    0.04,   0.2,     0.1,     0.01,   0.04,   0.06,    0.16,    0.03,      0.01,   0.3], index=all_layers).to_dict(),
        'iso_factor':  pd.Series([1.0, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor], index=all_layers).to_dict(),
        'quar_factor': pd.Series([1.0, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor], index=all_layers).to_dict(),
        'variants': imported_infections,
        'location': 'Australia',
        'rand_seed': 21,
        'start_day': start_date,
        'end_day':   end_date,
        'use_waning': True,
        'verbose': 0}

# Define if we want testing and tracing
if tt == 1:
    label = 'tt'
else:
    label == 'no-tt'    
 

#--------------------------------------------------------------------------------------------------------------------------------------------#

# Hesitancy: Run simulation with hesitancy
str_label = 'H_Pow = ' + str(H_Pow) 
sim_hesitance = cv.Sim(pars, label= str_label)
sim_hesitance.initialize()  # need to intialize to generate People()   

hesitancy_uids = get_hesitancy_uids(sim_hesitance,Hesitancy,AgeGroups)

print('H_Pow = ', str(H_Pow), ', Number of Hesitant Agents = ',  len(hesitancy_uids))

# Define if we want testing and tracing

if tt == 1:    
    sim_hesitance = testing_and_tracing_interventions(sim_hesitance, start_date, end_date, label='tt')        
else:
    sim_hesitance = testing_and_tracing_interventions(sim_hesitance, start_date, end_date, label='no-tt')
    
sim_hesitance.initialize()  # need to intialize to generate People()
# Rollout with hesitancy
vaccine_phases = ['Full_Vaccination']
vaccine_rollout_phase_dict = vaccine_phase_subtargets_instantaneous(sim_hesitance)

for key in vaccine_phases:
    vaccine_brand, phase_start, phase_target = get_rollout_phase_data(sim_hesitance, vaccine_rollout_phase_dict, key)
    Before = np.sum(phase_target['vals'])
    phase_target['vals'][hesitancy_uids] = 0
    After = np.sum(phase_target['vals'])
    print('Hesitancy caused a ', (Before-After)/(Before)*100, '% drop in vaccinations for' , key)    

    sim_hesitance.pars['interventions'].append(cv.vaccinate(vaccine=vaccine_brand, prob = 1.0, days=[phase_start], subtarget=phase_target))
sim_hesitance.initialize()  #need to initialize again otherwise covasim complains that the intervention has not been initialized
      
# Run the scenario N_sims times    
  
msim = cv.MultiSim(sim_hesitance,par_args={'ncpus': CPU_NUM}, verbose = 0)
msim.run(n_runs=N_sims, reseed=True) 

sim_name = 'Janurary_Hesitancy_Sim.obj'
msim.save(sim_name)

#--------------------Analysis-------------------------#
# Find the number of people in each age group who were vaccinated, infected or
# dead. Furthermore, find the people who were infected/died AND were vaccinated 

# Need it to work for multisim, or at least cycle through each simulation and 
# save a matrix (to be averaged column-wise).

Infected_per_age = np.zeros((np.size(msim.sims),np.size(AgeGroups)))
Dead_per_age = np.zeros((np.size(msim.sims),np.size(AgeGroups)))
Vaccinated_per_age = np.zeros((np.size(msim.sims),np.size(AgeGroups)))
VaccInfected_per_age = np.zeros((np.size(msim.sims),np.size(AgeGroups)))
VaccDead_per_age = np.zeros((np.size(msim.sims),np.size(AgeGroups)))

for ii in range(np.size(msim.sims)):      

    sim = msim.sims[ii]  
    AgeBounds =  [0,17,24,34,44,54,64,200]    
    AgeLabels = ['0-17','18-24','25-34','35-44','45-54','55-64','65+']

    for jj in range(np.size(AgeBounds)-1):
      
        # Find the index for the people in this age group
        this_age_group = cv.true(((sim.people.age > AgeBounds[jj] )*(sim.people.age <= AgeBounds[jj+1])))  
        # Get the ID of this age group         
        this_age_uids = sim.people.uid[this_age_group]
        # Find which of the following IDs have the following attributes:
        Infected_per_age[ii,jj] = np.sum(msim.people.date_infectious[this_age_group]>0)/pop_size
        Dead_per_age[ii,jj] = np.sum(msim.people.date_dead[this_age_group]>0)/pop_size
        Vaccinated_per_age[ii,jj] = np.sum(msim.people.date_vaccinated[this_age_group]>0)/pop_size
        VaccInfected_per_age[ii,jj] = np.sum((msim.people.date_vaccinated[this_age_group]>0)*(msim.people.date_infectious[this_age_group]>0))/pop_size
        VaccDead_per_age[ii,jj] = np.sum((msim.people.date_vaccinated[this_age_group]>0)*(msim.people.date_dead[this_age_group]>0))/pop_size


# Finding mean and std for each
AgeSpecificData = pd.DataFrame(data = [np.mean(Vaccinated_per_age,0),
                          np.mean(Infected_per_age,0),
                          np.mean(VaccInfected_per_age,0),
                          np.mean(Dead_per_age,0),
                          np.mean(VaccDead_per_age,0),
                          np.std(Vaccinated_per_age,0),
                          np.std(Infected_per_age,0),
                          np.std(VaccInfected_per_age,0),
                          np.std(Dead_per_age,0),
                          np.std(VaccDead_per_age,0)],
                  index = ['Mean_Vaccinated',
                           'Mean_Infected',
                           'Mean_Infected_Vaccinated',
                           'Mean_Dead',
                           'Mean_Dead_Vaccinated',
                           'Std_Vaccinated',
                           'Std_Infected',
                           'Std_Infected_Vaccinated',
                           'Std_Dead',
                           'Std_Dead_Vaccinated'],
                  columns = AgeLabels)


AgeSpecificData.to_csv('Janurary_Hesitancy_Results.csv')


# Plotting these results to stacked bar graphs

labels = ['Vaccinated','Infected','Infected \n and Vaccinated','Dead','Dead \n and Vaccinated']

width = 0.35       # the width of the bars

fig, ax = plt.subplots()
this_level = np.zeros(np.size(AgeSpecificData[AgeLabels[ii]][0:5]))
for ii in range(len(AgeLabels)):   
    if ii == 0:
        ax.bar(labels, np.array(AgeSpecificData[AgeLabels[ii]][0:5]), width, yerr= np.array(AgeSpecificData[AgeLabels[ii]][5:]), label= AgeLabels[ii])
    else:
        this_level = this_level + np.array(AgeSpecificData[AgeLabels[ii-1]][0:5])
        ax.bar(labels, np.array(AgeSpecificData[AgeLabels[ii]][0:5]), width, yerr= np.array(AgeSpecificData[AgeLabels[ii]][5:]), label= AgeLabels[ii], bottom = this_level )

ax.set_ylabel('Percentage of QLD Population')
ax.legend()

plt.show()
plt.savefig('Janurary_Hesitancy_AgeSpec_Results.pdf')

fig_name = 'Janurary_Hesitancy_Trajectory.pdf'
msim.reduce()
to_plot = sc.objdict({
    'Total infections': ['cum_infections'],
    'New infections per day': ['new_infections'],
    'Total deaths': ['cum_deaths']
})
msim_fig = msim.plot(to_plot=to_plot, do_save=False, do_show=False, legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=35)
msim_fig.savefig(fig_name, dpi=100)
  
df_compare = msim.compare(output=True)
df_stats = pd.concat([df_compare.mean(axis=1), df_compare.median(axis=1), df_compare.std(axis=1)], axis=1, 
                      keys=['mean', 'median', 'std'])   

# Add containment day and r_eff for each simulation 
time_contain = np.zeros(N_sims)
reff = np.zeros(N_sims)
peak_day = np.zeros(N_sims)

for ii in range(np.size(msim.sims)):      
   
   sim = msim.sims[ii]
   # Get the time series for each simulation
   infection_TS = np.array(sim.results['new_infections'])
   reff_cv_TS = np.array(sim.results['r_eff']) # r_effective
   time = np.array(range(np.size(infection_TS)))            
   # Find containment time:  
   containment_times = time[np.where(infection_TS<=5)]
   # Want to find where three 1s are adjacent
   time_gap = np.diff(containment_times) 
   gap_neighbourhood= time_gap[0:-3]+time_gap[1:-2]+time_gap[2:-1]        
   #peak_day = scs.find_peaks(smooth_curve(infection_TS,5),width=5)[0][0]
   
   # r_eff for the simulation is the average of all non-zero and n
   # divergent values
   reff_cv_TS[np.where(~np.isfinite(reff_cv_TS))]= 0
   reff_cv_TS[np.where(reff_cv_TS>15)]= 0
   reff_cv_TS[np.where(reff_cv_TS<0)]= 0      
   reff[ii] = np.mean(reff_cv_TS[np.where(reff_cv_TS!=0)])              
   
   if np.size(np.where(gap_neighbourhood==3)) != 0:
       time_contain[ii] = np.min(containment_times[np.where(gap_neighbourhood==3)])+2
   else:
       time_contain[ii] = 500 # Add failsafe - if not captured by sim, set 500 

  
    
df_stats =  df_stats.append(pd.Series(data={'mean':H_Pow, 'median':H_Pow,'std':H_Pow}, name='H_Pow'))
FracVac = np.sum(phase_target['vals'])/pop_size
df_stats =  df_stats.append(pd.Series(data={'mean': FracVac, 'median':FracVac,'std': FracVac}, name='FracVac'))
df_stats =  df_stats.append(pd.Series(data={'mean':np.mean(time_contain), 'median': np.median(time_contain),'std': np.std(time_contain)}, name='contain_day'))
#df_stats =  df_stats.append(pd.Series(data={'mean':np.mean(peak_day), 'median': np.mean(peak_day),'std': np.std(peak_day)}, name='peak_day'))
df_stats =  df_stats.append(pd.Series(data={'mean':np.mean(reff), 'median': np.median(reff),'std': np.std(reff)}, name='reff'))
   

file_name = 'Janurary_Hesitancy_GeneralStats.csv'

df_stats.transpose().to_csv(file_name)


''' Snippits from the old code
    pars = dict(

        default = dict(
            wild   = 1.0,
            b117   = 1.0,
            b1351  = 1.0,
            p1     = 1.0,
            b16172 = 1.0,
        ),

        pfizer = dict(
            wild   = 1.0,
            b117   = 1/2.0,
            b1351  = 1/6.7,
            p1     = 1/6.5,
            b16172 = 1/2.9, # https://www.researchsquare.com/article/rs-637724/v1
        ),

        moderna = dict(
            wild   = 1.0,
            b117   = 1/1.8,
            b1351  = 1/4.5,
            p1     = 1/8.6,
            b16172 = 1/2.9,  # https://www.researchsquare.com/article/rs-637724/v1
        ),

        az = dict(
            wild   = 1.0,
            b117   = 1/2.3,
            b1351  = 1/9,
            p1     = 1/2.9,
            b16172 = 1/6.2,  # https://www.researchsquare.com/article/rs-637724/v1
        ),

        jj = dict(
            wild   = 1.0,
            b117   = 1.0,
            b1351  = 1/6.7,
            p1     = 1/8.6,
            b16172 = 1/6.2,  # Assumption, no data available yet
        ),

        novavax = dict( # Data from https://ir.novavax.com/news-releases/news-release-details/novavax-covid-19-vaccine-demonstrates-893-efficacy-uk-phase-3
            wild   = 1.0,
            b117   = 1/1.12,
            b1351  = 1/4.7,
            p1     = 1/8.6, # Assumption, no data available yet
            b16172 = 1/6.2, # Assumption, no data available yet
        ),
    )

    if default:
        return pars['default']
    else:
        return pars
'''



































