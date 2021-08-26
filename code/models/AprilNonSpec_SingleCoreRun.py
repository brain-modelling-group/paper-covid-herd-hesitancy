def AprilNonSpec_SingleCoreRun(Cluster_Size,ArrayNum):       
    import covasim as cv
    import pandas as pd
    import numpy as np
    import sciris as sc
    
    AgeGroups = [17,24,34,44,54,64,200]   
    tt = 1  # Testing and tracing, 1 for yes and 0 for no
    Variant = ['b117', 'b16172']
    kk = 1 # variant number, 0 = alpha, 1 = delta
    pop_size = 2e5
    
    # Hesitancy values from Biddel et al., 2021
    Hesitancy_Apr21 = (0.150,0.162,0.174,0.153,0.133,0.08)
    Hesitancy = Hesitancy_Apr21    
    
    # Convert to non-specific values using the population values
    
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
    # Get population in each category
    AgeGroupSize = []
    AgeGroupSize.append(round(qld_pop['15-19']/5*2)+qld_pop['20-24']) #18-24
    AgeGroupSize.append(qld_pop['25-29']+qld_pop['30-34']) #25-34
    AgeGroupSize.append(qld_pop['35-39']+qld_pop['40-44']) #35-44
    AgeGroupSize.append(qld_pop['45-49']+qld_pop['50-54']) #45-54
    AgeGroupSize.append(qld_pop['55-59']+qld_pop['60-64']) #55-64
    AgeGroupSize.append(qld_pop['65-69']+qld_pop['70-74'] + qld_pop['75-79']+qld_pop['80-84'] + qld_pop['85-89']+qld_pop['90-94'] + qld_pop['95-99']+qld_pop['100-104']) #65+

       
    # Convert age-specific hesitancy to the expected fraction of unvaccinated
    # population.
    
    PopSize = sum(AgeGroupSize)
    FracUnvaccinated = np.dot(AgeGroupSize,Hesitancy)/PopSize
    Hesitancy = np.ones(np.shape(Hesitancy))
    Hesitancy = FracUnvaccinated*Hesitancy # Evenly spread hesitancy
    
    
    popfile = '../inputs/qldppl-abs2020-200k-' + Variant[kk] + '-ClusterSize' + str(Cluster_Size)+'.pop'
    
    # Testing and tracing probabilities for the 14-layer model
    def testing_and_tracing_interventions(sim,start_simulation_date, end_simulation_date, label, num_tests=7501):
      
      ntpts = sim.day(end_simulation_date)-sim.day(start_simulation_date)
    
      if label == 'tt': 
        # Testing
        sim.pars['interventions'].append(cv.test_num(daily_tests=[num_tests]*ntpts, 
                                                     start_day=sim.day(start_simulation_date), 
                                                     end_day=sim.day(end_simulation_date)+5, 
                                                     symp_test=100.0, test_delay=0))
    
        # Tracing
        trace_probs = {'H': 1.00, 'S': 0.95, 
                       'W': 0.80, 'C': 0.05, 
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
    
        trace_time = {'H': 1, 'S': 2, 
                      'W': 2, 'C': 14, 
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
      
    start_date = '2020-12-10' # Start with infections on 2021-01-01
    end_date = '2021-03-31' 
    iq_factor = 0.01 # high compliance of quarantine and isolation
    
    # Importing infections   
    # When infections are imported
    
    # Define list with dates and number of imports and virus variant
    
    if Variant[kk] == 'b117':
        imported_infections = cv.variant('uk', days= 21, n_imports=Cluster_Size, rescale=False) 
    elif Variant[kk] == 'b16172':
        b16172 = dict(
                    rel_beta        = 2.2, # Estimated to be 1.25-1.6-fold more transmissible than B117: https://www.researchsquare.com/article/rs-637724/v1
                    rel_symp_prob   = 1.0,
                    rel_severe_prob = 3.2, # 2x more transmissible than alpha from https://mobile.twitter.com/dgurdasani1/status/1403293582279294983
                    rel_crit_prob   = 1.0,
                    rel_death_prob  = 1.0,
                )
        # imported_infections = cv.variant(variant = b16172, days=22, n_imports=Cluster_Size, rescale=False)
        imported_infections = cv.variant(variant = 'b16172', days=21, n_imports=Cluster_Size, rescale=False)
     
    
    
    all_layers = ['H', 'S', 'W', 'C', 
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
    
    dynam_layers = ['C', 'entertainment', 'cafe_restaurant', 'pub_bar', 'transport', 'public_parks', 'large_events']
    
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
            'pop_infected': 0,    # Original population infedcted
            'rescale': False,      # Population dynamics rescaling
            'dynam_layer': pd.Series([0.0,     0.0,    0.0,    1.0,   0.0,    0.0,    0.0,      1.0,    1.0,    1.0,    1.0,     1.0,        1.0,    0.0], index=all_layers).to_dict(),
                                      # H        S       W       C   church   psport  csport    ent     cafe    pub     trans    park        event    soc
            'contacts':    pd.Series([4.0,    21.0,    5.0,    1.0,   20.0,   40.0,    30.0,    25.0,   19.00,  30.00,   25.00,   10.00,     50.00,   6.0], index=all_layers).to_dict(),
            'beta_layer':  pd.Series([1.0,     0.3,    0.2,    0.1,    0.04,   0.2,     0.1,     0.01,   0.04,   0.06,    0.16,    0.03,      0.01,   0.3], index=all_layers).to_dict(),
            'iso_factor':  pd.Series([1.0, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor], index=all_layers).to_dict(),
            'quar_factor': pd.Series([1.0, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor], index=all_layers).to_dict(),
            'n_variants': 2,
            'variants': imported_infections,
            'location': 'Australia',
            'rand_seed': 21 + ArrayNum,
            'start_day': start_date,
            'end_day':   end_date,
            'use_waning': True,
            'verbose': 1}
    
   
    # Define if we want testing and tracing
    if tt == 1:
        label = 'tt'
    else:
        label == 'no-tt'    
     
    #--------------------------------------------------------------------------------------------------------------------------------------------#
    
    # Hesitancy: Run simulation with hesitancy  

    sim_hesitance = cv.Sim(pars = pars,popfile=popfile, load_pop=True)
    sim_hesitance.initialize()  # need to intialize to generate People()   
    
    hesitancy_uids = get_hesitancy_uids(sim_hesitance,Hesitancy,AgeGroups)
    
    print('Number of Hesitant Agents = ',  len(hesitancy_uids))
    
    # Define if we want testing and tracing    
    if tt == 1:    
        sim_hesitance = testing_and_tracing_interventions(sim_hesitance, start_date, end_date, label='tt')        
    else:
        sim_hesitance = testing_and_tracing_interventions(sim_hesitance, start_date, end_date, label='no-tt')
        
    # Vaccination with hesitancy    
    FullPopulation = sim_hesitance.people.uid
    FullVacc_pop_vals = np.ones(np.size(FullPopulation)) # 1 is vaccinate, 0 is leave
    FullVacc_pop_vals[sim_hesitance.people.uid< 18] = 0 #Remove under 18
    FullVacc_pop_vals[hesitancy_uids] = 0 #Remove hesitants    

    # Effectiveness of pfizer on different strains is here:
            # pfizer = dict(
            #     wild   = 1.0,
            #     b117   = 1/2.0,
            #     b1351  = 1/6.7,
            #     p1     = 1/6.5,
            #     b16172 = 1/2.9, # https://www.researchsquare.com/article/rs-637724/v1
            # )
            
    if kk == 0: # Alpha
        vax_efficacy = 1/2.0
    else:  # Delta
        vax_efficacy = 1/2.9
        
    # Relevant code from covasim - hybridize pfizer to have the same effectiveness,
    # but only a single dose
    default_nab_eff = dict(
         alpha_inf      =  1.11,
         beta_inf       =  1.219,
         alpha_symp_inf = -1.06,
         beta_symp_inf  =  0.867,
         alpha_sev_symp =  0.268,
         beta_sev_symp  =  3.4
     )    
    pfizer_custom = dict(
        nab_eff   = sc.dcp(default_nab_eff),
        nab_init  = dict(dist='normal', par1=2, par2=2),
        nab_boost = 3,
        doses     = 1,
        interval  = None,
    )     
           
           
    # vaccinate reads a dictionary of uids to vaccinate and the probability of
    # vaccinating.     
    vaccination_dict = dict(inds = FullPopulation, vals = FullVacc_pop_vals)                   
    # vaccine = cv.vaccinate(vaccine = pfizer_custom, days= 0, subtarget=vaccination_dict,label = 'pfizer') # rel_sus = 1-vax_efficacy,
    vaccine = cv.vaccinate(vaccine = 'pfizer', days= 0, subtarget=vaccination_dict,label = 'pfizer')
    # vaccine = cv.simple_vaccine(days= 0,rel_sus = 1-vax_efficacy, subtarget=vaccination_dict) # rel_sus = 1-vax_efficacy,
    sim_hesitance.pars['interventions'].append(vaccine)
    sim_hesitance.initialize()  #need to initialize again otherwise covasim complains that the intervention has not been initialized
    # sim_hesitance.pars['vaccine_pars']['pfizer']['b16172'] =  1/2.9

    msim = sim_hesitance
    msim.run()     
    
    #--------------------Analysis-------------------------#
    
    
    # Find the number of people in each age group who were vaccinated, infected or
    # dead. Furthermore, find the people who were infected/died AND were vaccinated 
    
    # Need it to work for multisim, or at least cycle through each simulation and 
    # save a matrix (to be averaged column-wise).
       
    
    ii = 0 # Single simulation   
    sim = msim  
    AgeBounds =  [0,17,24,34,44,54,64,200]    
    AgeLabels = ['0-17','18-24','25-34','35-44','45-54','55-64','65+']
    
    Size_per_age = np.zeros((1,np.size(AgeGroups)))
    Infected_per_age = np.zeros((1,np.size(AgeGroups)))
    Dead_per_age = np.zeros((1,np.size(AgeGroups)))
    Vaccinated_per_age = np.zeros((1,np.size(AgeGroups)))
    VaccInfected_per_age = np.zeros((1,np.size(AgeGroups)))
    VaccDead_per_age = np.zeros((1,np.size(AgeGroups)))
    Critical_per_age = np.zeros((1,np.size(AgeGroups)))
    VaccCritical_per_age = np.zeros((1,np.size(AgeGroups)))
    for jj in range(np.size(AgeBounds)-1):
      
        # Find the index for the people in this age group
        this_age_group = cv.true(((sim.people.age > AgeBounds[jj] )*(sim.people.age <= AgeBounds[jj+1])))  
        # Get the ID of this age group         
        this_age_uids = sim.people.uid[this_age_group]
        # Find which of the following IDs have the following attributes:
        Size_per_age[ii,jj] = np.size(this_age_uids)/pop_size
        Infected_per_age[ii,jj] = np.sum(msim.people.date_infectious[this_age_group]>=0)/pop_size
        Infected_per_age[ii,jj] = np.sum(msim.people.date_infectious[this_age_group]>=0)/pop_size
        Dead_per_age[ii,jj] = np.sum(msim.people.date_dead[this_age_group]>=0)/pop_size
        Vaccinated_per_age[ii,jj] = np.sum(msim.people.date_vaccinated[this_age_group]>=0)/pop_size
        VaccInfected_per_age[ii,jj] = np.sum((msim.people.date_vaccinated[this_age_group]>=0)*(msim.people.date_infectious[this_age_group]>0))/pop_size
        VaccDead_per_age[ii,jj] = np.sum((msim.people.date_vaccinated[this_age_group]>=0)*(msim.people.date_dead[this_age_group]>0))/pop_size
        Critical_per_age[ii,jj] = np.sum(msim.people.date_critical[this_age_group]>=0)/pop_size
        VaccCritical_per_age[ii,jj] = np.sum((msim.people.date_vaccinated[this_age_group]>=0)*(msim.people.date_critical[this_age_group]>0))/pop_size
    
    # Finding mean and std for each
    AgeSpecificData = pd.DataFrame(data = [np.mean(Size_per_age,0),
                              np.mean(Vaccinated_per_age,0),
                              np.mean(Infected_per_age,0),
                              np.mean(VaccInfected_per_age,0),
                              np.mean(Dead_per_age,0),
                              np.mean(VaccDead_per_age,0),                              
                              np.mean(Critical_per_age,0),
                              np.mean(VaccCritical_per_age,0),
                              np.std(Size_per_age,0),
                              np.std(Vaccinated_per_age,0),
                              np.std(Infected_per_age,0),
                              np.std(VaccInfected_per_age,0),
                              np.std(Dead_per_age,0),
                              np.std(VaccDead_per_age,0),
                              np.std(Critical_per_age,0),
                              np.std(VaccCritical_per_age,0)],
                      index = ['Mean_Size_per_age',
                               'Mean_Vaccinated',
                               'Mean_Infected',
                               'Mean_Infected_Vaccinated',
                               'Mean_Dead',
                               'Mean_Dead_Vaccinated',
                               'Mean_Critical',
                               'Mean_Critical_Vaccinated',
                               'Std_Size_per_age',
                               'Std_Vaccinated',
                               'Std_Infected',
                               'Std_Infected_Vaccinated',
                               'Std_Dead',
                               'Std_Dead_Vaccinated',
                               'Std_Critical',
                               'Std_Critical_Vaccinated'],
                      columns = AgeLabels)   
    
    # Save infection stats and full simulation
    
    filename_agespec = "Data/AprilNonSpecDelta_AgeSpec_ClusterSize_" + str(Cluster_Size) + "_Rep_" + str(ArrayNum) + ".csv"
    AgeSpecificData.transpose().to_csv(filename_agespec)
    
    # Get the time series for each simulation
    infection_TS = np.array(sim.results['new_infections'][21:])
    reff_cv_TS = np.array(sim.results['r_eff'][21:]) # r_effective
    time = np.array(range(np.size(infection_TS)))            
    # Find containment time:  
    containment_times = time[np.where(infection_TS<=5)]
    # Want to find where three 1s are adjacent
    time_gap = np.diff(containment_times) 
    gap_neighbourhood= time_gap[0:-3]+time_gap[1:-2]+time_gap[2:-1]       
    # r_eff for the simulation is the average of all non-zero and n
    # divergent values
    reff_cv_TS[np.where(~np.isfinite(reff_cv_TS))]= 0
    reff_cv_TS[np.where(reff_cv_TS>15)]= 0
    reff_cv_TS[np.where(reff_cv_TS<0)]= 0      
    reff = np.mean(reff_cv_TS[np.where(reff_cv_TS!=0)])              
    
    if np.size(np.where(gap_neighbourhood==3)) != 0:
        time_contain = np.min(containment_times[np.where(gap_neighbourhood==3)])+2
    else:
        time_contain = 500 # Add failsafe - if not captured by sim, set 500      
  
    DynamicsData = pd.DataFrame({'Infections': np.array(sim.results['new_infections'][21:]),
                                'Deaths':     np.array(sim.results['new_deaths'][21:]),
                                'r_eff': np.array(sim.results['r_eff'][21:]),
                                'Diagnosis': np.array(sim.results['n_diagnosed'][21:]),
                                'Critical': np.array(sim.results['n_critical'][21:]),
                                'r_eff_smooth': reff_cv_TS,
                                'Containment_Time' : time_contain,
                                'r_eff_static' : reff
                                }
                       )  
        
    filename_dynamics = "Data/AprilNonSpecDelta_Dynamics_ClusterSize_" + str(Cluster_Size) + "_Rep_" + str(ArrayNum) + ".csv"
    DynamicsData.to_csv(filename_dynamics)
    
    sim_name = "Data/AprilNonSpecDelta_Sim_ClusterSize_" + str(Cluster_Size) + "_Rep_" + str(ArrayNum) + ".obj"
    msim.save(sim_name)    


  
