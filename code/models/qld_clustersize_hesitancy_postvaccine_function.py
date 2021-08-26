def qld_clustersize_hesitancy_postvaccine_function(Hesitancy,AgeGroups,N_sims,H_Pow,Cluster_Size,tt,N_CPUs): 
    import covasim as cv
    import pandas as pd
    import numpy as np
   
    Variant = ['b117', 'b16172']
    kk = 1 # variant number, 0 = alpha, 1 = delta
    pop_size = 2e5
      
    popfile = 'inputs/qldppl-abs2020-200k-' + Variant[kk] + '-ClusterSize' + str(int(Cluster_Size))+'.pop'
    
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
        hesitancy_uids = sim.people.uid[cv.true(sim.people.age < 18)]
        
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
            'rand_seed': 21 ,
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
       
           
    # vaccinate reads a dictionary of uids to vaccinate and the probability of
    # vaccinating.     
    vaccination_dict = dict(inds = FullPopulation, vals = FullVacc_pop_vals)                   
    # vaccine = cv.vaccinate(vaccine = pfizer_custom, days= 0, subtarget=vaccination_dict,label = 'pfizer') # rel_sus = 1-vax_efficacy,
    vaccine = cv.vaccinate(vaccine = 'pfizer', days= 0, subtarget=vaccination_dict,label = 'pfizer')
    # vaccine = cv.simple_vaccine(days= 0,rel_sus = 1-vax_efficacy, subtarget=vaccination_dict) # rel_sus = 1-vax_efficacy,
    sim_hesitance.pars['interventions'].append(vaccine)
    sim_hesitance.initialize()  #need to initialize again otherwise covasim complains that the intervention has not been initialized
    # sim_hesitance.pars['vaccine_pars']['pfizer']['b16172'] =  1/2.9

    msim = cv.MultiSim(sim_hesitance,par_args={'ncpus': N_CPUs}, verbose = 0)
    msim.run(n_runs=N_sims, reseed=True)
    
      
    # Add containment day and r_eff for each simulation 
    time_contain = np.zeros(N_sims)
    reff = np.zeros(N_sims)
    reff_30 = np.zeros(N_sims)
    
    for ii in range(np.size(msim.sims)):      
       
       sim = msim.sims[ii]
       # Get the time series for each simulation
       infection_TS = np.array(sim.results['new_infections'])
       reff_cv_TS = np.array(sim.results['r_eff']) # r_effective
       reff_30[ii] = np.array(sim.results['r_eff'][51])
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
       #peak_days =    scs.find_peaks(smooth_curve(infection_TS,5),width=5)      
       #if np.size(peak_days[0]) != 0:   
       #    peak_day[ii] = peak_days[0][0]
       #else: 
       #    peak_day[ii] = 1000      
    df_compare = msim.compare(output=True)
    df_stats = pd.concat([df_compare.mean(axis=1), df_compare.median(axis=1), df_compare.std(axis=1)], axis=1, 
                         keys=['mean', 'median', 'std'])   
    df_stats =  df_stats.append(pd.Series(data={'mean':H_Pow, 'median':H_Pow,'std':H_Pow}, name='H_Pow'))
    FracVac = np.size(hesitancy_uids)/pop_size
    df_stats =  df_stats.append(pd.Series(data={'mean': FracVac, 'median':FracVac,'std': FracVac}, name='FracVac'))
    df_stats =  df_stats.append(pd.Series(data={'mean':np.mean(time_contain), 'median': np.median(time_contain),'std': np.std(time_contain)}, name='contain_day'))
    #df_stats =  df_stats.append(pd.Series(data={'mean':np.mean(peak_day), 'median': np.mean(peak_day),'std': np.std(peak_day)}, name='peak_day'))
    df_stats =  df_stats.append(pd.Series(data={'mean':np.mean(reff), 'median': np.median(reff),'std': np.std(reff)}, name='reff'))
    df_stats =  df_stats.append(pd.Series(data={'mean':np.mean(reff_30), 'median': np.median(reff_30),'std': np.std(reff_30)}, name='reff_30'))
    

    if kk == 1:    
        file_name = "Results/Delta_Variant/ClusterSize_Hesitancy_Data/Hesitancy_" + str(H_Pow) + "ClusterSize_" + str(int(Cluster_Size)) + "_PostVaccineTT.csv"
        # fig_name = "Hesitancy_PostVaccine_Data/H_Pow_" + str(H_Pow) + "ClusterSize_" + str(Cluster_Size) + "_PostVaccineTT.pdf"
        # sim_name = "Hesitancy_PostVaccine_Data/H_Pow_" + str(H_Pow) + "ClusterSize_" + str(Cluster_Size) + "_PostVaccine_SimTT.obj"
      
    else:       
        file_name = "Results/Alpha_Variant/ClusterSize_Hesitancy_Data/Hesitancy_" + str(H_Pow) + "ClusterSize_" + str(int(Cluster_Size)) + "_PostVaccine.csv"
        # fig_name = "Hesitancy_PostVaccine_Data/H_Pow_" + str(H_Pow) + "ClusterSize_" + str(Cluster_Size) + "_PostVaccine.pdf"
        # sim_name = "Hesitancy_PostVaccine_Data/H_Pow_" + str(H_Pow) + "ClusterSize_" + str(Cluster_Size) + "_PostVaccine_Sim.obj"
 
 
    df_stats.transpose().to_csv(file_name)
      