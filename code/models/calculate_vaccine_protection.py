# This script will simulate a QLD population with 0 infections of delta, then omicron 
# entering after the entire population has been given two doses of pfizer.

# This version will output measurements of immunity to: transmission, symptoms and
# severe infection (sys_imm, symp_imm and sev_imm respectively). We will also 
# output the nab levels in time. 

# Requires population files created by Generate_HesitancyPop.py.

import covasim as cv
import covasim.parameters as cvp
import pandas as pd
import numpy as np
from py import process
import sciris as sc   
import pdb 
import matplotlib.pyplot as plt
from covasim.immunity import nab_to_efficacy

#Multiprocessing 
from multiprocessing import Pool


# Cluster_Size = 20
# inf_day = 21 # day to infect
Cluster_Size = 10000 # Number of people to infect
inf_day = 43 # day to infect
AgeSpecific = ""
ArrayNum = 1
sims = []

## Class to measure the protection to sus, symp and sev from nabs
class store_efficacy(cv.Analyzer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # This is necessary to initialize the class properly
        self.t = []
        self.eff_sus = []
        self.eff_symp = []
        self.eff_sev = []
        return

    def apply(self, sim):
        ppl = sim.people # Shorthand
        self.t.append(sim.t)

        nabs = ppl.nab
        # 
        vax_pars = sim.pars['interventions'][2].p['nab_eff']
        
        eff_susc = nab_to_efficacy(nabs, 'sus', vax_pars)
        self.eff_sus.append(eff_susc.mean())

        eff_symp = nab_to_efficacy(nabs, 'symp', vax_pars)
        self.eff_symp.append(eff_symp.mean())
        
        eff_sev = nab_to_efficacy(nabs, 'sev', vax_pars)
        self.eff_sev.append(eff_sev.mean())
        return

    def plot(self):
        plt.figure()
        plt.plot(self.t, self.eff_sus, label='eff_sus', color='blue')
        plt.plot(self.t, self.eff_symp, label='eff_symp', color='red')
        plt.plot(self.t, self.eff_sev, label='eff_sev', color='green')
        plt.legend()
        plt.xlabel('Day')
        plt.ylabel('vax_efficacy')
        sc.setylim() # Reset y-axis to start at 0
        sc.commaticks() # Use commas in the y-axis labels
        return




# sim_counter = 0 
def run_sims(Variant_Name,Hesitancy_Name,Seed):

        tt = 1  # Testing and tracing, 1 for yes and 0 for no

        pop_size = 2e5

        # Age groups describe starting age for population
        AgeGroups_Dict = {
        "August2020":      [15,17,24,34,44,54,64,200],  
        "January2021":     [15,17,24,34,44,54,64,200],
        "April2021":       [15,17,24,34,44,54,64,200],
        "April_12_2021":   [11,17,24,34,44,54,64,200],
        "LowHesitancy":    [11,17,24,34,44,54,64,200],
        "FullHesitancy":   [11,17,24,34,44,54,64,200],
        "NoHesitancy":     [-1,17,24,34,44,54,64,200]
        }

        AgeGroups = AgeGroups_Dict[Hesitancy_Name]
        # Hesitancy values from Biddel et al., 2021, note <18 is estimated by 35-44 age
        Hesitancy_Dict = {
        "August2020":      (0.163,0.092,0.171,0.117,0.163,0.104,0.080),  
        "January2021":     (0.247,0.145,0.298,0.247,0.270,0.121,0.094),
        "April2021":       (0.174,0.150,0.162,0.174,0.153,0.133,0.080),
        "April_12_2021":   (0.174,0.150,0.162,0.174,0.153,0.133,0.080),
        "LowHesitancy":    (0.080,0.080,0.080,0.080,0.080,0.080,0.080),
        "FullHesitancy":   (1.000,1.000,1.000,1.000,1.000,1.000,1.000),
        "NoHesitancy":     (0.000,0.000,0.000,0.000,0.000,0.000,0.000)
        }

        Hesitancy = Hesitancy_Dict[Hesitancy_Name]
        # print(Hesitancy)

        if AgeSpecific == 0: # If non-specific
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
            if AgeGroups[0] == 11: # 12+
                AgeGroupSize.append(round(qld_pop['10-14']/5*3)+round(qld_pop['15-19']/5*3)) #12-17
            else: # 16+
                AgeGroupSize.append(round(qld_pop['15-19']/5*2)) #16-17

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

        popfile = 'inputs/qldppl-abs2020-' + str(int(pop_size)) + '-' + Variant_Name + '-ClusterSize' + str(int(Cluster_Size))+'.pop'

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
            for ii in range(len(hes)):
                # Find the index for the people in this age group
                this_age_group = cv.true(((sim.people.age > ages[ii] )*(sim.people.age <= ages[ii+1])))
                this_groupsize = int(np.size(this_age_group)*hes[ii]) # Find the number of hesitant people 
                # Get the ID of the hesitant people      

                this_agent_uids = np.random.choice(sim.people.uid[this_age_group], this_groupsize, replace=False)
                hesitancy_uids = np.append(hesitancy_uids,this_agent_uids) # Add the IDs to the output vector 

                hesitancy_uids = hesitancy_uids[1:-1] # Remove the first element - it's a placeholder

            return hesitancy_uids

        #################################################################################

        start_date = '2020-12-10' # Start with infections on 2021-01-01
        end_date = '2021-03-31' 
        iq_factor = 0.01 # high compliance of quarantine and isolation

        # Importing infections   
        # When infections are imported

        # Define list with dates and number of imports and virus variant

        if Variant_Name == 'alpha':
            imported_infections = cv.variant('alpha', label='alpha', days= inf_day, n_imports=Cluster_Size, rescale=False) 
        elif Variant_Name == 'delta':
            # delta = dict(
            #             rel_beta        = 2.2, # Estimated to be 1.25-1.6-fold more transmissible than B117: https://www.researcchsquare.com/article/rs-637724/v1
            #             rel_symp_prob   = 1.0,
            #             rel_severe_prob = 3.2, # 2x more transmissible than alpha from https://mobile.twitter.com/dgurdasani1/status/1403293582279294983
            #             rel_crit_prob   = 1.0,
            #             rel_death_prob  = 1.0,
            #         )
            imported_infections = cv.variant(variant = 'delta', label = 'delta', days=inf_day, n_imports=Cluster_Size, rescale=False)

        elif Variant_Name == 'omicron': # From covasim 3.1.2
            omicron = dict(
            rel_beta        = 3.0, # Estimated to be 1.25-1.6-fold more transmissible than B117: https://www.researchsquare.com/article/rs-637724/v1
            rel_symp_prob   = 1.0,
            rel_severe_prob = 0.8, # 2x more transmissible than alpha from https://mobile.twitter.com/dgurdasani1/status/1403293582279294983
            rel_crit_prob   = 0.5,
            rel_death_prob  = 1.0,
            )
            imported_infections = cv.variant(variant = omicron, label  = 'omicron', days=inf_day, n_imports=Cluster_Size, rescale=False)
            # Label it b117 to ensure it uses the correct immunity
            # imported_infections = cv.variant(variant = omicron, label  = 'b117', days=21, n_imports=Cluster_Size, rescale=False)


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
        'pop_scale': 5100000/pop_size, # Population scales to 5.1M ppl in QLD
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
        'rand_seed': 21 + Seed,
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

        # sim_hesitance = cv.Sim(pars = pars,popfile=popfile, load_pop=True,analyzers=store_efficacy())
        sim_hesitance = cv.Sim(pars = pars,popfile=popfile, load_pop=True,label = Variant_Name + '_' + Hesitancy_Name)
        sim_hesitance.initialize()  # need to intialize to generate People()   

        hesitancy_uids = get_hesitancy_uids(sim_hesitance,Hesitancy,AgeGroups)

        # print('Number of Hesitant Agents = ',  len(hesitancy_uids))

        # Define if we want testing and tracing    
        if tt == 1:    
            sim_hesitance = testing_and_tracing_interventions(sim_hesitance, start_date, end_date, label='tt')        
        else:
            sim_hesitance = testing_and_tracing_interventions(sim_hesitance, start_date, end_date, label='no-tt')

        # Vaccination with hesitancy    
        FullPopulation = sim_hesitance.people.uid
        FullVacc_pop_vals = np.ones(np.size(FullPopulation)) # 1 is vaccinate, 0 is leave
        FullVacc_pop_vals[sim_hesitance.people.age <= AgeGroups[0]] = 0 #Remove under age
        FullVacc_pop_vals[hesitancy_uids] = 0 #Remove hesitants    

        # Effectiveness of pfizer on different strains. From covasim parameters,
        # with omicron taken from the unpublished branch at:
        # https://github.com/amath-idm/covasim/blob/gr-feb2022-djkongoing/covasim/parameters.py
        #  
        #    pfizer = dict(
        #             wild  = 1.0,
        #             alpha = 1/2.0, # https://www.nejm.org/doi/full/10.1056/nejmc2100362
        #             beta  = 1/10.3, # https://www.nejm.org/doi/full/10.1056/nejmc2100362
        #             gamma = 1/6.7, # https://www.nejm.org/doi/full/10.1056/nejmc2100362
        #             delta = 1/2.9, # https://www.researchsquare.com/article/rs-637724/v1
        #             omicron = 1/40 # https://www.medrxiv.org/content/10.1101/2021.12.28.21268481v1
        #         )

        if Variant_Name == 'alpha': # Alpha
            vax_efficacy = 1/2.0
        elif Variant_Name == 'delta':  # Delta
            vax_efficacy = 1/2.9
        elif Variant_Name == 'omicron': # Omicron
            vax_efficacy = 1/40

        # # Relevant code from covasim - hybridize pfizer to have the same effectiveness,
        # # but only a single dose
        # default_nab_eff = dict(
        #      alpha_inf      =  1.11,
        #      beta_inf       =  1.219,
        #      alpha_symp_inf = -1.06,
        #      beta_symp_inf  =  0.867,
        #      alpha_sev_symp =  0.268,
        #      beta_sev_symp  =  3.4
        #  )    

        # pfizer_custom = dict(
        #     nab_eff   = sc.dcp(default_nab_eff),
        #     nab_init  = dict(dist='normal', par1=2, par2=2),
        #     nab_boost = 3,
        #     doses     = 1,
        #     interval  = None,
        #     p = {Variant_Name: vax_efficacy}
        # )                

        dose_pars = cvp.get_vaccine_dose_pars()['pfizer']
        variant_pars = cvp.get_vaccine_variant_pars()['pfizer']
        variant_pars[Variant_Name] = vax_efficacy
        pfizer_custom = sc.mergedicts({'label':'pfizer'}, sc.mergedicts(dose_pars, variant_pars))

        # vaccinate reads a dictionary of uids to vaccinate and the probability of
        # vaccinating.     
        vaccination_dict = dict(inds = FullPopulation, vals = FullVacc_pop_vals)

        # vaccine = cv.vaccinate(vaccine = 'pfizer', days= 0, subtarget=vaccination_dict,label = 'pfizer')
        # # Add this variant and corresponding efficacy to the vaccine
        # vaccine.p[Variant_Name] = vax_efficacy

        vaccine = cv.vaccinate(vaccine = pfizer_custom, days= 0, subtarget=vaccination_dict,label = 'pfizer') # rel_sus = 1-vax_efficacy,
        # vaccine = cv.simple_vaccine(days= 0,rel_sus = 1-vax_efficacy, subtarget=vaccination_dict) # rel_sus = 1-vax_efficacy,

        sim_hesitance.pars['interventions'].append(vaccine)
        sim_hesitance.initialize()  #need to initialize again otherwise covasim complains that the intervention has not been initialized
        msim = sim_hesitance
        msim.run()

        return msim

        # sim_hesitance.pars['vaccine_pars']['pfizer']['b16172'] =  1/2.9
        # sim_hesitance.pars['vaccine_pars']['pfizer'][Variant_Name] =  vax_efficacy
        # msim = sim_hesitance
        # sims.append(sim_hesitance)
        # print(Variant_Name + '_' + Hesitancy_Name)




# # What I need  to run
# msim = cv.MultiSim(sims)
# # pdb.set_trace()
# msim.run(n_runs = 10)  
# msim.plot_result('cum_infections')


if __name__ == '__main__':
    #Run settings
    # n_workers = 4 

    #Create and queue the Dask jobs
    # client = Client(n_workers=n_workers)
    # queued = []
    n_runs = 10 
    scale = 5100000/2e5

    Inputs = []   
    #Define parameters to be put into multiprocessing method
    for Variant_Name in ("delta","omicron"):
        for Hesitancy_Name in ("FullHesitancy","NoHesitancy"):
            for seed in range(n_runs):
                Inputs += [(Variant_Name,Hesitancy_Name,seed)]

    processes = 10 # number of cores
    with Pool(processes=processes) as pool: 
        sims_parallel = pool.starmap(run_sims,Inputs)

    # sims_parallel.plot_result('cum_infections')
    

    # Estimate protection from infections, symptoms and severity
    total_infectious_vax = np.zeros((2,n_runs))
    total_symptomatic_vax = np.zeros((2,n_runs))
    total_severe_vax = np.zeros((2,n_runs))
    total_infectious_unvax = np.zeros((2,n_runs))
    total_symptomatic_unvax = np.zeros((2,n_runs))
    total_severe_unvax = np.zeros((2,n_runs))

    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    # day to look at efficacy - should be away from saturation, but long enough for severe cases to appear
    measure_day = 43 + 14 
    for this_sim in sims_parallel:

        if this_sim.label == "delta_NoHesitancy":
            total_infectious_vax[0,c1] = np.array(this_sim.results['cum_infectious'][measure_day])
            total_symptomatic_vax[0,c1] = np.array(this_sim.results['cum_symptomatic'][measure_day])
            total_severe_vax[0,c1] = np.array(this_sim.results['cum_severe'][measure_day])
            c1 = c1+1

        elif this_sim.label == "delta_FullHesitancy":
            total_infectious_unvax[0,c2] = np.array(this_sim.results['cum_infectious'][measure_day])
            total_symptomatic_unvax[0,c2] = np.array(this_sim.results['cum_symptomatic'][measure_day])
            total_severe_unvax[0,c2] = np.array(this_sim.results['cum_severe'][measure_day])
            c2 = c2+1

        elif this_sim.label == "omicron_NoHesitancy":
            total_infectious_vax[1,c3] = np.array(this_sim.results['cum_infectious'][measure_day])
            total_symptomatic_vax[1,c3] = np.array(this_sim.results['cum_symptomatic'][measure_day])
            total_severe_vax[1,c3] = np.array(this_sim.results['cum_severe'][measure_day])
            c3 = c3+1

        elif this_sim.label == "omicron_FullHesitancy":
            total_infectious_unvax[1,c4] = np.array(this_sim.results['cum_infectious'][measure_day])
            total_symptomatic_unvax[1,c4] = np.array(this_sim.results['cum_symptomatic'][measure_day])
            total_severe_unvax[1,c4] = np.array(this_sim.results['cum_severe'][measure_day])
            c4 = c4+1

    # Estimate the protection between runs
    print('Delta')
    id_var = 0
    inf_eff = 1-np.mean(total_infectious_vax[id_var,:]/total_infectious_unvax[id_var,:])    
    symp_eff = 1-np.mean((total_symptomatic_vax[id_var,:]/total_infectious_vax[id_var,:]) / (total_symptomatic_unvax[id_var,:]/total_infectious_unvax[id_var,:])) 
    sev_eff = 1-np.mean((total_severe_vax[id_var,:]/total_symptomatic_vax[id_var,:]) / (total_severe_unvax[id_var,:]/total_symptomatic_unvax[id_var,:])) 
    # symp_eff = 1-np.mean(inf_eff*total_symptomatic_vax[id_var,:]/total_symptomatic_unvax[id_var,:])
    # sev_eff = 1-np.mean(inf_eff*symp_eff*total_severe_vax[id_var,:]/total_severe_unvax[id_var,:])
    print('Protection against infection = ' + str(inf_eff*100) + '%')
    print('Protection against symptoms = ' + str(symp_eff*100) + '%')
    print('Protection against severe infection = ' + str(sev_eff*100) + '%')

    print('Omicron')
    id_var = 1
    inf_eff = 1-np.mean(total_infectious_vax[id_var,:]/total_infectious_unvax[id_var,:])    
    symp_eff = 1-np.mean((total_symptomatic_vax[id_var,:]/total_infectious_vax[id_var,:]) / (total_symptomatic_unvax[id_var,:]/total_infectious_unvax[id_var,:])) 
    sev_eff = 1-np.mean((total_severe_vax[id_var,:]/total_symptomatic_vax[id_var,:]) / (total_severe_unvax[id_var,:]/total_symptomatic_unvax[id_var,:])) 
    print('Protection against infection = ' + str(inf_eff*100) + '%')
    print('Protection against symptoms = ' + str(symp_eff*100) + '%')
    print('Protection against severe infection = ' + str(sev_eff*100) + '%')

    pdb.set_trace()  
    cum_infectious = {}
    cum_symptomatic = {}
    cum_severe = {}
    cum_critical = {}
    cum_deaths = {}
    count = 1
    for this_sim in sims_parallel:
        cum_infectious[this_sim.label + '_'+str(count % n_runs)] = np.array(this_sim.results['cum_infectious'])
        cum_symptomatic[this_sim.label + '_'+str(count % n_runs)] = np.array(this_sim.results['cum_symptomatic'])
        cum_severe[this_sim.label + '_'+str(count % n_runs)] = np.array(this_sim.results['cum_severe'])
        cum_critical[this_sim.label + '_'+str(count % n_runs)] = np.array(this_sim.results['cum_critical'])
        cum_deaths[this_sim.label + '_'+str(count % n_runs)] = np.array(this_sim.results['cum_deaths'] )  
        count = count+1
        
    Infectious = pd.DataFrame(cum_infectious)
    Infectious.to_csv('Infectious.csv')
    Symptomatic = pd.DataFrame(cum_symptomatic)
    Symptomatic.to_csv('Symptomatic.csv')
    Severe = pd.DataFrame(cum_severe)
    Severe.to_csv('Severe.csv')
    Critical = pd.DataFrame(cum_critical)
    Critical.to_csv('Critical.csv')
    Deaths = pd.DataFrame(cum_deaths)
    Deaths.to_csv('Deaths.csv')


