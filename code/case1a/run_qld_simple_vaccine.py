#!/usr/bin/env python
# coding: utf-8
"""
Use simple vaccine to get 
herd immunity, only looking at tranmission blocking, 
for the adult Queensland population, using "legacy" covasim architecture, that is, no immunity 
waning dynamics, no cv.variants(), etc

# author: Paula Sanz-Leon, QIMRB, May 2021
"""

# Import scientific python
import pandas as pd
import numpy as np
import pathlib as pathlib

# Import IDM/Optima code
import covasim as cv
import covasim_australia.analyzers as cva
import covasim_australia.interventions as cvi
import covasim.misc as cvm
import sciris as sc

# Add argument parser
import argparse


# Check covasim version is the one we actually need
cv.check_version('>=3.0.7', die=True)

parser = argparse.ArgumentParser()

parser.add_argument('--ncpus', default=8, 
                               type=int, 
                               help='''Maximum number of cpus used by MultiSim runs.''')

parser.add_argument('--nruns', default=8, 
                               type=int, 
                               help='''Number of simulations to run per scenario. 
                                       Uses different PRNG seeds for each simulations.''')

parser.add_argument('--results_path', 
                              default='results', 
                              type=str, 
                              help='''The relative and/or absolute path to the results folder, without the trailing /''')

parser.add_argument('--label', default='simple-vax', 
                               type=str, 
                               help=''' An ID string.''')

parser.add_argument('--tt_strategy', 
                              default='apply-tt', 
                              type=str, 
                              help=''' A string to determine what set of testing and tracing strategy to use. Available: 'apply-tt' or 'no-tt.''')

parser.add_argument('--global_beta', 
                              default=0.0113*1.7, 
                              type=float,
                              help='''Global transmissibility from calibrated model * 70% more tranmissible to simulate B117.''') 

parser.add_argument('--num_tests', 
                              default=7501, 
                              type=int, 
                              help='''Average number of tests per day. Value based on average over period 2021-02-15 to 2021-05-11. Std dev: 6325''')


parser.add_argument('--age_lb', 
                              default=16, 
                              type=int, 
                              help='''Age lower boundary to be within the fraction of the population that is vaccinated.''')


parser.add_argument('--iq_factor', 
                              default=1.0,
                              type=float, 
                              help=''' Isolation factor. [0: absolute adherence]; [>1: no adherence]''')

parser.add_argument('--cluster_size', 
                              default=20, 
                              type=int, 
                              help='''The number of infected people entering QLD community on a given date (default, 2020-10-01)''')

parser.add_argument('--start_simulation_date', default='2021-06-30', 
                              type=str, 
                              help='''The date at which to start simulation (eg, 2020-12-31).''')

parser.add_argument('--end_simulation_date', default='2021-10-04', 
                              type=str, 
                              help='''The date at which to stop simulation (eg, 2020-12-31).''')

parser.add_argument('--layer_betas_file', 
                              default='qld_model_layer_betas_02.csv', 
                              type=str, 
                              help='''The name of the csv file with layer-specific betas.''')

parser.add_argument('--vax_file',
                             default='../input_data/aus_air_vaccine_data.csv',
                             type=str,
                             help=''' The name of the csv file with age-specific vaccination statistics''')

parser.add_argument('--vax_current_date', default='2021-06-30', 
                              type=str, 
                              help='''The date that is consider to reflect the 'current' vaccination status.''')

parser.add_argument('--vax_age_coverage',
                              default='none',
                              type=str,
                              help='''A string with the assumption we make about age-targeted vaccine coverage.
                                     Options: worst, halfsies, best, none. Worst uses the *correct* coverage, which is 
                                     given by the proportion of the population that has received two doses.
                                     Halfsies, uses the average between the proprotion of ppl who received first dose
                                     and the proportion of ppl who received second dose. Best, assumes that that coverage
                                     is given by the proportion of people who received only one dose. none uses random vaccination.''')

parser.add_argument('--vaccinate', 
                              default='apply', 
                              type=str, 
                              help='''A string to determine whether we apply vaccine or no. Available: "apply" "donot-apply" ''')

parser.add_argument('--vax_coverage', 
                              default=0.0, 
                              type=float, 
                              help='''A float between 0 and 1. [0: no one is vaccinated] [1: 1005 of adult pop 19-64].''')

parser.add_argument('--vax_efficacy', 
                              default=0.05, 
                              type=float, 
                              help='''[0: no efficacy] [1: 100% efficacy] - only refers to tranmission blocking ''')

################################################################################################################
###########################                FUNCTIONS TO SET UP SCENARIOS                 #######################
################################################################################################################

def set_layer_betas(betasfile, layers, input_args):

        beta_data  = pd.read_csv(betasfile, parse_dates=['date'])
        betas_interventions = []
        change_date = input_args.start_simulation_date
        for this_layer in layers:
          # Load data for this layer
          beta_layer = beta_data[['date', this_layer]].set_index('date')
          betas_interventions.append(cv.change_beta(days=[change_date], 
                                                    changes= beta_layer.loc[change_date].tolist(), 
                                                    layers=[this_layer], do_plot=False))
        return betas_interventions

def set_up_layer_beta_changes(betasfile, layers):

    beta_data  = pd.read_csv(betasfile, parse_dates=['date'])
    betas_interventions = []

    for this_layer in layers:
      # Load data for this layer
      beta_layer = np.array(beta_data[this_layer])
      # Find index of change, the date in which the change is implemented is the change index + 1
      change_idx = np.argwhere(beta_layer[0:-2]-beta_layer[1:-1]) + 1
      betas_interventions.append(cv.change_beta(days=['2020-01-01'] + [beta_data['date'][change_idx.flat[ii]] for ii in range(len(change_idx.flat))], 
                                                changes= [1.0] + [beta_data[this_layer][change_idx.flat[ii]] for ii in range(len(change_idx.flat))], 
                                                layers=[this_layer], do_plot=False))
    return betas_interventions

def get_mass_vaccination_subtargets(sim, vax_coverage, age_lb):
    """
    Create dictionary needed by simple_vaccine().
    It assigns vaccines randomly to people of 18 years 
    old and above.
    
    The vaccine coverage is expressed in vax_coverage
    as a value between [0, 1]

    Returns:
    subtarget_dict

    """
    sim_pop_size = len(sim.people)
    adult_pop = cv.true(sim.people.age >= age_lb)
    adult_pop_size = adult_pop.shape[0]
    num_adults_to_vaccinate = int(adult_pop_size*vax_coverage)
    
    # Select phase1a_pop_size agents within the age-bracket specified  
    adults_to_vaccinate = np.random.choice(sim.people.uid[adult_pop], num_adults_to_vaccinate, replace=False)
    adult_vals = np.zeros(len(sim.people))
    adult_vals[adults_to_vaccinate] = 1.0

    subtarget_dict = dict(inds = sim.people.uid,
                          vals = adult_vals)
    return subtarget_dict


def get_vax_vals_and_ids(sim, input_args, vax_data, group_idx):
    """
    Creates dictionaries to be read by 
    
    cv.simple_vaccine()
    or by
    cv.vaccinate() 
    
    sim: the simulator object
    input_args: input arguments given by argparse()
    vax_data: a data frame pre-filtered by date of interest
    group_idx: an integer to index the info of a specific age_group 
    
    The ages specified in "age_lb" and "age_ub" are 
    included in this subpopulation, that is, 
    this people's ages are in the closed interval 
    [age_lb, age_ub]. Age-groups are non-overlapping.

    """
    # The total size of this model's population

    sim_pop_size = len(sim.people)
    this_age_group_is = cv.true((sim.people.age >= vax_data["age_lb"][group_idx]) & 
                                (sim.people.age <= vax_data["age_lb"][group_idx]))
    # number of people in this age group, for this specific population
    num_people_group = len(this_age_group_is)
    # There are three options or assumptions we can make here:
    # (1): we assume 'worst case scenario' -- asumes vax_coverage is given by second_dose_perc]
    # (2): we assume 'intermediate case scenario' -- assumes vax_coverage is between first and second dose percentages
    # (3): we assume 'best case scenario' -- assumes vaccine coverage is the percentage reflected by first_dose_perc
    vax_coverage = input_args.vax_age_coverage
    if vax_coverage == "worst":
        proportion_to_be_vaccinated = vax_data["second_dose_perc"][group_idx]
    elif vax_coverage == "intermediate":
        proportion_to_be_vaccinated = (vax_data["first_dose_perc"][group_idx] + vax_data["second_dose_perc"][group_idx]) / 2.0
    elif vax_coverage == "best":
        proportion_to_be_vaccinated = vax_data["first_dose_perc"][group_idx]

    num_agents_to_vaccinate = int(np.floor((proportion_to_be_vaccinated/100) * num_people_group))
    # Select X number of agents within this age group, without replacement
    this_age_group_selected = np.random.choice(sim.people.uid[this_age_group_is], num_agents_to_vaccinate, replace=False)
    this_age_group_agents = np.zeros((sim_pop_size, ), dtype=bool)
    this_age_group_agents[this_age_group_selected] = True

    # Create dictionary with all the parameters needed for this age group
    age_group_label = str(vax_data["age_group"][group_idx])
    this_age_group_dict = {age_group_label: { 'start_date': input_args.start_simulation_date,
                                              'inds':this_age_group_selected,
                                              'vals': 1.0}}
     
    return this_age_group_dict


def get_age_targeted_subtargets(sim, input_args):
    # Create dictionary needed by simple_vaccine().
    # This function needs the data from the AIR. 
    # Need the date to load the most current info.
    # 
    # Not sure i will need later
    sim_pop_size = len(sim.people)

    vax_data_df = pd.read_csv(input_args.vax_file)
    vax_data_current = vax_data_df[vax_data_df["date"] == input_args.vax_current_date]
    vax_data_current = vax_data_current.reset_index()
    number_of_age_groups = len(vax_data_current["age_lb"][:-1])

    # Read the file - we need to read everything but the last
    # age group which is the whole population
    subtarget_dict = dict()
    for this_group in range(number_of_age_groups):
        subtarget_dict = sc.mergedicts(subtarget_dict, get_vax_vals_and_ids(sim, input_args, vax_data_current, this_group)) 

    return subtarget_dict


def set_up_testing_and_tracing(sim, input_args):
  ntpts = sim.day(input_args.end_simulation_date)-sim.day(input_args.start_simulation_date)

  if input_args.tt_strategy == 'apply-tt': 
      # Testing
      sim.pars['interventions'].append(cv.test_num(daily_tests=[input_args.num_tests]*ntpts, 
                                                   start_day=sim.day(input_args.start_simulation_date), 
                                                   end_day=sim.day(input_args.end_simulation_date)+5, 
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

      trace_time = {'H': 1, 'S': 1, 
                    'W': 1, 'C': 7, 
                    'church': 3, 
                    'pSport': 1, 
                    'cSport': 1, 
                    'entertainment': 3,
                    'cafe_restaurant': 2, 
                    'pub_bar': 2, 
                    'transport': 7, 
                    'public_parks': 14,  
                    'large_events': 14,
                    'social': 1}

      sim.pars['interventions'].append(cv.contact_tracing(trace_probs=trace_probs,  
                                                          trace_time=trace_time, 
                                                          start_day=0, do_plot=False))
  return sim


def set_up_vaccination_rollout(sim, input_args):

    # Convert to match/case when we get to python 3.10   
    # Define if we apply the vaccines or not
    vax_directive = input_args.vaccinate 
    if vax_directive == "apply":
        vax_age_strategy = input_args.vax_age_coverage
        if vax_age_strategy == "none":
            if input_args.vax_efficacy != 0.0:
                # Apply only if efficacy is nonzero
                sim.initialize() 
                vaccine_subtarget = get_mass_vaccination_subtargets(sim, input_args.vax_coverage, input_args.age_lb)
                vaccine = cv.simple_vaccine(days=0, rel_sus=1.0-input_args.vax_efficacy, subtarget=vaccine_subtarget)
                sim.pars['interventions'].append(vaccine)
            else:
                return sim
        else:
            if input_args.vax_efficacy != 0.0:
                sim.initialize()
                vaccine_subtargets = get_age_targeted_subtargets(sim, input_args)
                for age_group in vaccine_subtargets.keys():
                    vaccine = cv.simple_vaccine(days=0, rel_sus=1.0-input_args.vax_efficacy, subtarget=vaccine_subtargets[age_group])
                    sim.pars['interventions'].append(vaccine)
    elif vax_directive == "donot-apply":
        pass

    return sim

def make_sim(input_args=None, betasfile=None, popfile=None):
    start_day = input_args.start_simulation_date
    iq_factor = input_args.iq_factor

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

    pars = {'pop_size': 200000,   # Population size
            'pop_infected': input_args.cluster_size,    # Original population infedcted
            'pop_scale': 25.5,    # Population scales to 5.1M ppl in QLD
            'rescale': True,      # Population dynamics rescaling
            'rand_seed': 42,      # Random seed to use
            'beta': input_args.global_beta, # UK variant TODO: use default covasim values 
            'dynam_layer': pd.Series([0.0,     0.0,    0.0,    1.0,   0.0,    0.0,    0.0,      1.0,    1.0,    1.0,    1.0,     1.0,        1.0,    0.0], index=all_layers).to_dict(),
                                      # H        S       W       C   church   psport  csport    ent     cafe    pub     trans    park        event    soc
            'contacts':    pd.Series([4.0,    21.0,    5.0,    1.0,   20.0,   40.0,    30.0,    25.0,   19.00,  30.00,   25.00,   10.00,     50.00,   6.0], index=all_layers).to_dict(),
            'beta_layer':  pd.Series([1.0,     0.3,    0.2,    0.1,    0.04,   0.2,     0.1,     0.01,   0.04,   0.06,    0.16,    0.03,      0.01,   0.3], index=all_layers).to_dict(),
            'iso_factor':  pd.Series([0.2, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor], index=all_layers).to_dict(),
            'quar_factor': pd.Series([0.2, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor], index=all_layers).to_dict(),
            'n_imports': 0, # Number of new cases per day -- can be varied over time as part of the interventions
            'start_day': input_args.start_simulation_date,
            'end_day':   input_args.end_simulation_date,
            'use_waning': False,
            'verbose': False}

    sim = cv.Sim(pars=pars, 
                 popfile=popfile,
                 load_pop=True, 
                 analyzers = cva.LayerInfections(layers = all_layers, label = 'lay_infs'))

    # Layer-specific betas    
    beta_ints = set_up_layer_beta_changes(betasfile, all_layers) 
    #beta_ints = set_layer_betas(betasfile, all_layers, input_args)            
    sim.pars['interventions'].extend(beta_ints)

    # Define if we want testing and tracing
    sim = set_up_testing_and_tracing(sim, input_args)

    # Define if we want and how we do vaccination
    sim = set_up_vaccination_rollout(sim, input_args)

    #sim.pars['interventions'].append(cvi.TargetInitialInfections(age_lb = args.age_lb))

    sim.initialize()
    return sim


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


def detect_when(data, use_nan=False):
    """
    Get the index of when something happened
    """
    idx = np.argmax(data, axis=0)  
    return idx 

def generate_layer_infection_df(sims, layers):

    df_list = []

    for layer in layers:
        data_list = []
        for idx, sim in enumerate(sims):
            layer_infs = sim.get_analyzer('lay_infs')
            layer_infs = layer_infs.to_df()[layer]
            layer_infs.rename(f'Sim {idx}', inplace = True)
            data_list.append(layer_infs)

        data_df = pd.concat(data_list, axis = 1)
        df_list.append(data_df)
    
    output = dict(zip(layers, df_list))
    return output

def generate_results_filename(input_args):
    # Plot all sims together 
    res_filename = f"qld_{input_args.label}_{input_args.tt_strategy}_"
    res_filename += f"{input_args.cluster_size:04d}_"
    res_filename += f"iqf_{input_args.iq_factor:.{2}f}_"
    res_filename += f"vxstr_{input_args.vax_age_coverage}_"
    if input_args.vax_age_coverage == "none":
        res_filename += f"vxcov_{input_args.vax_coverage:.{2}f}_"
    else:
        # Give the date of data used here
        res_filename += f"vx-age-dist_{input_args.vax_current_date}_"
    # Assumed efficacy    
    res_filename += f"vxeff_{input_args.vax_efficacy:.{2}f}_"   
    res_filename += f"age-lb_{input_args.age_lb}"
    return res_filename

if __name__ == '__main__':
        
    T = sc.tic()

    # Load argparse
    args = parser.parse_args()

    # Inputs
    inputsfolder = '../input_data'
    betasfile = f'{inputsfolder}/{args.layer_betas_file}'
    popfile = f'{inputsfolder}/qldppl-abs2020-200k.pop'


    # Results paths
    resultsfolder = args.results_path
    # simulation data path
    summary_folder = f'{resultsfolder}/sim-data-vax/summary-stats'
    reff_folder = f'{resultsfolder}/sim-data-vax/r_effs'
    # figures data path
    figfolder = f'{resultsfolder}/figures-vax'

    pathlib.Path(summary_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(reff_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(figfolder).mkdir(parents=True, exist_ok=True)

    # Get name for results files
    res_filename = generate_results_filename(args)

    # Create instance of simulator
    sim  = make_sim(input_args=args,
                    betasfile=betasfile,
                    popfile=popfile)

    # Do the stuff & save results
    msim = cv.MultiSim(base_sim=sim, par_args={'ncpus': args.ncpus})
    sc.heading('Run simulations with simplified vaccine rollout')
    msim.run(n_runs=args.nruns, reseed=True)
    #msim.save(f"{simfolder}/{res_filename}.obj")

    df_compare = msim.compare(output=True)
    df_stats = pd.concat([df_compare.mean(axis=1), df_compare.median(axis=1), df_compare.std(axis=1)], axis=1, 
                         keys=['mean', 'median', 'std'])
     
    new_df_stats = df_stats.append(pd.Series(data={'mean':args.iq_factor, 'median':args.iq_factor,'std':args.iq_factor}, name='iq_factor'))
    new_df_stats = new_df_stats.append(pd.Series(data={'mean':args.cluster_size, 'median':args.cluster_size,'std':args.cluster_size}, name='cluster_size'))
    new_df_stats = new_df_stats.append(pd.Series(data={'mean':args.vax_coverage, 'median':args.vax_coverage,'std':args.vax_coverage}, name='vax_coverage'))
    new_df_stats = new_df_stats.append(pd.Series(data={'mean':args.vax_efficacy, 'median':args.vax_efficacy,'std':args.vax_efficacy}, name='vax_efficacy'))
    
    ################################# ADDITIONAL STATS #############################
    #                                                                              #
    ################################################################################
    
    # Get the maximum number of new infections in a given day
    data = get_individual_traces('new_infections', msim.sims)
    # Get the date of the maximum of the wave
    idx = detect_when(data, use_nan=True)
    # Append to dataframe
    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(idx), 'median':np.nanmedian(idx),'std':np.nanstd(idx)}, name='day_max_new_infections'))
    
    r_effs = get_individual_traces('r_eff', msim.sims)
    r_eff_df = pd.DataFrame(r_effs, columns = [f'Sim {n}' for n in range(args.nruns)], index = np.arange(msim.sims[0].t + 1))
    r_eff_df.to_csv(f"{reff_folder}/{res_filename}_r_effTS.csv")

    reff07 = r_effs[6, :]
    reff15 = r_effs[14, :]
    reff30 = r_effs[29, :]
    reff60 = r_effs[59, :]
    reff75 = r_effs[74, :]
    reff90 = r_effs[89, :]

    #all_layers = ['H', 'S', 'W', 'C', 
    #              'church', 
    #              'pSport', 
    #              'cSport', 
    #              'entertainment', 
    #              'cafe_restaurant', 
    #              'pub_bar', 
    #              'transport', 
    #              'public_parks', 
    #              'large_events', 
    #              'social']

    #layer_df_dict = generate_layer_infection_df(msim.sims, all_layers)

    #for layer, df in layer_df_dict.items():
        #df.to_csv(f"{simfolder}/{res_filename}_{layer}_infectionTS.csv")

    #dfs = []

    #for sim in msim.sims:

        #seed_infs = sim.get_analyzer('rsi')
        #dfs.append(seed_infs.to_df())

    #seed_inf_df = pd.concat(dfs, axis = 1).T
    #seed_inf_df.to_csv(f"{simfolder}/{res_filename}_seed_infs.csv")

    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff07), 
                                                       'median':np.nanmedian(reff07),
                                                       'std':np.nanstd(reff07)}, name='r_eff_07'))

    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff15), 
                                                       'median':np.nanmedian(reff15),
                                                       'std':np.nanstd(reff15)}, name='r_eff_15'))
    
    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff30), 
                                                       'median':np.nanmedian(reff30),
                                                       'std':np.nanstd(reff30)}, name='r_eff_30'))

    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff60), 
                                                       'median':np.nanmedian(reff60),
                                                       'std':np.nanstd(reff60)}, name='r_eff_60'))

    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff75), 
                                                       'median':np.nanmedian(reff75),
                                                       'std':np.nanstd(reff75)}, name='r_eff_75'))

    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff90), 
                                                       'median':np.nanmedian(reff90),
                                                       'std':np.nanstd(reff90)}, name='r_eff_90'))
    
    # Save results to csv
    new_df_stats.transpose().to_csv(f"{summary_folder}/{res_filename}.csv")
    msim.reduce()
    to_plot = sc.objdict({
        'Total infections': ['cum_infections'],
        'New infections per day': ['new_infections'],
        'Total deaths': ['cum_deaths']
    })
    msim_fig = msim.plot(to_plot=to_plot, do_save=False, do_show=False, legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=35)
    msim_fig.savefig(f"{figfolder}/{res_filename}.png", dpi=100)

    sc.toc(T)
