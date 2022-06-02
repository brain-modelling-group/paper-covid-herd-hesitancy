#!/usr/bin/env python
# coding: utf-8
"""
Case 1b: 
Use brand vaccine to get herd immunity values for the adult Queensland population.
This model also enables the comparison of results with and without rescaling. 
This model captures first and second doses delivery.

Authors:
Sebastian J. Raison
Paula Sanz-Leon
Lachlan H. Hamilton

QIMR Berghofer Medical Research Institute 2022
"""

# Import scientific python
import pandas as pd
import numpy as np
import pathlib as pathlib

# Import IDM/Optima code
import covasim as cv
import covasim.misc as cvm
import sciris as sc

import covasim_australia as cv_aus

# Add argument parser
import argparse


# Check covasim version is the one we actually need
cv.check_version('>=3.0.7', die=True)

parser = argparse.ArgumentParser()


parser.add_argument('--ncpus', default=8, 
                               type=int, 
                               help='''Maximum number of cpus used by MultiSim runs.''')

parser.add_argument('--nruns', default=2, 
                               type=int, 
                               help='''Number of simulations to run per scenario. 
                                       Uses different PRNG seeds for each simulations.''')

parser.add_argument('--results_path', default='results', 
                              type=str, 
                              help='''The relative and/or absolute path to the results folder, without the trailing /''')

parser.add_argument('--label', default='pfizer-vax', 
                               type=str, 
                               help=''' An ID string.''')

parser.add_argument('--tt_strategy', 
                              default='apply-tt', 
                              type=str, 
                              help=''' A string to determine what set of testing and tracing strategy to use. Available: 'apply-tt' or 'no-tt.''')

parser.add_argument('--global_beta', 
                              default=0.0113*2.4, 
                              type=float,
                              help='''Global transmissibility from calibrated model using ancestral strain, multiplied by 2.4 for the delta strain''') 

parser.add_argument('--num_tests', 
                              default=7501, 
                              type=int, 
                              help='''Average number of tests per day. Value based on average over period 2021-02-15 to 2021-05-11. Std dev: 6325''')


parser.add_argument('--age_lb', 
                              default=16, 
                              type=int, 
                              help='''Age lower boundary to be within the fraction of the population that is vaccinated.''')


parser.add_argument('--iq_factor', default=0.01,
                                   type=float, 
                                   help=''' Isolation factor. [0: absolute adherence]; [>1: no adherence]''')

parser.add_argument('--cluster_size', 
                              default=20, 
                              type=int, 
                              help='''The number of infected people entering QLD community on a given date (default, 2020-10-01)''')

parser.add_argument('--start_simulation_date', default='2021-08-30', 
                              type=str, 
                              help='''The date at which to start simulation (eg, 2020-12-31).''')

parser.add_argument('--end_simulation_date', default='2021-12-04', 
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

parser.add_argument('--vax_current_date', default='2021-08-17', 
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

parser.add_argument('--vax_coverage_full', 
                              default=0.3, 
                              type=float, 
                              help='''A float between 0 and 1. [0: no one is vaccinated] [1: all agents aged age_lb+].''')


parser.add_argument('--vax_coverage_partial', 
                              default=0.5, 
                              type=float, 
                              help='''A float between 0 and 1. [0: no one is vaccinated] [1: all agents aged age_lb+].''')

parser.add_argument('--vax_interval', 
                              default=1000, 
                              type=int, 
                              help='''Number of days between first and second dose of pfizer. default: default interval value..''')

parser.add_argument('--vax_efficacy', 
                              default=1.0, 
                              type=float, 
                              help='''[0: no efficacy] [1: 100% efficacy] - only refers to tranmission blocking ''')


parser.add_argument('--pop_scale', 
                              default=1.0, 
                              type=float, 
                              help='''Population scale factor 25.5 is based on a 200k agent simulation, so as to achieve 5.1M people in QLD''')


parser.add_argument('--rescale', 
                              dest    = 'rescale', 
                              type = int,
                              default = 0,
                              help    = '''0 or 1, indexing the list [False, True] disable/enable (dynamic) rescaling''')

################################################################################################################
###########################                FUNCTIONS TO SET UP SCENARIOS                 #######################
################################################################################################################

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


def get_mass_vaccination_subtargets(sim, vax_coverage_full, vax_coverage_partial, age_lb):
    """
    Create dictionary needed by vaccinate_prob().
    
    
    The vaccine coverage is expressed in vax_coverage_full and vax_coverage_partial
    as a value between [0, 1], though their sum should never be > 1.

    Returns:
    subtarget_dict

    """

    sim_pop_size = len(sim.people)
    eligible_pop = cv.true(sim.people.age >= age_lb)
    eligible_pop_size = eligible_pop.shape[0]


    if (vax_coverage_full + vax_coverage_partial) > 1.0:
       print("This is not a valid combination. Stopping program. ")
       quit()

    num_ppl_to_vaccinate = int(eligible_pop_size * (vax_coverage_full + vax_coverage_partial))
    num_ppl_to_fully_vaccinate = int(eligible_pop_size * (vax_coverage_full))
    
    # Select all agents that will get vaccinated   
    ppl_to_vaccinate = np.random.choice(sim.people.uid[eligible_pop], num_ppl_to_vaccinate, replace=False)

    # Select group that will be vacccinated with two doses
    ppl_to_fully_vaccinate = ppl_to_vaccinate[:num_ppl_to_fully_vaccinate]

    # Select group that will be vaccinated only once
    ppl_to_partially_vaccinate = ppl_to_vaccinate[num_ppl_to_fully_vaccinate:]

    full_vax_prob = np.zeros(len(sim.people))
    partial_vax_prob = np.zeros(len(sim.people))
    full_vax_prob[ppl_to_fully_vaccinate] = 1.0
    partial_vax_prob[ppl_to_partially_vaccinate] = 1.0

    subtarget_dict = { "full_vax_subtarget": dict(inds = sim.people.uid, vals = full_vax_prob),
                       "partial_vax_subtarget":  dict(inds = sim.people.uid, vals = partial_vax_prob)}
    return subtarget_dict


def get_mass_vaccination_subtargets_first_second(sim, vax_coverage_full, vax_coverage_partial, age_lb):
    """
    Create dictionary needed by vaccinate_prob().
    
    
    The vaccine coverage is expressed in vax_coverage_full and vax_coverage_partial
    as a value between [0, 1]. Their sum can be up to 2. For instance:
    if vax_coverage_partial is 0.7 and vax_coverage_full 0.35, it means
    half of the agents that got one doses, get a second dose (ie, 50% of the people to be vaccinated in this case). 
    vax_coverage_partial sets the maximum number of people getting vaccinated

    Returns:
    subtarget_dict

    """

    sim_pop_size = len(sim.people)
    eligible_pop = cv.true(sim.people.age >= age_lb)
    eligible_pop_size = eligible_pop.shape[0]


    if vax_coverage_full > vax_coverage_partial:
       print("This is not a valid combination. Stopping program. ")
       quit()

    num_ppl_to_vaccinate = int(eligible_pop_size * vax_coverage_partial)

    num_ppl_to_fully_vaccinate = int(eligible_pop_size * vax_coverage_full)
    
    # Select all agents that will get vaccinated   
    ppl_to_vaccinate = np.random.choice(sim.people.uid[eligible_pop], num_ppl_to_vaccinate, replace=False)

    # Select group that will be vacccinated with two doses
    ppl_to_fully_vaccinate = ppl_to_vaccinate[:num_ppl_to_fully_vaccinate]

    # Select group that will be vaccinated only once
    ppl_to_partially_vaccinate = ppl_to_vaccinate[num_ppl_to_fully_vaccinate:]
    # People who have gotten at least one dose is pppl_to_fully_vaccinated + ppl_to_partially_vaccinate

    full_vax_prob = np.zeros(len(sim.people))
    partial_vax_prob = np.zeros(len(sim.people))
    full_vax_prob[ppl_to_fully_vaccinate] = 1.0
    partial_vax_prob[ppl_to_partially_vaccinate] = 1.0

    subtarget_dict = { "full_vax_subtarget": dict(inds = sim.people.uid, vals = full_vax_prob),
                       "partial_vax_subtarget":  dict(inds = sim.people.uid, vals = partial_vax_prob)}
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

  
def set_up_testing_and_tracing(sim, input_args):
  ntpts = sim.day(input_args.end_simulation_date)-sim.day(input_args.start_simulation_date)

# Testing - conisder changin the number of testing

  if (input_args.pop_scale == 1.0) and (input_args.rescale == 0):
    num_tests = input_args.num_tests / 25.5 # "rescale" down num tests
  else:
    num_tests = input_args.num_tests 

  sim.pars['interventions'].append(cv.test_num(daily_tests=[num_tests]*ntpts, 
                                               start_day=sim.day(input_args.start_simulation_date), 
                                               end_day=sim.day(input_args.end_simulation_date)+5,
                                               symp_test=100.0,
                                               quar_test=100.0, 
                                               quar_policy="both", 
                                               test_delay=0))

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


def set_up_brand_vaccination_rollout(sim, input_args):

    # Convert to match/case when we get to python 3.10   
    # Define if we apply the vaccines or not
    vax_directive = input_args.vaccinate 
    if vax_directive == "apply":
        vax_age_strategy = input_args.vax_age_coverage
        if vax_age_strategy == "none":
                # Apply only if efficacy is nonzero
                sim.initialize() 
                # Get people who are going to be 
                vaccine_subtargets  = get_mass_vaccination_subtargets_first_second(sim, input_args.vax_coverage_full, input_args.vax_coverage_partial, input_args.age_lb)
                vaccine_full = cv_aus.vaccinate_prob_base(vaccine='pfizer', 
                                                          prob=0.0, days=input_args.start_simulation_date, 
                                                          subtarget=vaccine_subtargets["full_vax_subtarget"])
                #vaccine_full.p['interval'] =  1
                vaccine_partial = cv_aus.vaccinate_prob_base(vaccine='pfizer', prob=0.0, days=input_args.start_simulation_date, subtarget=vaccine_subtargets["partial_vax_subtarget"])
                vaccine_partial.p["interval"] = input_args.vax_interval
                sim.pars['interventions'].append(vaccine_full)
                sim.pars['interventions'].append(vaccine_partial)
        else:
                return sim
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
    rescale = [False, True]
    pars = {'pop_size': 200000,               # Agent population size
            'pop_infected': 0,                # Original population infected
            'pop_scale': input_args.pop_scale, # Population scales to 5.1M ppl in QLD
            'rescale': rescale[input_args.rescale],    # Population dynamics rescaling
            'rand_seed': 42,                  # Random seed to use
            'beta': input_args.global_beta,   # UK variant TODO: use default covasim values 
            'dynam_layer': pd.Series([0.0,     0.0,    0.0,    1.0,   0.0,    0.0,    0.0,      1.0,    1.0,    1.0,    1.0,     1.0,        1.0,    0.0], index=all_layers).to_dict(),
                                      # H        S       W       C   church   psport  csport    ent     cafe    pub     trans    park        event    soc
            'contacts':    pd.Series([4.0,    21.0,    5.0,    1.0,   20.0,   40.0,    30.0,    25.0,   19.00,  30.00,   25.00,   10.00,     50.00,   6.0], index=all_layers).to_dict(),
            'beta_layer':  pd.Series([1.0,     0.3,    0.2,    0.1,    0.04,   0.2,     0.1,     0.01,   0.04,   0.06,    0.16,    0.03,      0.01,   0.3], index=all_layers).to_dict(),
            'iso_factor':  pd.Series([0.2, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor], index=all_layers).to_dict(),
            'quar_factor': pd.Series([0.2, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor, iq_factor], index=all_layers).to_dict(),
            'n_imports': 0, # Number of new cases per day -- can be varied over time as part of the interventions
            'start_day': input_args.start_simulation_date,
            'end_day':   input_args.end_simulation_date,
            #'nab_decay': dict(form='nab_growth_decay', growth_time=1, decay_rate1=np.log(2)/10000, decay_time1=365, decay_rate2=np.log(2)/3650, decay_time2=700),
            'nab_decay': dict(form='nab_growth_decay', growth_time=22, decay_rate1=np.log(2)/100, decay_time1=250, decay_rate2=np.log(2)/3650, decay_time2=365),
            'use_waning': True,
            'verbose': False}

    sim = cv.Sim(pars=pars, 
                 popfile=popfile,
                 load_pop=True,
                 analyzers=[cv_aus.store_eird(label='eird'), 
                            cv.age_histogram(days=[0, 14, 29, 44, 49, 59],
                                             states = ['infectious', 'severe', 'critical', 'recovered', 'dead', 'vaccinated'],
                                             label="age_histogram"),
                            cv.daily_age_stats(states = ['infectious', 'severe','critical', 'recovered', 'dead', 'vaccinated'],
                                               edges = np.arange(0, 101, 4),
                                               label='daily_age_stats'),
                            cv_aus.daily_age_stats_intersect(edges = np.arange(0, 101, 4),
                                                             label='daily_age_stats_vax_intersect')],
                 label="pfizer-vax")

    # Layer-specific betas    
    beta_ints = set_up_layer_beta_changes(betasfile, all_layers)             
    sim.pars['interventions'].extend(beta_ints)

    # Insert cluster 21 days after start of the simulation 
    sim.pars['interventions'].append(cv_aus.SeedInfection({sim.day(input_args.start_simulation_date)+43: input_args.cluster_size}))

    # Define if we want testing and tracing
    sim = set_up_testing_and_tracing(sim, input_args)

    # Define if we want and how we do vaccination
    sim = set_up_brand_vaccination_rollout(sim, input_args)

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


def generate_results_filename(input_args):
    # Plot all sims together 
    res_filename = f"qld_{input_args.label}_"
    res_filename += f"{input_args.cluster_size:04d}_"
    res_filename += f"iqf_{input_args.iq_factor:.{2}f}_"
    if input_args.vax_age_coverage == "none":
        res_filename += f"vxfull_{input_args.vax_coverage_full:.{2}f}_"
        res_filename += f"vxpart_{input_args.vax_coverage_partial:.{2}f}"
    else:
        # Give the date of data used here
        res_filename += f"vx-age-dist_{input_args.vax_current_date}_"
    # Assumed efficacy    
    #res_filename += f"vxeff_{input_args.vax_efficacy:.{2}f}"   
    return res_filename


def msim_dfs_to_df(msim):
    """
    Function specific to this simulation to calculate dataframes with 
    the average and standard deviation across nruns, of 
    cumulative statistics grouped by age.
    """

    # cumulative values of certain states (infected, critical, )
    df1 = msim.sims[0].get_analyzer("daily_age_stats").to_total_df()

    # cumulative values of certain states (infected, critical) & vaccinated state
    df2 = msim.sims[0].get_analyzer("daily_age_stats_vax_intersect").to_total_df()

    # Allocate variables
    df1_add  = pd.DataFrame(index=df1.index, columns=df1.columns).transpose()
    df1_mean = pd.DataFrame(index=df1.index, columns=df1.columns).transpose()
    df1_std  = pd.DataFrame(index=df1.index, columns=df1.columns).transpose()
    
    df2_add  = pd.DataFrame(index=df2.index, columns=df2.columns).transpose()
    df2_mean = pd.DataFrame(index=df2.index, columns=df2.columns).transpose()
    df2_std  = pd.DataFrame(index=df2.index, columns=df2.columns).transpose()

    cols1 = list(df1.columns.values)
    cols1 = cols1[1:]
    cols2 = list(df2.columns.values)
    cols2 = cols2[1:]


    for this_sim in msim.sims:
        ana1 = this_sim.get_analyzer("daily_age_stats")
        ana2 = this_sim.get_analyzer("daily_age_stats_vax_intersect") # gives cumulative 
        df1 =  ana1.to_total_df().transpose()
        df2 =  ana2.to_total_df().transpose()
        df1_add = pd.concat((df1_add, df1))
        df2_add = pd.concat((df2_add, df2))
        
    for col in cols1:
        df1_mean.loc[col] = df1_add.loc[col].mean()
        df1_std.loc[col]  = df1_add.loc[col].std()

    for col in cols2:
        df2_mean.loc[col] = df2_add.loc[col].mean()
        df2_std.loc[col]  = df2_add.loc[col].std()


    # Restore values
    for col in  ["age"]:
        df1_mean.loc[col] = df1.loc[col]
        df1_std.loc[col]  = df1.loc[col]
        df2_mean.loc[col] = df2.loc[col]
        df2_std.loc[col] = df2.loc[col]
    
    df1_mean = df1_mean.transpose()
    df1_std  = df1_std.transpose()

    df2_mean = df2_mean.transpose()
    df2_std  = df2_std.transpose()

    return pd.concat((df1_mean, df2_mean.iloc[:, 2:]), axis=1), pd.concat((df1_std, df2_std.iloc[:, 2:]),  axis=1)


if __name__ == '__main__':
        
    T = sc.tic()

    # Load argparse
    args = parser.parse_args()

    # Inputs
    inputsfolder = '../input_data/'
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
                    betasfile=betasfile)

    # Do the stuff & save results
    msim = cv.MultiSim(base_sim=sim, par_args={'ncpus': args.ncpus})
    sc.heading('Run simulations with Pfizer vaccine')
    msim.run(n_runs=args.nruns, reseed=True)
    #msim.save(f"{simfolder}/{res_filename}.obj")

    df_compare = msim.compare(output=True)
    df_stats = pd.concat([df_compare.mean(axis=1), df_compare.median(axis=1), df_compare.std(axis=1)], axis=1, 
                         keys=['mean', 'median', 'std'])
     
    new_df_stats = df_stats.append(pd.Series(data={'mean':args.iq_factor, 'median':args.iq_factor,'std':args.iq_factor}, name='iq_factor'))
    new_df_stats = new_df_stats.append(pd.Series(data={'mean':args.cluster_size, 'median':args.cluster_size,'std':args.cluster_size}, name='cluster_size'))
    new_df_stats = new_df_stats.append(pd.Series(data={'mean':args.vax_coverage_full, 'median':args.vax_coverage_full,'std':args.vax_coverage_full}, name='vax_coverage_full'))
    new_df_stats = new_df_stats.append(pd.Series(data={'mean':args.vax_coverage_partial, 'median':args.vax_coverage_partial,'std':args.vax_coverage_partial}, name='vax_coverage_partial'))
    
    ################################# ADDITIONAL STATS #############################
    #                                                                              #
    ################################################################################
    # Get the maximum number of new infections in a given day
    data = get_individual_traces('new_infections', msim.sims)
    # Get the date of the maximum of the wave
    idx = detect_when(data, use_nan=True)
    # Append to dataframe
    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(idx), 'median':np.nanmedian(idx),'std':np.nanstd(idx)}, name='day_max_new_infections'))
    
    data = get_individual_traces('r_eff', msim.sims)
    r_eff_df = pd.DataFrame(data, columns = [f'Sim {n}' for n in range(args.nruns)], index = np.arange(msim.sims[0].t + 1))
    r_eff_df.to_csv(f"{reff_folder}/{res_filename}_r_effTS.csv")

    reff07 = data[6, :]
    reff15 = data[14, :]
    reff30 = data[29, :]
    reff50 = data[49, :]
    reff60 = data[59, :]
    reff75 = data[73, :]
    reff90 = data[89, :]


    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff07), 
                                                       'median':np.nanmedian(reff07),
                                                       'std':np.nanstd(reff07)}, name='r_eff_07'))

    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff15), 
                                                       'median':np.nanmedian(reff15),
                                                       'std':np.nanstd(reff15)}, name='r_eff_15'))
    
    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff30), 
                                                       'median':np.nanmedian(reff30),
                                                       'std':np.nanstd(reff30)}, name='r_eff_30'))

    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff50), 
                                                       'median':np.nanmedian(reff50),
                                                       'std':np.nanstd(reff50)}, name='r_eff_50'))

    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff60), 
                                                       'median':np.nanmedian(reff60),
                                                       'std':np.nanstd(reff60)}, name='r_eff_60'))

    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff75), 
                                                       'median':np.nanmedian(reff75),
                                                       'std':np.nanstd(reff75)}, name='r_eff_74'))

    new_df_stats = new_df_stats.append(pd.Series(data={'mean':np.nanmean(reff90), 
                                                       'median':np.nanmedian(reff90),
                                                       'std':np.nanstd(reff90)}, name='r_eff_90'))
    
    # Save results to csv
    new_df_stats.transpose().to_csv(f"{summary_folder}/{res_filename}.csv")

    # Save exemplar
    age_hist_novax = msim.sims[1].get_analyzer("daily_age_stats")
    age_hist_vax   = msim.sims[1].get_analyzer("daily_age_stats_vax_intersect") # gives cumulative timeseries

    # make figs
    sim_fig1 = age_hist_novax.plot(total=True, do_show=False)
    sim_fig2 = age_hist_vax.plot(total=True,  do_show=False)
    
    # save figs
    sim_fig1.savefig(f"{figfolder}/{res_filename}-age_hist_total.png", dpi=100)
    sim_fig2.savefig(f"{figfolder}/{res_filename}-age_hist_total_intersect_vax.png", dpi=100)
    
    df3, df4 = msim_dfs_to_df(msim)
    
    # # insert paramters of interst to group by at a later stage
    # df3.insert(df3.shape[1], 'cluster_size', [args.cluster_size]*df3.shape[0])
    # df3.insert(df3.shape[1], 'vax_coverage_full', [args.vax_coverage_full]*df3.shape[0])
    # df3.insert(df3.shape[1], 'vax_coverage_partial', [args.vax_coverage_partial]*df3.shape[0])

    # # insert parameters of interst to group by at a later stage
    # df4.insert(df4.shape[1], 'cluster_size', [args.cluster_size]*df4.shape[0])
    # df4.insert(df4.shape[1], 'vax_coverage_full', [args.vax_coverage_full]*df4.shape[0])
    # df4.insert(df4.shape[1], 'vax_coverage_partial', [args.vax_coverage_partial]*df4.shape[0])

    # df3.to_csv(f"{simfolder}/{res_filename}-age_hist_results_mean.csv")
    # df4.to_csv(f"{simfolder}/{res_filename}-age_hist_results_std.csv")

    msim.reduce()
    to_plot = sc.objdict({
        'Total infections': ['cum_infections'],
        'New infections per day': ['new_infections'],
        'Total critical': ['cum_critical'],
        'New critical cases per day': ['new_critical'],
        'Total deaths': ['cum_deaths'],
        'New deaths per day': ['new_deaths'],
        'Total doses administered': ['cum_vaccinations'],
        'New doses administered': ['new_vaccinations'],
        'Total vaccinated agents': ['cum_vaccinated'],
        'New vaccinated agents': ['new_vaccinated'],
    })
    msim_fig = msim.plot(to_plot=to_plot, do_save=False, do_show=False, legend_args={'loc': 'upper left'}, axis_args={'hspace': 0.4}, interval=35)
    msim_fig.savefig(f"{figfolder}/{res_filename}.png", dpi=100)

    sc.toc(T)