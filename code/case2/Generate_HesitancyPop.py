import sys
sys.path.append('../') # Add path to covasim_australia

import covasim as cv
import covasim_australia.contacts as co
import covasim_australia.data as data
import covasim_australia.parameters as parameters
import covasim_australia.utils as utils
import sciris as sc
import numpy as np
# Create a .pop file for delta simulation with 200k agents - based on 200k file
# and Paula's code

Cluster_Size = 20 # Number of people to infect
    
seed=None
pop_size=int(2e5)
pop_scale=5.1e6/pop_size
pop_infected=0
savepeople=True

"""
Generate  popdict() and People() for Queensland population
"""
location = 'QLD'
# This file has a lot of information to build the networks and layers of the model

db_name  = 'input_data_Australia'

all_lkeys = ['H', 'S', 'W', 'C', 
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

dynamic_lkeys = ['C', 'entertainment', 'cafe_restaurant', 'pub_bar',
                    'transport', 'public_parks', 'large_events'] 

# Sweep throgh each variant
Variant_Names = ['alpha','delta','omicron']
for Variant_Name in Variant_Names:       

    # Get parameters for each strain
    if Variant_Name == 'alpha':
        imported_infections = cv.variant('alpha', days= 21, n_imports=Cluster_Size, rescale=False)

    elif Variant_Name == 'delta':
    
        imported_infections = cv.variant('delta', days=21, n_imports=Cluster_Size, rescale=False)
   
    elif Variant_Name == 'omicron': # From covasim 3.1.3 unfinished version parameters

        omicron = dict(
                    rel_beta        = 3.0, # Estimated to be 1.25-1.6-fold more transmissible than B117: https://www.researchsquare.com/article/rs-637724/v1
                    rel_symp_prob   = 1.0,
                    rel_severe_prob = 0.8, # 2x more transmissible than alpha from https://mobile.twitter.com/dgurdasani1/status/1403293582279294983
                    rel_crit_prob   = 0.5,
                    rel_death_prob  = 1.0,
                )

        imported_infections = cv.variant(omicron, days=21, n_imports=Cluster_Size, rescale=False,label = 'omicron')    

    user_pars = {'pop_size': pop_size,
                    'pop_scale': pop_scale,
                    'rescale': False,              
                    'variants': imported_infections,
                    'n_variants': 2,
                    'calibration_end': None,
                    'pop_infected': 0
                } # Pass in a minimal set of sim pars

    # return data relevant to each specified location in "locations"
    all_data = data.read_data(locations=[location],
                                db_name=db_name,
                                epi_name=None,
                                all_lkeys=all_lkeys,
                                dynamic_lkeys=dynamic_lkeys,
                                calibration_end={'QLD':'2022-12-31'})


    loc_data = all_data

    # setup parameters object for this simulation
    params = parameters.setup_params(location=location,
                                        loc_data=loc_data,
                                        sim_pars=user_pars)

    people, popdict = co.make_people(params)

    popfile = 'qldppl-abs2020-' + str(int(pop_size)) + '-' + Variant_Name + '-ClusterSize' + str(int(Cluster_Size))+'.pop'

    if savepeople: 
        sc.saveobj(popfile, people)
        
# return people, popdict
    
    