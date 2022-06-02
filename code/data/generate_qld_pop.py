import covasim as cv
import covasim_australia.contacts as co
import covasim_australia.data as data
import covasim_australia.parameters as parameters
import covasim_australia.utils as utils
import sciris as sc
import numpy as np
# Create a .pop file for delta simulation with 200k agents - based on 200k file
# and Paula's code

ClusterSize_Vector = np.linspace(14,20,7) # Create a file for each input
# ClusterSize = 10
# ClusterSize_Vector = 10
for ii in range(np.size(ClusterSize_Vector)):
    ClusterSize = ClusterSize_Vector[ii]
    
    seed=None
    pop_size=int(2e5)
    pop_scale=5.1e6/pop_size
    pop_infected=0
    savepeople=True
    Variant = 'b16172'
    Variant = 'b117'
    popfile = 'inputs/qldppl-abs2020-200k-' + Variant + '-ClusterSize' + str(int(ClusterSize)) + '.pop'
    
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
        
            
     # Define list with imports 
    if Variant == 'b16172':
        imported_infections = [cv.variant('b16172', days=22, n_imports=ClusterSize,rescale = False)]
    else:
        imported_infections = [cv.variant('b117', days=22, n_imports=ClusterSize,rescale = False)]
    
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
    
    # utils.set_rand_seed({'seed': seed})
    # params.pars['rand_seed'] = seed
    people, popdict = co.make_people(params)
    #import pdb; pdb.set_trace()
    if savepeople: 
        sc.saveobj(popfile, people)
        
    # return people, popdict
    
    