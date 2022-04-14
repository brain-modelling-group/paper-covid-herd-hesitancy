    # Defining the hesitancy for the simulation: 
    # Each hesitancy measurement (Hesitancy) is a vector of numbers that 
    # represents the Australian hesitancy per age group. We have two values in 
    # time, and from this create a hesitancy variable (H_Pow) to represent 
    # hypothetical levels of age-specific hesitancy.     
    
    # We set the most recent hesitancy measurement to H_Pow = 0 
    # (Janurary 2021), then H_Pow = -1 to the last measurement (August 2020).
    # To change by 1, the transformation is Hesitancy_1 = Hesitancy_0*dH, where
    # dH is fraction of age-specific hesitancy between Janurary and August 
    # (i.e. dH = Hesitancy_Janurary/Hesitancy_August). 
    
    # This value represents an empirically relevant and possible change in 
    # age-specific hesitancy that could occur if public opinion changed. 
    # A change towards pro-vaccination is represented by dH^(-H_Pow), where 
    # H_Pow is a positive number. A change towards more hesitancy is dH^(H_Pow)
    
    # The bounds we have chosen for H_Pow are [-1.85,1.85], because 
    # H_Pow < -1.85 gives age-specific hesitancy > 100%.  
    
    # This script will run multiple repititions of one hesitancy value, and 
    # output a csv file. 
    
def qld_clustersize_fracvac_postvaccine_cluster(job_ID,N_jobs,NCPUS):
        
    import numpy as np
   
    from qld_clustersize_fracvac_postvaccine_function import qld_clustersize_fracvac_postvaccine_function
    
     # Upper bounds for age groups - i.e. a category is between:
    # AgeGroups[ii-1] < Age <= AgeGroups[ii]
    # 0 and 200 are the upper bounds - for the ease of indexing
    
    AgeGroups = [17,24,34,44,54,64,200]    
    
    # Getting the number of people in each age group
    
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
    
    AgeGroupSize = []
    AgeGroupSize.append(round(qld_pop['15-19']/5*2)+qld_pop['20-24']) #18-24
    AgeGroupSize.append(qld_pop['25-29']+qld_pop['30-34']) #25-34
    AgeGroupSize.append(qld_pop['35-39']+qld_pop['40-44']) #35-44
    AgeGroupSize.append(qld_pop['45-49']+qld_pop['50-54']) #45-54
    AgeGroupSize.append(qld_pop['55-59']+qld_pop['60-64']) #55-64
    AgeGroupSize.append(qld_pop['65-69']+qld_pop['70-74'] + qld_pop['75-79']+qld_pop['80-84'] + qld_pop['85-89']+qld_pop['90-94'] + qld_pop['95-99']+qld_pop['100-104']) #65+

    
    
    # Hesitancy values from Edwards et al., 2021
    Hesitancy_Aug20 = (0.092, 0.171,0.117,0.163, 0.104,0.08)
    Hesitancy_Jan21 = (0.145, 0.298,0.247,0.270, 0.121,0.094)
    
    # Fraction of age-specific change from August to Janurary   
    dH = np.divide(Hesitancy_Aug20,Hesitancy_Jan21)
   

    H_Pow_lower_bound = -1.85 # Limits to hestiancy powers
    H_Pow_upper_bound = 1.85
    
    # Create a new value for each job submitted to the cluster
    H_Pow_Vector = np.linspace(H_Pow_lower_bound,H_Pow_upper_bound,N_jobs)   
    H_Pow = round(H_Pow_Vector[job_ID],3) # Pick the hesitancy used by this job. 
        
    Hesitancy = dH**(H_Pow)*Hesitancy_Jan21
    
    # Convert age-specific hesitancy to the expected fraction of unvaccinated
    # population.
    
    PopSize = sum(AgeGroupSize)
    FracUnvaccinated = np.dot(AgeGroupSize,Hesitancy)/PopSize
    Hesitancy = np.ones(np.shape(Hesitancy))
    Hesitancy = FracUnvaccinated*Hesitancy
    
    
    
    print(Hesitancy)
    print(H_Pow)
    
    N_sims = 100 # Number of sims to run for each input
    tt = 1 # testing and tracing, 1 for yes, 0 for no
    
    ClusterSizes = np.linspace(1,20,20) # Sizes to use for each run
    
    for ClusterSize in ClusterSizes:     
      
        qld_clustersize_fracvac_postvaccine_function(Hesitancy,AgeGroups,N_sims,H_Pow,ClusterSize,tt,NCPUS)    
    
    
