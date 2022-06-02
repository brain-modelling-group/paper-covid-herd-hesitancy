import math
import numpy as np
import pandas as pd
import sciris as sc
import covasim_australia.utils as utils
import warnings
from collections import defaultdict

def _get_layers(locations, databook, all_lkeys):

    all_layers = {}

    # read in all the layer sheets
    for key in all_lkeys:
        default_layer = databook.parse(f'layer-{key}', header=0, index_col=0)
        default_layer = default_layer.loc[locations].to_dict(orient='index')
        all_layers[key] = default_layer

    return all_layers


def _get_pars(locations, databook, all_lkeys):
    """
    Read in & format the parameters for each location.
    :return a dictionary of form (location, pars)
    """

    all_layers = _get_layers(locations, databook, all_lkeys)

    # the parameters that are in a different sheet
    other_pars = databook.parse('other_par', index_col=0)
    other_pars = other_pars.loc[locations].to_dict(orient='index')

    all_pars = {}
    for location in locations:
        pars = {}
        o = other_pars[location]
        for pkey in utils.par_keys():
            if pkey == 'n_days':
                ndays = utils.get_ndays(o['start_day'], o['end_day'])
                temp = {pkey: ndays}
            elif o.get(pkey) is not None:
                temp = {pkey: o[pkey]}
            else:
                # must be layer-dependent parameter
                temp = {}
                temp[pkey] = {}
                for lkey in all_lkeys:
                    l_pars = all_layers[lkey]
                    temp[pkey].update({lkey: l_pars[location][pkey]})
            pars.update(temp)
        all_pars[location] = pars

    return all_pars


def _get_extrapars(locations, databook, all_lkeys):

    all_layers = _get_layers(locations, databook, all_lkeys)

    # those in other_par sheet
    other_pars = databook.parse('other_par', index_col=0)
    other_pars = other_pars.loc[locations].to_dict(orient='index')

    all_extrapars = {}
    for location in locations:
        extrapars = {}
        o = other_pars[location]
        for ekey in utils.extrapar_keys():
            if o.get(ekey) is not None:
                temp = {ekey: o[ekey]}
            else:
                # must be layer-dependent parameter
                temp = {}
                temp[ekey] = {}
                for lkey in all_lkeys:
                    l_pars = all_layers[lkey]
                    temp[ekey].update({lkey: l_pars[location][ekey]})
            extrapars.update(temp)
        all_extrapars[location] = extrapars
    return all_extrapars


def get_dispersion_parameter(value):
    if pd.isna(value):
        return None, None
    elif value.startswith('random_'):
        cluster_type, dispersion = value.split('_')
        dispersion = float(dispersion)
    else:
        cluster_type = value
        dispersion = None
    return cluster_type, dispersion


def _get_layerchars(locations, databook, all_lkeys):

    all_layers = _get_layers(locations, databook, all_lkeys)
    all_layerchars = {}

    for location in locations:
        layerchars = defaultdict(dict) # {characteristic:{layer:value}}

        for ckey in utils.layerchar_keys():
            for lkey in all_lkeys:
                value = all_layers[lkey][location][ckey]

                if ckey == 'cluster_type':
                    cluster_type, dispersion = get_dispersion_parameter(value)
                    layerchars[ckey][lkey] = cluster_type
                    layerchars['dispersion'][lkey] = dispersion
                else:
                    layerchars[ckey][lkey] = value
        all_layerchars[location] = dict(layerchars)

    return all_layerchars


def read_policies(locations, databook, all_lkeys):
    """
    Read in the policies sheet
    :param databook:
    :return:
    """

    start_days = databook.parse('other_par', index_col=0, header=0).to_dict(orient='index')
    pol_sheet = databook.parse('policies', index_col=[0,1], header=0)  # index by first 2 cols to avoid NAs in first col
    trace_sheet = databook.parse('tracing_policies', index_col=[0,1], header=0)

    all_policies = {}
    for location in locations:
        start_sim = start_days[location]['start_day']
        pols = pol_sheet.loc[location]
        trace = trace_sheet.loc[location]

        policies = {}
        policies['beta_policies'] = {}
        policies['import_policies'] = {}
        policies['clip_policies'] = {}
        policies['policy_dates'] = {}
        policies['tracing_policies'] = {}

        for pol_name, row in pols.iterrows():
            # get the number of days til policy starts and ends (relative to simulation start)
            start_pol = row['date_implemented']
            end_pol = row['date_ended']
            if not pd.isna(start_pol):
                days_to_start = utils.get_ndays(start_sim, start_pol)
                n_days = [days_to_start]
                if not pd.isna(end_pol):
                    days_to_end = utils.get_ndays(start_sim, end_pol)
                    n_days.append(days_to_end)
                policies['policy_dates'][pol_name] = n_days

            # check if there is a change in beta values on this layer (i.e. change in transmission risk)
            layers_in_use = all_lkeys + ['beta']
            beta_vals = row.loc[row.index.intersection(layers_in_use)]
            beta_change = beta_vals.prod()  # multiply series together
            if not math.isclose(beta_change, 1, abs_tol=1e-9):
                policies['beta_policies'][pol_name] = {}
                beta = row['beta']
                for layer_key in all_lkeys:
                    beta_layer = row[layer_key]
                    policies['beta_policies'][pol_name][layer_key] = beta * beta_layer

            # policies impacting imported cases
            imports = row['imported_infections']
            if imports > 0:
                policies['import_policies'][pol_name] = {'n_imports': imports}

            # policies that remove people from layers
            to_clip = [row['clip_edges_layer'], row['clip_edges']]
            percent_to_clip = to_clip[1]
            if not pd.isna(percent_to_clip):
                policies['clip_policies'][pol_name] = {}
                policies['clip_policies'][pol_name]['change'] = percent_to_clip
                layers_to_clip = to_clip[0]
                if not pd.isna(layers_to_clip):
                    policies['clip_policies'][pol_name]['layers'] = [lk for lk in all_lkeys if lk in layers_to_clip]
                else:
                    policies['clip_policies'][pol_name]['layers'] = all_lkeys

        # tracing policies
        for pol_name, row in trace.iterrows():

            # dates
            start_pol = row['date_implemented']
            end_pol = row['date_ended']
            if not pd.isna(start_pol):
                days_to_start = utils.get_ndays(start_sim, start_pol)
                n_days = [days_to_start]
                days_to_end = None
                if not pd.isna(end_pol):
                    days_to_end = utils.get_ndays(start_sim, end_pol)
                    n_days.append(days_to_end)
                policies['policy_dates'][pol_name] = n_days

                # only add this if has a start date
                layers = row['layers']
                layers = layers.replace(' ', '').split(',')  # list of layer strings
                cov = row['coverage'].replace(' ', '')
                cov = [float(x) for x in cov.split(',')]
                days = row['days_changed'].replace(' ', '')
                days = [int(x) for x in days.split(',')]
                trace_time = int(row['trace_time'])

                policies['tracing_policies'][pol_name] = {'layers': layers,
                                                          'coverage': cov,
                                                          'days': days,
                                                          'trace_time': trace_time,
                                                          'start_day': days_to_start,
                                                          'end_day': days_to_end}

        all_policies[location] = policies
    return all_policies


# def read_sex(databook):
#     sex = databook.parse('age_sex')
#     sex['frac_male'] = sex['Male'] / (sex['Male'] + sex['Female'])
#     sex['frac_male'] = sex['frac_male'].fillna(0.5)  # if 0, replace with 0.5


def read_popdata(locations, databook):
    agedist_sheet = databook.parse('age_sex', index_col=[0,1], usecols="A:R")  # ignore totals column
    household_sheet = databook.parse('households', index_col=[0])

    all_agedist = {}
    all_householddist = {}
    for location in locations:

        # age distribution
        # total number of men & women in each age bracket
        totals = agedist_sheet.loc[location, 'Male'] + agedist_sheet.loc[location, 'Female']
        # break up bracket into individual years, distribute numbers uniformly
        age_dist = {}
        for age_group in totals.index:
            age_total = totals[age_group]
            age_l = int(age_group.split('-')[0])
            age_u = int(age_group.split('-')[1])
            age_interval = np.arange(age_l, age_u+1)
            to_distrib = int(age_total / len(age_interval))
            temp = {age: to_distrib for age in age_interval}
            age_dist.update(temp)

        all_agedist[location] = pd.Series(age_dist)

        # household distribution
        household_dist = household_sheet.loc[location]
        household_dist.index = [1, 2, 3, 4, 5, 6]  # used as the number of people per household

        all_householddist[location] = household_dist

    return all_agedist, all_householddist


def read_imported_cases(locations, epidata, pars, default_val=0):
    """Read in the number of imported cases as a time series.
    If not in the epidata, create one from the default_val"""
    imported_cases = {}
    if 'imported_cases' in epidata.columns:
        for location in locations:
            i_cases = epidata.loc[location]['imported_cases'].to_numpy()
            imported_cases[location] = i_cases

    else:
        # use default value
        print(f'Unable to locate imported_cases in epi data, replacing with {default_val}')
        for location in locations:
            n_days = pars[location]['n_days']
            i_cases = np.full(shape=n_days, fill_value=default_val)
            imported_cases[location] = i_cases

    return imported_cases


def format_daily_tests(location, tests, default_val):

    # not including the start values, find percenatage of nans
    ind_notnan = np.where(np.isnan(tests) == False)[0]
    first_notnan = ind_notnan[0]
    num_nan = np.sum(np.isnan(tests[first_notnan:]))
    percent_nan = num_nan / len(tests[first_notnan:])
    if percent_nan > .2:  # arbitrarily 20%
        print(f'Warning: "num_tests" column has {percent_nan}% nan values. Switching to av_daily_tests')
        tests_copy = np.full(shape=len(tests), fill_value=default_val)

    else:
        # either the value is nan, in which case we fill with previous non-nan,
        # or the value is a number, in which case store for use to fill later
        replace_val = 0
        tests_copy = sc.dcp(tests)
        for i, val in np.ndenumerate(tests):
            if np.isnan(val):
                tests_copy[i] = replace_val
            else:
                replace_val = val

        if np.isnan(tests_copy).any():
            warnings.warn(f'"new_tests" column appears to be all nan for {location}')
    return tests_copy


def extrapolate_tests(tests, future_tests, start_day, n_days, calibration_date):
    if calibration_date is None:
        last_day_data = tests['date'][-1]
        test_data = tests['new_tests'].copy().to_numpy()
        days_with_data = utils.get_ndays(start_day, last_day_data)
        days_without_data = n_days - days_with_data
        tests = np.append(test_data, [future_tests] * days_without_data)
    else:
        days_for_calibration = utils.get_ndays(start_day, calibration_date)
        days_after_calibration = n_days - days_for_calibration 
        test_data = tests.loc[tests['date'] <= calibration_date].copy()  # subset tests
        test_data = test_data['new_tests'].to_numpy()
        tests = np.append(test_data, [future_tests] * days_after_calibration)

    return tests


def read_daily_tests(locations, epidata, default_val=0):
    """Read in the number of tests performed daily.
    If not in the epidata, create one from the default_val"""
    if 'new_tests' in epidata.columns:
        for location in locations:
            tests = epidata.loc[location]['new_tests'].to_numpy()
            tests = format_daily_tests(location, tests, default_val)
            # update epidata so it is formatted correctly
            epidata.at[location, 'new_tests'] = tests  # pandas is weird
            # extrapolate after last entry (not included in epidata df)
            # tests = extrapolate_tests(tests, extrapars[location]['future_daily_tests'], pars[location]['n_days'],
            #                           pars[location]['start_day'], calibration_end)
            # daily_tests[location] = tests
    else:
        print(f'Unable to locate new_tests in epi data, replacing with {default_val}')
        n_rows = len(epidata.index)
        epidata['new_tests'] = [None] * n_rows
        for location in locations:
            tests = np.full(shape=n_rows, fill_value=default_val)
            epidata.at[location, 'new_tests'] = tests
            # tests = extrapolate_tests(tests, default_val, pars[location]['n_days'])
            # daily_tests[location] = tests

    return epidata


def read_epi_data(where, index_col='location', **kwargs):
    """By default, will return daily global data from URL below"""
    if where == 'url':
        url = utils.epi_data_url()
        epidata = pd.read_csv(url, index_col=index_col, parse_dates=['date'], **kwargs)
    else:
        epidata = pd.read_csv(where, index_col=index_col, parse_dates=['date'], **kwargs)
    return epidata


def format_epidata(locations, epidata, extrapars):
    """Convert the dataframe to a dictionary of dataframes, where the key is the location"""
    # rename the columns
    epidata = epidata.rename(columns=utils.colnames())
    to_keep = ['date', 'new_diagnoses', 'cum_diagnoses', 'cum_deaths', 'new_deaths', 'new_tests', 'cum_tests','n_severe']
    epidata_dict = {}
    for l in locations:
        this_country = epidata.loc[l]
        this_country = this_country.reset_index(drop=True)  # reset row numbers
        this_country = this_country.reindex(to_keep,axis=1)  # drop unwanted columns, add NaN columns for missing variables
        # scale the cumulative infections by undiagnosed
        undiagnosed = extrapars[l]['undiag']
        #this_country['cum_infections'] = this_country['cum_infections'] * (1 + undiagnosed)

        # cumulative tests
        this_country['cum_tests'] = this_country['new_tests'].cumsum()

        epidata_dict[l] = this_country
    return epidata_dict


def subset_epidata(locations, epidata, calibration_end):
    calbration_epidata = {}
    for location in locations:
        epidata_thisloc = epidata[location]
        calibartion_thisloc = calibration_end[location]
        if calibartion_thisloc is None:
            epidata_subset = epidata_thisloc.copy()
        else:
            epidata_subset = epidata_thisloc.loc[epidata_thisloc['date'] <= calibartion_thisloc].copy()

        calbration_epidata[location] = epidata_subset

    return calbration_epidata


def get_daily_tests(locations, epidata, pars, extrapars, calibration_end):
    daily_tests = {}
    for location in locations:
        calibration_date = calibration_end[location]
        tests = epidata.loc[location, ['date', 'new_tests']].copy()
        tests = extrapolate_tests(tests,
                                  extrapars[location]['future_daily_tests'],
                                  pars[location]['start_day'],
                                  pars[location]['n_days'],
                                  calibration_date)
        daily_tests[location] = tests

    return daily_tests


def get_epi_data(locations, where, pars, extrapars, calibration_end, **kwargs):
    epidata = read_epi_data(where, **kwargs)
    imported_cases = read_imported_cases(locations, epidata, pars)
    epidata = read_daily_tests(locations, epidata)
    daily_tests = get_daily_tests(locations, epidata, pars, extrapars, calibration_end)

    # subset epidata
    complete_epidata = format_epidata(locations, epidata, extrapars)
    calibration_epidata = subset_epidata(locations, complete_epidata, calibration_end)

    return complete_epidata, calibration_epidata, imported_cases, daily_tests


def read_contact_matrix(locations, databook):
    """
    Load Prem et al. matrices then transform into a symmetric matrix
    :param databook:
    :return:
    """

    matrix_sheet = databook.parse('contact matrices-home', usecols="A:W", index_col=[0,1])

    all_matrices = {}

    for location in locations:
        mixing_matrix0 = matrix_sheet.loc[location]
        contact_matrix = {}
        # make symmetric with ((rowi, colj) + (rowj, coli)) / 2
        mixing_matrix = mixing_matrix0.copy()
        #import pdb; pdb.set_trace()
        for i in range(len(mixing_matrix0)):
            for j in range(len(mixing_matrix0)):
                mixing_matrix.values[i, j] = (mixing_matrix0.values[i, j] + mixing_matrix0.values[j, i]) / 2.0
        age_lb = [int(x.split('-')[0]) for x in mixing_matrix.index]  # lower age in bin
        age_ub = [int(x.split('-')[1]) for x in mixing_matrix.index]  # upper age in bin

        contact_matrix['matrix'] = mixing_matrix
        contact_matrix['age_lb'] = age_lb
        contact_matrix['age_ub'] = age_ub

        all_matrices[location] = contact_matrix

    return all_matrices


def load_databook(db_path):
    databook = pd.ExcelFile(db_path)
    return databook


def read_params(locations, db, all_lkeys):
    """
    Read all the necessary parameters from the databook
    """
    if all_lkeys is None:
        all_lkeys = utils.get_all_lkeys()

    pars = _get_pars(locations, db, all_lkeys)
    extrapars = _get_extrapars(locations, db, all_lkeys)
    layerchars = _get_layerchars(locations, db, all_lkeys)
    return pars, extrapars, layerchars


def read_data(locations=None, db_name=None, epi_name=None, all_lkeys=None, dynamic_lkeys=None, calibration_end=None):
    """Reads in all data in the appropriate format"""
    db_path, epi_path = utils.get_file_paths(db_name=db_name, epi_name=epi_name)

    calibration_end = utils.clean_calibration_end(locations, calibration_end)

    db = load_databook(db_path)

    # handle layer names
    all_lkeys, default_lkeys, dynamic_lkeys, custom_lkeys = utils.get_lkeys(all_lkeys, dynamic_lkeys)

    pars, extrapars, layerchars = read_params(locations, db, all_lkeys)
    policies = read_policies(locations, db, all_lkeys)
    contact_matrix = read_contact_matrix(locations, db)
    if epi_path is not None:
        complete_epidata, calibration_epidata, imported_cases, daily_tests = get_epi_data(locations, epi_path, pars, extrapars, calibration_end)
    else:
        complete_epidata = defaultdict(lambda: None)
        calibration_epidata = defaultdict(lambda: None)
        imported_cases = defaultdict(lambda: None)
        daily_tests = defaultdict(lambda: None)
    age_dist, household_dist = read_popdata(locations, db)

    # convert so that outer key is the location
    all_data = {}
    for location in locations:
        all_data[location] = {'pars': pars[location],
                              'extrapars': extrapars[location],
                              'layerchars': layerchars[location],
                              'policies': policies[location],
                              'contact_matrix': contact_matrix[location],
                              'complete_epidata': complete_epidata[location],
                              'calibration_epidata': calibration_epidata[location],
                              'age_dist': age_dist[location],
                              'household_dist': household_dist[location],
                              'imported_cases': imported_cases[location],
                              'daily_tests': daily_tests[location],
                              'all_lkeys': all_lkeys,
                              'default_lkeys': default_lkeys,
                              'dynamic_lkeys': dynamic_lkeys,
                              'custom_lkeys': custom_lkeys}

    # don't return a dict if you're just looking at one location
    if len(locations)==1:   return all_data[locations[0]]
    else:                   return all_data
