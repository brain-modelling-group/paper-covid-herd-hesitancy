import covasim as cv
import numpy as np
import os
import sciris as sc
# import utils, policy_updates
import covasim.utils as cvu
import covasim.defaults as cvd
import covasim.base as cvb
import covasim.misc as cvm
import networkx as nx
import pandas as pd

def get_ndays(start_day, end_day):
    """Calculates the number of days for simulation"""
    # get start and end date
    start_day = sc.readdate(str(start_day))
    end_day = sc.readdate(str(end_day))
    n_days = (end_day - start_day).days
    return n_days


def epi_data_url():
    """Contains URL of global epi data for COVID-19"""
    url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
    return url


def colnames():
    """Get the desired column names for the epi data"""
    names = {'date': 'date',
             'new_cases': 'new_diagnoses',
             'new_deaths': 'new_deaths',
             'new_tests': 'new_tests',
             'total_tests': 'cum_tests',
             'total_deaths': 'cum_deaths',
             'cum_infections': 'cum_diagnoses',
             'total_cases': 'cum_diagnoses',
             'hospitalised': 'n_severe'}  # either total cases or cum_infections in epi book
    return names


def _format_paths(db_name, epi_name, root):
    # databook
    databook_path = os.path.join(root, 'data', db_name)
    if 'xlsx' not in db_name:
        databook_path += '.xlsx'

    if epi_name == 'url':
        epidata_path = 'url'
    elif epi_name is None:
        epidata_path = None
    else:
        # must be stored elsewhere
        epidata_path = os.path.join(root, 'data', epi_name)
        if 'csv' not in epi_name:
            epidata_path += '.csv'

    return databook_path, epidata_path


def get_file_paths(db_name, epi_name, root=None):
    if root is None:
        from covasim_australia import datadir
        root = str(datadir.parent)

    db_path, epi_path = _format_paths(db_name=db_name,
                                      epi_name=epi_name,
                                      root=root)

    return db_path, epi_path


def set_rand_seed(metapars):
    if metapars.get('seed') is None:
        seed = 1
    else:
        seed = metapars['seed']
    np.random.seed(seed)
    return


def par_keys():
    keys = ['contacts', 'beta_layer', 'quar_factor',
            'pop_size', 'pop_scale', 'rescale',
            'rescale_threshold', 'rescale_factor',
            'pop_infected', 'start_day',
            'n_days', 'iso_factor']
    return keys


def metapar_keys():
    keys = ['n_runs', 'noise']
    return keys


def extrapar_keys():
    keys = ['trace_probs', 'trace_time', 'restart_imports',
            'restart_imports_length', 'relax_day', 'future_daily_tests',
            'undiag', 'av_daily_tests', 'symp_test', 'quar_test',
            'sensitivity', 'test_delay', 'loss_prob']
    return keys


def layerchar_keys():
    keys = ['proportion', 'age_lb', 'age_ub', 'cluster_type']
    return keys


def get_dynamic_lkeys(all_lkeys=None):
    """These are the layers that are re-generated at each time step since the contact networks are dynamic.
    Layers not in this list are treated as static contact networks"""
    defaults = get_default_lkeys(all_lkeys)
    if 'C' in defaults:
        layers = ['C']
    else:
        layers = []

    return layers


def get_default_lkeys(all_lkeys=None):
    """
    These are the standard layer keys: household (H), school (S), low risk work, high risk work, and community (C)
    :return:
    """
    defaults = ['H', 'S', 'W', 'C']
    if all_lkeys is None:
        layers = defaults
    else:
        layers = list(set(all_lkeys).intersection(set(defaults)))  # only what is in common
    return layers


def get_all_lkeys():
    layers = list(set(get_default_lkeys()) | set(get_dynamic_lkeys()))
    return layers


def get_custom_lkeys(all_lkeys=None):
    """Layer keys that are part of the simulation but not by default"""
    if all_lkeys is None:
        all_lkeys = get_all_lkeys()

    default_lkeys = set(get_default_lkeys(all_lkeys))
    custom_lkeys = [x for x in all_lkeys if x not in default_lkeys]  # Don't change the order, otherwise runs are not reproducible due to rng

    return custom_lkeys


def get_lkeys(all_lkeys, dynamic_lkeys):
    # check if they are user-specified, otherwise just use hard-coded keys
    if all_lkeys is None:
        all_lkeys = get_all_lkeys()
    if dynamic_lkeys is None:
        dynamic_lkeys = get_dynamic_lkeys(all_lkeys)

    default_lkeys = get_default_lkeys(all_lkeys)
    custom_lkeys = get_custom_lkeys(all_lkeys)

    return all_lkeys, default_lkeys, dynamic_lkeys, custom_lkeys


def clean_pars(user_pars, locations):
    par_keys = cv.make_pars().keys()
    calibration_end = {}
    new_user_pars = {}

    for location in locations:
        user_pars_oneloc = user_pars[location]
        if user_pars_oneloc.get('calibration_end') is not None:
            calibration_end[location] = user_pars_oneloc['calibration_end']
        else:
            calibration_end[location] = None

        new_user_pars[location] = {key: val for key, val in user_pars_oneloc.items() if key in par_keys}

    return new_user_pars, calibration_end


def clean_calibration_end(locations, calibration_end):
    if calibration_end is None:  # not specified for any country
        calibration_end_all_locs = {loc: None for loc in locations}
    else:
        calibration_end_all_locs = {}
        for location in locations:
            if calibration_end.get(location) is None:  # doesn't exist or set to None
                calibration_end_all_locs[location] = None
            else:
                calibration_end_all_locs[location] = calibration_end[location]

    return calibration_end_all_locs


def policy_plot2(scens, plot_ints=False, to_plot=None, do_save=None, fig_path=None, fig_args=None, plot_args=None,
                 axis_args=None, fill_args=None, legend_args=None, as_dates=True, dateformat=None, plot_base=False,
                 interval=None, n_cols=1, font_size=18, font_family=None, grid=True, commaticks=True,
                 do_show=True, sep_figs=False, verbose=1, y_lim=None):
    from matplotlib.ticker import StrMethodFormatter
    import sciris as sc
    import numpy as np
    import matplotlib.ticker as ticker
    import datetime as dt
    import pylab as pl
    import matplotlib as mpl
    from covasim import defaults as cvd

    '''
    Plot the results -- can supply arguments for both the figure and the plots.

    Args:
        scen        (covasim Scenario): Scenario with results to be plotted
        scen_name   (str):  Name of the scenario with intervention start dates to plot
        plot_ints   (Bool): Whether or not to plot intervention start dates
        to_plot     (dict): Dict of results to plot; see default_scen_plots for structure
        do_save     (bool): Whether or not to save the figure
        fig_path    (str):  Path to save the figure
        fig_args    (dict): Dictionary of kwargs to be passed to pl.figure()
        plot_args   (dict): Dictionary of kwargs to be passed to pl.plot()
        axis_args   (dict): Dictionary of kwargs to be passed to pl.subplots_adjust()
        fill_args   (dict): Dictionary of kwargs to be passed to pl.fill_between()
        legend_args (dict): Dictionary of kwargs to be passed to pl.legend()
        as_dates    (bool): Whether to plot the x-axis as dates or time points
        dateformat  (str):  Date string format, e.g. '%B %d'
        interval    (int):  Interval between tick marks
        n_cols      (int):  Number of columns of subpanels to use for subplot
        font_size   (int):  Size of the font
        font_family (str):  Font face
        grid        (bool): Whether or not to plot gridlines
        commaticks  (bool): Plot y-axis with commas rather than scientific notation
        do_show     (bool): Whether or not to show the figure
        sep_figs    (bool): Whether to show separate figures for different results instead of subplots
        verbose     (bool): Display a bit of extra information

    Returns:
        fig: Figure handle
    '''

    sc.printv('Plotting...', 1, verbose)

    if to_plot is None:
        to_plot = ['cum_deaths', 'new_infections', 'cum_infections']
    to_plot = sc.dcp(to_plot)  # In case it's supplied as a dict

    scens['verbose'] = True
    scen = scens['scenarios']
    epidata = scens['complete_epidata']
    calibration_end = scens['calibration_end']

    # Handle input arguments -- merge user input with defaults
    fig_args = sc.mergedicts({'figsize': (16, 14)}, fig_args)
    plot_args = sc.mergedicts({'lw': 3, 'alpha': 0.7}, plot_args)
    axis_args = sc.mergedicts(
        {'left': 0.15, 'bottom': 0.1, 'right': 0.95, 'top': 0.90, 'wspace': 0.25, 'hspace': 0.25}, axis_args)
    fill_args = sc.mergedicts({'alpha': 0.2}, fill_args)
    legend_args = sc.mergedicts({'loc': 'best'}, legend_args)

    if sep_figs:
        figs = []
    else:
        fig = pl.figure(**fig_args)
    pl.subplots_adjust(**axis_args)
    pl.rcParams['font.size'] = font_size
    if font_family:
        pl.rcParams['font.family'] = font_family

    n_rows = np.ceil(len(to_plot) / n_cols)  # Number of subplot rows to have
    baseline_days = []
    for rk, reskey in enumerate(to_plot):
        otherscen_days = []
        title = scen[next(iter(scen))].base_sim.results[reskey].name  # Get the name of this result from the base simulation
        if sep_figs:
            figs.append(pl.figure(**fig_args))
            ax = pl.subplot(111)
        else:
            if rk == 0:
                ax = pl.subplot(n_rows, n_cols, rk + 1)
            else:
                ax = pl.subplot(n_rows, n_cols, rk + 1, sharex=ax)

        resdata0 = scen[next(iter(scen))].results[reskey]
        if plot_base:
            resdata = {key: val for key, val in resdata0.items()}
        else:
            resdata = {key: val for key, val in resdata0.items() if key != 'baseline'}
        colors = sc.gridcolors(len(resdata.items()))
        scennum = 0
        for scenkey, scendata in resdata.items():

            pl.fill_between(scen[next(iter(scen))].tvec, scendata.low, scendata.high, **fill_args)
            pl.plot(scen[next(iter(scen))].tvec, scendata.best, label=scendata.name, c=colors[scennum], **plot_args)
            scennum += 1
            pl.title(title)
            if rk == 0:
                pl.legend(**legend_args)

            pl.grid(grid)
            if commaticks:
                sc.commaticks()

            epidata[next(iter(scen))]['validate'] = 0  # which data is for validation vs calibration
            for j in range(len(epidata[next(iter(scen))])):
                if (epidata[next(iter(scen))]['date'][j]) >= sc.readdate(calibration_end[next(iter(scen))][next(iter(scen))]):
                    epidata[next(iter(scen))].loc[j, 'validate'] = 1

            if scen[next(iter(scen))].base_sim.data is not None and reskey in scen[next(iter(scen))].base_sim.data:
                data_t = np.array((scen[next(iter(scen))].base_sim.data.index - scen[next(iter(scen))].base_sim['start_day']) / np.timedelta64(1, 'D'))
                # pl.plot(data_t, epidata.base_sim.data[reskey], 'sk', **plot_args)
                cmap, norm = mpl.colors.from_levels_and_colors(levels=[0, 1], colors=['black', 'black'], extend='max')
                pl.scatter(x=epidata[next(iter(scen))].index, y=epidata[next(iter(scen))][reskey], c=epidata[next(iter(scen))]['validate'],
                           edgecolor='none', marker='s', cmap=cmap, norm=norm, **plot_args)
                # pl.plot(epidata[next(iter(scen))].index, epidata[next(iter(scen))][reskey],
                #        sc.mergedicts({'c': epidata[next(iter(scen))]['validate'],'cmap': cmap, 'norm':norm}, plot_args))

            # Optionally reset tick marks (useful for e.g. plotting weeks/months)
            if interval:
                xmin, xmax = ax.get_xlim()
                ax.set_xticks(pl.arange(xmin, xmax + 1, interval))

            # Set xticks as dates
            if as_dates:
                @ticker.FuncFormatter
                def date_formatter(x, pos):
                    return (scen[next(iter(scen))].base_sim['start_day'] + dt.timedelta(days=x)).strftime('%b-%d')

                ax.xaxis.set_major_formatter(date_formatter)
                if not interval:
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Plot interventions
        day_projection_starts = utils.get_ndays(scen[next(iter(scen))].base_sim['start_day'],
                                                sc.readdate(calibration_end[next(iter(scen))][next(iter(scen))]))
        # pl.axvline(x=day_projection_starts, color='black', linestyle='--')
        scennum = 0
        if plot_ints:
            for scenkey, scendata in resdata.items():
                if scenkey.lower() != 'baseline':
                    for intervention in scen[next(iter(scen))].scenarios[scenkey]['pars']['interventions']:
                        if hasattr(intervention, 'days'):  # and isinstance(intervention, PolicySchedule):
                            otherscen_days = [day for day in intervention.days if day not in otherscen_days]
                        elif hasattr(intervention, 'start_day'):
                            if intervention.start_day != 0:
                                otherscen_days.append(intervention.start_day)
                        for day in otherscen_days:
                            # pl.axvline(x=day, color=colors[scennum], linestyle='--')
                            pl.axvline(x=day, color='grey', linestyle='--')
                        # intervention.plot(scen.sims[scen_name][0], ax)
                if scenkey.lower() == 'baseline':
                    if plot_base:
                        for intervention in scen[next(iter(scen))].scenarios['baseline']['pars']['interventions']:
                            if hasattr(intervention, 'days') and isinstance(intervention,
                                                                            policy_updates.PolicySchedule) and rk == 0:
                                baseline_days = [day for day in intervention.days if day not in baseline_days]
                            elif hasattr(intervention, 'start_day'):
                                if intervention.start_day not in baseline_days and intervention.start_day != 0:
                                    baseline_days.append(intervention.start_day)
                            for day in baseline_days:
                                # pl.axvline(x=day, color=colors[scennum], linestyle='--')
                                pl.axvline(x=day, color='grey', linestyle='--')
                        # intervention.plot(scen.sims[scen_name][0], ax)
                scennum += 1
        if y_lim:
            if reskey in y_lim.keys():
                ax.set_ylim((0, y_lim[reskey]))
                if y_lim[reskey] < 20:  # kind of arbitrary limit to add decimal places so that it doesn't just plot integer ticks on small ranges
                    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    # Ensure the figure actually renders or saves
    if do_save:
        if fig_path is None:  # No figpath provided - see whether do_save is a figpath
            fig_path = 'covasim_scenarios.png'  # Just give it a default name
        fig_path = sc.makefilepath(fig_path)  # Ensure it's valid, including creating the folder
        pl.savefig(fig_path)

    if do_show:
        pl.show()
    else:
        pl.close(fig)

    return fig

    #
    # # Plot interventions
    # scennum = 0
    # if plot_ints:
    #     for s, scen_name in enumerate(scen.sims):
    #         if scen_name.lower() != 'baseline':
    #             for intervention in scen.sims[scen_name][0]['interventions']:
    #                 if hasattr(intervention, 'days') and isinstance(intervention, PolicySchedule):
    #                     otherscen_days = [day for day in intervention.days if day not in baseline_days and day not in otherscen_days]
    #                 elif hasattr(intervention, 'start_day'):
    #                     if intervention.start_day not in baseline_days and intervention.start_day not in otherscen_days and intervention.start_day != 0:
    #                         otherscen_days.append(intervention.start_day)
    #                     if intervention.end_day not in baseline_days and intervention.end_day not in otherscen_days and isinstance(intervention.end_day, int) and intervention.end_day < scen.sims[scen_name][0]['n_days']:
    #                         otherscen_days.append(intervention.end_day)
    #                 for day in otherscen_days:
    #                     #pl.axvline(x=day, color=colors[scennum], linestyle='--')
    #                     pl.axvline(x=day, color='grey', linestyle='--')
    #                 #intervention.plot(scen.sims[scen_name][0], ax)
    #         else:
    #             for intervention in scen.sims[scen_name][0]['interventions']:
    #                 if hasattr(intervention, 'days') and isinstance(intervention, PolicySchedule) and rk == 0:
    #                     baseline_days = [day for day in intervention.days if day not in baseline_days]
    #                 elif hasattr(intervention, 'start_day'):
    #                     if intervention.start_day not in baseline_days and intervention.start_day != 0:
    #                         baseline_days.append(intervention.start_day)
    #                 for day in baseline_days:
    #                     #pl.axvline(x=day, color=colors[scennum], linestyle='--')
    #                     pl.axvline(x=day, color='grey', linestyle='--')
    #                 #intervention.plot(scen.sims[scen_name][0], ax)
    #         scennum += 1


class SeedInfection(cv.Intervention):
    """
    Seed a fixed number of infections

    This class facilities seeding a fixed number of infections on a per-day
    basis.

    Infections will only be seeded on specified days

    """

    def __init__(self, infections: dict):
        """

        Args:
            infections: Dictionary with {day_index:n_infections}

        """
        super().__init__()
        self.infections = infections  #: Dictionary mapping {day: n_infections}. Day can be an int, or a string date like '20200701'

    def initialize(self, sim):
        super().initialize(sim)
        self.infections = {sim.day(k):v for k,v in self.infections.items()}  # Convert any day strings to ints

    def apply(self, sim):
        if sim.t in self.infections:
            susceptible_inds = cvu.true(sim.people.susceptible)

            if len(susceptible_inds) < self.infections[sim.t]:
                raise Exception('Insufficient people available to infect')

            targets = cvu.choose(len(susceptible_inds), self.infections[sim.t])
            target_inds = susceptible_inds[targets]
            sim.people.infect(inds=target_inds)


def generate_seed_infection_dict(sim_start_date, interv_start_date, interv_end_date, **kwargs):
    """
    Returns a dictionary to be passed to SeedInfection
    Args:
        sim_start_date (str in the following date format 'YYYY-MM-DD' ): the starting date of the simulation
        interv_start_date (str in the following date format 'YYYY-MM-DD'): start date of the seed infection intervention
        interv_end_date ((str in the following date format 'YYYY-MM-DD'): end date of the seed infection intervention)
        
    
    Returns:
        A dictionary of with number of infections for every day between start and end internvention dates
    """

    start_date_idx = cvm.day(interv_start_date, start_day=sim_start_date)
    end_date_idx = cvm.day(interv_end_date, start_day=sim_start_date)
    num_days = end_date_idx-start_date_idx
    
    seeded_infections = cvu.sample(size=num_days, **kwargs)
    seed_infections_dict = {start_date_idx+day_idx: num_infections for (day_idx, num_infections) in zip(range(num_days+1), seeded_infections)}
   
    return seed_infections_dict


class DynamicTrigger(cv.Intervention):
    """
    Execute callback during simulation execution
    """
    def __init__(self, condition, action, once_only=False):
        """
        Args:
            condition: A function `condition(sim)` function that returns True or False
            action: A function `action(sim)` that runs if the condition was true
            once_only: If True, the action will only execute once
        """
        super().__init__()
        self.condition = condition #: Function that
        self.action = action
        self.once_only = once_only
        self._ran = False

    def apply(self, sim):
        """
        Check condition and execute callback
        """
        if not (self.once_only and self._ran) and self.condition(sim):
            self.action(sim)
            self._ran = True


class test_prob_with_quarantine(cv.test_prob):
    """
    Testing based on probability with quarantine during tests
    """

    def __init__(self, *args, swab_delay, test_isolation_compliance, leaving_quar_prob,**kwargs):
        super().__init__(*args, **kwargs)
        self.swab_delay = swab_delay
        self.test_isolation_compliance = test_isolation_compliance  #: Compliance level for individuals in general population isolating after testing. People already in quarantine are assumed to be compliant
        self.leaving_quar_prob = leaving_quar_prob  # Probability of testing for people leaving quarantine e.g. set to 1 to ensure people test before leaving quarantine

    def apply(self, sim):
        ''' Perform testing '''
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # TEST LOGIC
        # 1. People who become symptomatic in the general community will wait `swab_delay` days before getting tested, at rate `symp_prob`
        # 2. People who become symptomatic while in quarantine will test immediately at rate `symp_quar_test`
        # 3. People who are symptomatic and then are ordered to quarantine, will test immediately at rate `symp_quar_test`
        # 4. People who have severe symptoms will be tested
        # 5. People test before leaving quarantine at rate `leaving_quar_prob` (set to 1 to ensure everyone leaving quarantine must have been tested)
        # 6. People that have been diagnosed will not be tested
        # 7. People that are already waiting for a diagnosis will not be retested
        # 8. People quarantine while waiting for their diagnosis with compliance `test_isolation_compliance`
        # 9. People already on quarantine while tested will not have their quarantine shortened, but if they are tested at the end of their
        #    quarantine, the quarantine will be extended

        # Construct the testing probabilities piece by piece -- complicated, since need to do it in the right order
        test_probs = np.zeros(sim.n)  # Begin by assigning equal testing probability to everyone

        # (1) People wait swab_delay days before they decide to start testing. If swab_delay is 0 then they will be eligible as soon as they are symptomatic
        symp_inds = cvu.true(sim.people.symptomatic) # People who are symptomatic
        symp_test_inds = symp_inds[sim.people.date_symptomatic[symp_inds] == (t-self.swab_delay)]  # People who became symptomatic previously and are eligible to test today
        test_probs[symp_test_inds] = self.symp_prob

        # People whose symptomatic scheduled day falls during quarantine will test at the symp_quar_prob rate
        # People who are already symptomatic, missed their test, and then enter quarantine, will test at the symp_quar_prob rate
        # People get quarantined at 11:59pm so the people getting quarantined today haven't been quarantined yet.
        # The order is
        # Day 4 - Test, quarantine people waiting for results
        # Day 4 - Trace
        # Day 4 - Quarantine known contacts
        # Day 5 - Test, nobody has entered quarantine on day 5 yet - if someone was symptomatic and untested and was quarantined *yesterday* then
        #         they need to be tested *today*

        # Someone waiting for a test result shouldn't retest. So we check that they aren't already waiting for their test.
        # Note that people are diagnosed after interventions are executed,
        # therefore if someone is tested on day 3 and the test delay is 2, on day 5 then sim.people.diagnosed will NOT
        # be true at the point where this code is executing. Therefore, they should not be eligible to retest. It's
        # like they are going to receive their results at 11:59pm so the decisions they make during the day are based
        # on not having been diagnosed yet. Hence > is used here so that on day 3+2=5, they won't retest. (i.e. they are
        # waiting for their results if the day they recieve their results is > the current day). Note that they become
        # symptomatic prior to interventions e.g. they wake up with symptoms
        if sim.t > 0:
            # If quarantined, there's no swab delay

            # (2) People who become symptomatic while quarantining test immediately
            quarantine_test_inds = symp_inds[sim.people.quarantined[symp_inds] & (sim.people.date_symptomatic[symp_inds] == t)] # People that became symptomatic today while already on quarantine
            test_probs[quarantine_test_inds] = self.symp_quar_prob  # People with symptoms in quarantine are eligible to test without waiting

            # (3) People who are symptomatic and undiagnosed before entering quarantine, test as soon as they are quarantined
            newly_quarantined_test_inds = cvu.true((sim.people.date_quarantined == (sim.t-1)) & sim.people.symptomatic & ~sim.people.diagnosed) # People that just entered quarantine, who are current symptomatic and undiagnosed
            test_probs[newly_quarantined_test_inds] = self.symp_quar_prob  # People with symptoms that just entered quarantine are eligible to test

        # (4) People with severe symptoms that would be hospitalised are guaranteed to be tested
        test_probs[sim.people.severe] = 1.0  # People with severe symptoms are guaranteed to be tested unless already diagnosed or awaiting results

        # (5) People leaving quarantine test before leaving
        # This tests policies for testing people at least once during quarantine
        # - If leaving_quar_prob=1 then everyone leaving quarantine must have been tested during quarantine
        # - If someone was tested during their quarantine, they don't need to test again
        if self.leaving_quar_prob:
            to_test = cvu.true(sim.people.quarantined)  # Everyone currently on quarantine
            to_test = to_test[(sim.people.date_end_quarantine[to_test]-self.test_delay) == sim.t] # Everyone leaving quarantine that needs to have been tested by today at the latest
            to_test = to_test[~(sim.people.date_tested[to_test] > sim.people.date_quarantined[to_test])] # Note that this is not the same as <= because of NaNs - if someone was never tested, then both <= and > are False
            test_probs[to_test] = np.maximum(test_probs[to_test], self.leaving_quar_prob) # If they are already supposed to test at a higher rate e.g. severe symptoms, keep it

        # (6) People that have been diagnosed aren't tested
        diag_inds = cvu.true(sim.people.diagnosed)
        test_probs[diag_inds] = 0.0  # People who are diagnosed or awaiting test results don't test

        # (7) People waiting for results don't get tested
        tested_inds = cvu.true(np.isfinite(sim.people.date_tested))
        pending_result_inds = tested_inds[(sim.people.date_tested[tested_inds] + self.test_delay) > sim.t]  # People who have been tested and will receive test results after the current timestep
        test_probs[pending_result_inds] = 0.0  # People awaiting test results don't test

        # Test people based on their per-person test probability
        test_inds = cvu.true(cvu.binomial_arr(test_probs))
        sim.people.test(test_inds, test_sensitivity=self.test_sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay) # Actually test people
        sim.results['new_tests'][t] += int(len(test_inds))

        if self.test_isolation_compliance:
            # If people are meant to quarantine while waiting for their test, then quarantine some/all of the people waiting for tests
            quar_inds = cvu.binomial_filter(self.test_isolation_compliance,test_inds)
            sim.people.quarantine(quar_inds, period=self.test_delay)

        return test_probs


class limited_contact_tracing(cv.contact_tracing):
    """
    Contact tracing with capacity limit

    """

    def __init__(self, capacity=np.inf, **kwargs):
        """

        Args:
            capacity: Maximum number of newly diagnosed people to trace per day
        """
        super().__init__(**kwargs) # Initialize the Intervention object
        self.capacity = capacity  #: Dict with capacity by layer e.g. {'H': 100, 'W': 50}
        assert not self.presumptive, 'Presumptive tracing not supported by this class' # Disable for simplicity (reduce execution paths here until needed)

    def apply(self, sim):
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Everyone that was diagnosed today could potentially be traced
        trace_from_inds = cvu.true(sim.people.date_diagnosed == t) # Diagnosed this time step, time to trace
        if not len(trace_from_inds):
            return
        trace_from_inds = trace_from_inds.astype(np.int64)

        capacity = np.floor(self.capacity / sim.rescale_vec[t])  # Scale capacity based on dynamic rescaling factor
        if len(trace_from_inds) > capacity:
            trace_from_inds = trace_from_inds[cvu.choose(len(trace_from_inds),capacity)]

        ind_set = set(trace_from_inds)

        actual_infections = [x for x in sim.people.infection_log if (x['source'] in ind_set or x['target'] in ind_set)] # People who were infected in a traceable layer involving the person being traced

        # Extract the indices of the people who'll be contacted
        for lkey, this_trace_prob in self.trace_probs.items():

            if this_trace_prob == 0:
                continue

            # Find current layer contacts
            notification_set = cvu.find_contacts(sim.people.contacts[lkey]['p1'], sim.people.contacts[lkey]['p2'], trace_from_inds)

            # Add interactions at previous timesteps that resulted in transmission. It's bi-directional because if the source
            # interacts with the target, the target would be able to name the source as a known contact with the same probability
            # as in the reverse direction.
            for infection in actual_infections:
                if infection['layer'] == lkey:
                    notification_set.add(infection['source'])
                    notification_set.add(infection['target'])

            # Check contacts
            edge_inds = np.fromiter(notification_set.difference(ind_set), dtype=cvd.default_int)
            edge_inds.sort()
            contact_inds = cvu.binomial_filter(this_trace_prob, edge_inds)  # Filter the indices according to the probability of being able to trace this layer
            if len(contact_inds):
                this_trace_time = self.trace_time[lkey]
                sim.people.known_contact[contact_inds] = True
                sim.people.date_known_contact[contact_inds] = np.fmin(sim.people.date_known_contact[contact_inds], sim.t + this_trace_time)
                sim.people.quarantine(contact_inds, start_date=sim.t + this_trace_time) # Schedule quarantine for the notified people to start on the date they will be notified


class limited_contact_tracing_2(cv.contact_tracing):
    """
    Contact tracing with capacity limit

    This implementation actually tracks who the contact was, for the purpose of tracking clusters.
    Although the tracing should be basically the same as `limited_contact_tracing`, this function
    is much slower as a result.

    """

    def __init__(self, capacity=np.inf, dynamic_layers=None, **kwargs):
        """

        Args:
            capacity: Maximum number of newly diagnosed people to trace per day
        """
        super().__init__(**kwargs)  # Initialize the Intervention object
        self.capacity = capacity  #: Dict with capacity by layer e.g. {'H': 100, 'W': 50}
        self.dynamic_layers = dynamic_layers or []  #: List of layers to trace via infection log (if their contacts are regenerated each timestep)
        self.notifications = nx.DiGraph()  # Notification edge a->b means that `b` was notified that `a` was a suspected case

    def apply(self, sim):
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Figure out whom to test and trace
        if not self.presumptive:
            trace_from_inds = cvu.true(sim.people.date_diagnosed == t)  # Diagnosed this time step, time to trace
        else:
            just_tested = cvu.true(sim.people.date_tested == t)  # Tested this time step, time to trace
            trace_from_inds = cvu.itruei(sim.people.exposed, just_tested)  # This is necessary to avoid infinite chains of asymptomatic testing

        capacity = np.floor(self.capacity / sim.rescale_vec[t])  # Scale capacity based on dynamic rescaling factor
        if len(trace_from_inds) > capacity:
            trace_from_inds = trace_from_inds[cvu.choose(len(trace_from_inds), capacity)]

        if not len(trace_from_inds):
            return

        traceable_layers = {k: v for k, v in self.trace_probs.items() if v != 0.}  # Only trace if there's a non-zero tracing probability
        dynamic_traceable = {k: v for k, v in traceable_layers.items() if k in self.dynamic_layers}

        for ind in trace_from_inds:

            # Interactions at previous timesteps recorded involving this person
            if dynamic_traceable:
                dynamic_infections = [x for x in sim.people.infection_log if (x['source'] == ind or x['target'] == ind)]
            else:
                dynamic_infections = []

            # Extract the indices of the people who'll be contacted
            for lkey, this_trace_prob in traceable_layers.items():

                layer = sim.people.contacts[lkey]

                # All contacts of this person in the current layer at the current timestep
                a = layer['p2'][layer['p1'] == ind]
                b = layer['p1'][layer['p2'] == ind]
                contacts = set(a)
                contacts.update(b)

                # Then add any dynamic contacts
                for infection in dynamic_infections:
                    if infection['layer'] == lkey:
                        contacts.add(infection['source'])
                        contacts.add(infection['target'])

                contacts.discard(ind)  # Can't be a contact of oneself

                identified_contacts = cvu.binomial_filter(this_trace_prob, np.array(list(contacts)))  # Filter the indices according to the probability of being able to trace this layer
                for contact in identified_contacts:
                    if not self.notifications.has_edge(ind, contact):
                        self.notifications.add_edge(ind, contact, t=t + self.trace_time[lkey], layer=lkey)

                if len(identified_contacts):
                    sim.people.known_contact[identified_contacts] = True
                    sim.people.date_known_contact[identified_contacts] = np.fmin(sim.people.date_known_contact[identified_contacts], t + self.trace_time[lkey])
                    sim.people.quarantine(identified_contacts, start_date=t + self.trace_time[lkey])  # Schedule quarantine for the notified people to start on the date they will be notified


def story(sim, uid):
    p = sim.people[uid]

    if not p.susceptible:
        if np.isnan(p.date_symptomatic):
            print(f'{uid} is a {p.age:.0f} year old that had asymptomatic COVID')
        else:
            print(f'{uid} is a {p.age:.0f} year old that contracted COVID')
    else:
        print(f'{uid} is a {p.age:.0f} year old')

    if len(p.contacts['H']):
        print(f'{uid} lives with {len(p.contacts["H"])} people')
    else:
        print(f'{uid} lives alone')

    if len(p.contacts['W']):
        print(f'{uid} works with {len(p.contacts["W"])} people')
    else:
        print(f'{uid} works alone')

    events = []

    dates = {
    'date_critical': 'became critically ill and needed ICU care',
    'date_dead': 'died',
    'date_diagnosed': 'was diagnosed with COVID',
    'date_end_quarantine': 'ended quarantine',
    'date_infectious': 'became infectious',
    'date_known_contact': f'was notified they may have been exposed to COVID',
    'date_pos_test': 'recieved their positive test result',
    'date_quarantined': 'entered quarantine',
    'date_recovered': 'recovered',
    'date_severe': 'developed severe symptoms and needed hospitalization',
    'date_symptomatic': 'became symptomatic',
    'date_tested': 'was tested for COVID',
    }

    for attribute, message in dates.items():
        date = getattr(p,attribute)
        if not np.isnan(date):
            events.append((date, message))

    for infection in sim.people.infection_log:
        if infection['target'] == uid:
            if infection["layer"]:
                events.append((infection['date'], f'was infected with COVID by {infection["source"]} at {infection["layer"]}'))
            else:
                events.append((infection['date'], f'was infected with COVID as a seed infection'))

        if infection['source'] == uid:
            x = len([a for a in sim.people.infection_log if a['source'] == infection['target']])
            events.append((infection['date'],f'gave COVID to {infection["target"]} at {infection["layer"]} ({x} secondary infections)'))

    for day, event in sorted(events, key=lambda x: x[0]):
        print(f'On Day {day:.0f}, {uid} {event}')


def result_df(sim):
    resdict = sim.export_results(for_json=False)
    result_df = pd.DataFrame.from_dict(resdict)
    result_df.index = sim.datevec[0:len(result_df)]
    result_df.index.name = 'date'
    return result_df


def save_csv(sim, fname):
    df = result_df(sim)
    result_df.to_csv(fname)



def get_individual_traces(key, sims, convolve=False, num_days=3):
    ys = []
    for this_sim in sims:
        ys.append(this_sim.results[key].values)
    yarr = np.array(ys)

    if convolve:
        for idx in range(yarr.shape[0]):
             yarr[idx, :] = np.convolve(yarr[idx, :], np.ones((num_days, ))/num_days, mode='same')

    yarr = np.array(yarr).T

    return yarr


def get_ensemble_trace(key, sims, return_traces=True, **kwargs):
    """
    Get median trace
    """
    yarr_ = get_individual_traces(key, sims, **kwargs)
    yarr = np.percentile(yarr_, 50, axis=1)
    if return_traces:
        return yarr, yarr_
    return yarr


def detect_outbreak(data, num_cases=5.0, use_nan=False):
    """
    Get the index of the last day of the first instance of three consecutive days above num_cases 
    """
    # Case outbreak
    idx = np.argmax((np.where(data >= num_cases, 1.0, 0.0) + np.roll(np.where(data >= num_cases, 1.0, 0.0), 1) + np.roll(np.where(data >= num_cases, 1.0, 0.0), -1))+1)
    
    # If there is no outbreak 
    if idx == 0:
        if use_nan:
            idx = np.nan
        else:
            idx = None
    return idx 


def detect_first_case(data, num_cases=1.0, use_nan=False):
    """
    Get the index of the last day of the first instance of three consecutive days above num_cases 
    """
    # Case outbreak
    idx = np.argmax((np.where(data >= num_cases, 1.0, 0.0)))
    
    # If there is no outbreak 
    if idx == 0:
        if use_nan:
            idx = np.nan
        else:
            idx = None
    return idx 

def detect_outbreak_case(data, day_idx):
    """
    Get the itype of outbreak: outbreak, under control, contained 
    """
    # Case outbreak
    
    # If there is no outbreak 
    if np.isnan(day_idx):
        # Rerun detection - will pick up cases >=1 and <=4
        if data.sum() == 0:
            outbreak_case = 'contained'
        else:
            outbreak_case = 'under_control' 
    else:
        outbreak_case = 'outbreak'
    return outbreak_case


def calculate_first_case_stats(data_nc, data_ni):
    """
    data_nc and data_ni have shape tpts x nruns
    data_nc --> new_cases
    data_nc --> 
    """
    nruns = data_nc.shape[1]
    local_first_case_idx = []
    local_first_case_inf = []
    for idx in range(data_nc.shape[1]):
        # First case day
        fc_day_idx = detect_first_case(data_nc[:, idx], use_nan=True)
        local_first_case_idx.append(fc_day_idx)
        if np.isnan(fc_day_idx):
           local_first_case_inf.append(np.nan)
        else:
           local_first_case_inf.append(data_ni[fc_day_idx, idx])

    local_first_case_day_dist = np.array(local_first_case_idx)
    local_first_case_inf_dist = np.array(local_first_case_inf)


    # Get stats in terms od day to first case
    fc_day_av = np.nanmean(local_first_case_day_dist)
    fc_day_md = np.nanmedian(local_first_case_day_dist)
    fc_day_sd = np.nanstd(local_first_case_day_dist)
    # get stats, number of infections 
    fc_inf_av = np.nanmean(local_first_case_inf_dist)
    fc_inf_md = np.nanmedian(local_first_case_inf_dist)
    fc_inf_sd = np.nanstd(local_first_case_inf_dist)

    # Get stats of first day cases
    return fc_day_av, fc_day_md, fc_day_sd, fc_inf_av, fc_inf_md, fc_inf_sd 


def calculate_outbreak_stats(data):
    """
    data has shape tpts x nruns

    """
    nruns = data.shape[1]
    local_outbreak_idx = []
    case_dict = {'outbreak': 0, 'under_control': 0, 'contained': 0}
    for idx in range(data.shape[1]):
        day_idx = detect_outbreak(data[:, idx], use_nan=True)
        case_label = detect_outbreak_case(data[: idx], day_idx)
        # Update tally for each case
        case_dict[case_label] += 1.0/nruns
        local_outbreak_idx.append(day_idx)

    local_outbreak_dist = np.array(local_outbreak_idx)

    # Get stats of proper "outbreaks"
    ou_day_av = np.nanmean(local_outbreak_dist)
    ou_day_md = np.nanmedian(local_outbreak_dist)
    ou_day_sd = np.nanstd(local_outbreak_dist)
    ou_prob = case_dict["outbreak"] * 100.0
    uc_prob = case_dict["under_control"] * 100.0
    co_prob = case_dict["contained"] *100.0

    # Get stats of first day cases

    return ou_day_av, ou_day_md, ou_day_sd, ou_prob, uc_prob, co_prob 
