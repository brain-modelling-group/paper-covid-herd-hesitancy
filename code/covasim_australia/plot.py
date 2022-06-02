import datetime as dt
import matplotlib.ticker as ticker
import numpy as np
import pylab as pl
import sciris as sc

from covasim import defaults as cvd
from matplotlib.ticker import StrMethodFormatter
from covasim_australia.policy_updates import PolicySchedule


def policy_plot(scens, scens_toplot=None, outcomes_toplot=None, plot_ints=False, do_save=False, fig_path=None, fig_args=None, plot_args=None,
    axis_args=None, fill_args=None, legend_args=None, as_dates=True, dateformat=None,
    interval=None, n_cols=1, font_size=18, font_family=None, grid=True, commaticks=True,
    do_show=True, sep_figs=False, verbose=1, y_lim=None, name = ' '):

    '''
    Plot the results -- can supply arguments for both the figure and the plots.

    Args:
        plot_ints (Bool): Whether or not to plot intervention start dates
        outcomes_toplot (dict): Dict of results to plot; see default_scen_plots for structure
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

    Returns:
        fig: Figure handle
    '''

    sc.printv('Plotting...', 1, verbose)

    if outcomes_toplot is None:
        outcomes_toplot = cvd.get_scen_plots()
    outcomes_toplot = sc.dcp(outcomes_toplot)  # In case it's supplied as a dict

    plot_args = sc.mergedicts({'lw': 1, 'alpha': 0.7}, plot_args)
    fill_args = sc.mergedicts({'alpha': 0.2}, fill_args)
    legend_args = sc.mergedicts({'loc': 'upper left'}, legend_args)

    # one location per column
    ncols = len(scens.keys())
    nrows = len(outcomes_toplot)

    fig, axes = pl.subplots(nrows=nrows, ncols=ncols, sharex='col', figsize=(8, 5))

    # plot each location as a column
    for i, loc in enumerate(scens):

        scen = scens[loc]

        if ncols == 1 & nrows == 1:
            axes.set_title(loc)
        elif ncols == 1 & nrows > 1:
            axes[0].set_title(loc)
        else:
            axes[0, i].set_title(loc)  # column title

        # get the scenarios to plot for this location
        s_toplot = None
        if scens_toplot is not None:
            if scens_toplot.get(loc) is not None:
                s_toplot = scens_toplot[loc]

        # plot each outcome in outcomes_toplot as a row
        for j, subplot_title in enumerate(outcomes_toplot):
            baseline_days = []
            otherscen_days = []

            if ncols == 1 & nrows == 1:
                axes.set_ylabel(subplot_title)
                this_subplot = axes
            elif ncols == 1 & nrows > 1:
                axes[j].set_ylabel(subplot_title)
                this_subplot = axes[j]
            else:
                axes[j,0].set_ylabel(subplot_title)
                this_subplot = axes[j, i]

            reskey = outcomes_toplot[subplot_title]
            if isinstance(reskey, list):  # if it came from an odict
                reskey = reskey[0]

            # check which scenarios to plot
            resdata = scen.results[reskey]
            if s_toplot is None:
                s_toplot = resdata.keys()
            colors = sc.gridcolors(len(s_toplot))

            # plot the outcomes for each scenario
            for k, scen_name in enumerate(s_toplot):
                scendata = resdata[scen_name]
                this_subplot.fill_between(scen.tvec, scendata.low, scendata.high, **fill_args)
                this_subplot.plot(scen.tvec, scendata.best, label=scendata.name, c=colors[k], **plot_args)

            # add legend to first plot in each column
            if j == 0:
                hs, lbs = this_subplot.get_legend_handles_labels()
                this_subplot.legend(hs, lbs, **legend_args)

            # plot the data
            if scen.base_sim.data is not None and reskey in scen.base_sim.data:
                data_t = np.array((scen.base_sim.data.index - scen.base_sim['start_day']) / np.timedelta64(1, 'D'))
                this_subplot.plot(data_t, scen.base_sim.data[reskey], 'sk', **plot_args)

            # Set xticks as dates
            if as_dates:
                @ticker.FuncFormatter
                def date_formatter(x, pos):
                    return (scen.base_sim['start_day'] + dt.timedelta(days=x)).strftime('%b-%d')

                this_subplot.xaxis.set_major_formatter(date_formatter)
                if not interval:
                    this_subplot.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            # format y-axis to use commas
            if commaticks:
                sc.commaticks(fig=fig)

            if plot_ints:
                scennum = 0
                # for s, scen_name in enumerate(scen.sims):
                for scen_name in s_toplot:
                    if scen_name.lower() != 'baseline':
                        for intervention in scen.sims[scen_name][0]['interventions']:
                            if hasattr(intervention, 'days') and isinstance(intervention, PolicySchedule):
                                otherscen_days = [day for day in intervention.days if
                                                  day not in baseline_days and day not in otherscen_days]
                            elif hasattr(intervention, 'start_day'):
                                if intervention.start_day not in baseline_days and intervention.start_day not in otherscen_days and intervention.start_day != 0:
                                    otherscen_days.append(intervention.start_day)
                                if intervention.end_day not in baseline_days and intervention.end_day not in otherscen_days and isinstance(
                                        intervention.end_day, int) and intervention.end_day < scen.sims[scen_name][0][
                                    'n_days']:
                                    otherscen_days.append(intervention.end_day)
                            for day in otherscen_days:
                                this_subplot.axvline(x=day, color=colors[scennum], linestyle='--')
                    else:
                        for intervention in scen.sims[scen_name][0]['interventions']:
                            if hasattr(intervention, 'days') and isinstance(intervention, PolicySchedule):
                                baseline_days = [day for day in intervention.days if day not in baseline_days]
                            elif hasattr(intervention, 'start_day'):
                                if intervention.start_day not in baseline_days and intervention.start_day != 0:
                                    baseline_days.append(intervention.start_day)
                            for day in baseline_days:
                                this_subplot.axvline(x=day, color=colors[scennum], linestyle='--')
                    scennum += 1

    # Ensure the figure actually renders or saves
    if do_save:
        if fig_path is None:  # No figpath provided - see whether do_save is a figpath
            fig_path = name + '.png'  # Just give it a default name
        fig_path = sc.makefilepath(fig_path)  # Ensure it's valid, including creating the folder
        pl.savefig(fig_path, dpi=300)

    if do_show:
        pl.show()
    else:
        pl.close(fig)

    return fig


def plot_scens(scens, fig_path=None, do_save=True, do_show=True, figsize=(5, 10), for_powerpoint=False):
    if do_save and fig_path is None:
        fig_path = '/figures/baseline_v2.png'

    fig_args = {'figsize': figsize}
    if for_powerpoint:
        outcomes_toplot = scens.results['new_infections']
    else:
        outcomes_toplot = ['new_infections', 'cum_infections', 'new_diagnoses', 'cum_deaths']

    policy_plot(scens,
                plot_ints=True,
                do_save=do_save,
                do_show=do_show,
                fig_path=fig_path,
                interval=14,
                fig_args=fig_args,
                font_size=8,
                outcomes_toplot=outcomes_toplot)

    return

