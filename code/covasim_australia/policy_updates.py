import covasim_australia.contacts as co
import covasim as cv
import covasim.defaults as cvd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pylab as pl
import sciris as sc
import covasim.misc as cvm
from covasim.interventions import process_days
from covasim import utils as cvu


class PolicySchedule(cv.Intervention):
    def __init__(self, baseline: dict, policies: dict):
        """
        Create policy schedule

        The policies passed in represent all of the possible policies that a user
        can subsequently schedule using the methods of this class

        Example usage:

            baseline = {'H':1, 'S':0.75}
            policies = {}
            policies['Close schools'] = {'S':0}
            schedule = PolicySchedule(baseline, policies)
            schedule.add('Close schools', 10) # Close schools on day 10
            schedule.end('Close schools', 20) # Reopen schools on day 20
            schedule.remove('Close schools')  # Don't use the policy at all

        Args:
            baseline: Baseline (relative) beta layer values e.g. {'H':1, 'S':0.75}
            policies: Dict of policies containing the policy name and relative betas for each policy e.g. {policy_name: {'H':1, 'S':0.75}}

        """
        super().__init__()
        self._baseline = baseline  #: Store baseline relative betas for each layer
        self.policies = sc.dcp(policies)  #: Store available policy interventions (name and effect)
        for policy, layer_values in self.policies.items():
            assert set(layer_values.keys()).issubset(self._baseline.keys()), f'Policy "{policy}" has effects on layers not included in the baseline'
        self.policy_schedule = []  #: Store the scheduling of policies [(start_day, end_day, policy_name)]
        self.days = {}  #: Internal cache for when the beta_layer values need to be recalculated during simulation. Updated using `_update_days`

    def start(self, policy_name: str, start_day: int) -> None:
        """
        Change policy start date

        If the policy is not already present, then it will be added with no end date

        Args:
            policy_name: Name of the policy to change start date for
            start_day: Day number to start policy

        Returns: None

        """
        n_entries = len([x for x in self.policy_schedule if x[2] == policy_name])
        if n_entries < 1:
            self.add(policy_name, start_day)
            return
        elif n_entries > 1:
            raise Exception('start_policy() cannot be used to start a policy that appears more than once - need to manually add an end day to the desired instance')

        for entry in self.policy_schedule:
            if entry[2] == policy_name:
                entry[0] = start_day

        self._update_days()

    def end(self, policy_name: str, end_day: int) -> None:
        """
        Change policy end date

        This only works if the policy only appears once in the schedule. If a policy gets used multiple times,
        either add the end days upfront, or insert them directly into the policy schedule. The policy should
        already appear in the schedule

        Args:
            policy_name: Name of the policy to end
            end_day: Day number to end policy (policy will have no effect on this day)

        Returns: None

        """

        n_entries = len([x for x in self.policy_schedule if x[2] == policy_name])
        if n_entries < 1:
            raise Exception('Cannot end a policy that is not already scheduled')
        elif n_entries > 1:
            raise Exception('end_policy() cannot be used to end a policy that appears more than once - need to manually add an end day to the desired instance')

        for entry in self.policy_schedule:
            if entry[2] == policy_name:
                if end_day <= entry[0]:
                    raise Exception(f"Policy '{policy_name}' starts on day {entry[0]} so the end day must be at least {entry[0]+1} (requested {end_day})")
                entry[1] = end_day

        self._update_days()

    def add(self, policy_name: str, start_day: int, end_day: int = np.inf) -> None:
        """
        Add a policy to the schedule

        Args:
            policy_name: Name of policy to add
            start_day: Day number to start policy
            end_day: Day number to end policy (policy will have no effect on this day)

        Returns: None

        """
        assert policy_name in self.policies, 'Unrecognized policy'
        self.policy_schedule.append([start_day, end_day, policy_name])
        self._update_days()

    def remove(self, policy_name: str) -> None:
        """
        Remove a policy from the schedule

        All instances of the named policy will be removed from the schedule

        Args:
            policy_name: Name of policy to remove

        Returns: None

        """

        self.policy_schedule = [x for x in self.policy_schedule if x[2] != policy_name]
        self._update_days()

    def _update_days(self) -> None:
        # This helper function updates the list of days on which policies start or stop
        # The apply() function only gets run on those days
        self.days = {x[0] for x in self.policy_schedule}.union({x[1] for x in self.policy_schedule if np.isfinite(x[1])})

    def _compute_beta_layer(self, t: int) -> dict:
        # Compute beta_layer at a given point in time
        # The computation is done from scratch each time
        beta_layer = self._baseline.copy()
        for start_day, end_day, policy_name in self.policy_schedule:
            rel_betas = self.policies[policy_name]
            if t >= start_day and t < end_day:
                for layer in beta_layer:
                    if layer in rel_betas:
                        beta_layer[layer] *= rel_betas[layer]
        return beta_layer

    def apply(self, sim: cv.BaseSim):
        if sim.t in self.days:
            sim['beta_layer'] = self._compute_beta_layer(sim.t)
            if sim['verbose']:
                print(f"PolicySchedule: Changing beta_layer values to {sim['beta_layer']}")
                for entry in self.policy_schedule:
                    if sim.t == entry[0]:
                        print(f'PolicySchedule: Turning on {entry[2]}')
                    elif sim.t == entry[1]:
                        print(f'PolicySchedule: Turning off {entry[2]}')

    def plot_gantt(self, max_time=None, start_date=None, interval=None, pretty_labels=None):
        """
        Plot policy schedule as Gantt chart

        Returns: A matplotlib figure with a Gantt chart

        """
        fig, ax = plt.subplots()
        if max_time:
            max_time += 5
        else:
            max_time = np.nanmax(np.array([x[1] for x in self.policy_schedule if np.isfinite(x[1])]))

        #end_dates = [x[1] for x in self.policy_schedule if np.isfinite(x[1])]
        if interval:
            xmin, xmax = ax.get_xlim()
            ax.set_xticks(pl.arange(xmin, xmax + 1, interval))

        if start_date:
            @ticker.FuncFormatter
            def date_formatter(x, pos):
                return (start_date + dt.timedelta(days=x)).strftime('%b-%d')

            ax.xaxis.set_major_formatter(date_formatter)
            if not interval:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_xlabel('Dates')
            ax.set_xlim((0, max_time + 5))  # Extend a few days so the ends of policies can be seen

        else:
            ax.set_xlim(0, max_time + 5)  # Extend a few days so the ends of policies can be seen
            ax.set_xlabel('Days')
        schedule = sc.dcp(self.policy_schedule)
        if pretty_labels:
            policy_index = {pretty_labels[x]: i for i, x in enumerate(self.policies.keys())}
            for p, pol in enumerate(schedule):
               pol[2] = pretty_labels[pol[2]]
            colors = sc.gridcolors(len(pretty_labels))
        else:
            policy_index = {x: i for i, x in enumerate(self.policies.keys())}
            colors = sc.gridcolors(len(self.policies))
        ax.set_yticks(np.arange(len(policy_index.keys())))
        ax.set_yticklabels(list(policy_index.keys()))
        ax.set_ylim(0 - 0.5, len(policy_index.keys()) - 0.5)

        for start_day, end_day, policy_name in schedule:
            if not np.isfinite(end_day):
                end_day = 1e6 # Arbitrarily large end day
            ax.broken_barh([(start_day, end_day - start_day)], (policy_index[policy_name] - 0.5, 1), color=colors[policy_index[policy_name]])

        return fig


class AppBasedTracing(cv.Intervention):
    def __init__(self, name, days, coverage, layers, start_day=0, end_day=None, trace_time=0):
        """
        App based contact tracing parametrized by coverage
        Args:
            days: List/array of day indexes on which a coverage value takes effect e.g. [14, 28]
            coverage: List/array of coverage values corresponding to days e.g. [0.2,0.4]
            layers: List of layer names traceable by the app e.g. ['Household','Beach']
            start_day (int): intervention start day.
            end_day (int): intervention end day
            trace_time: Tracing time (default is 0 as contacts are automatically notified via the system)
        """
        super().__init__()
        assert len(days) == len(coverage), 'Must specify same number of days as coverage values'
        self.name = name
        self.days = sc.promotetoarray(days)
        self.coverage = sc.promotetoarray(coverage)
        self.layers = layers
        assert self.days[0] <= start_day, f'Initial "{name}" coverage change must be before or on start day. \n First change day: {self.days[0]}. \n Start day: {start_day}'
        self.trace_time = dict.fromkeys(self.layers, trace_time)
        self.start_day = start_day
        self.end_day = end_day
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.start_day = sim.day(self.start_day)
        self.end_day = sim.day(self.end_day)
        self.days = process_days(sim, self.days)
        return

    def apply(self, sim):
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return
        # Index to use for the current day
        idx = np.argmax(self.days > sim.t)-1 # nb. if sim.t<self.days[0] this will be wrong, hence the validation in __init__()
        trace_prob = dict.fromkeys(self.layers, self.coverage[idx] ** 2)  # Probability of both people having the app
        just_diagnosed_inds = cvu.true(sim.people.date_diagnosed == t)
        if len(just_diagnosed_inds):
            sim.people.trace(just_diagnosed_inds, trace_prob, self.trace_time)
        return


class UpdateNetworks(cv.Intervention):
    def __init__(self, layers, contact_numbers, layer_members, start_day=0, end_day=None, dispersion=None):
        """
        Update random networks at each time step
        Args:
            layers: List of layer names to resample
            start_day (int): intervention start day.
            end_day (int): intervention end day
            contact_numbers: dictionary of average contacts for each layer
        """
        super().__init__()
        self.layers = layers
        self.start_day = start_day
        self.end_day = end_day
        self.contact_numbers = contact_numbers
        self.dispersion = dispersion
        self.layer_members = layer_members  # {lkey: [uids]}
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.start_day = sim.day(self.start_day)
        self.end_day = sim.day(self.end_day)
        return

    def apply(self, sim):
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        # Loop over dynamic keys
        for lkey in self.layers:

            # Sample new contacts, overwriting the existing ones
            sim.people.contacts[lkey]['p1'], sim.people.contacts[lkey]['p2'] = co.make_random_contacts(include_inds=self.layer_members[lkey],
                                                                             mean_number_of_contacts=self.contact_numbers[lkey],
                                                                             dispersion=self.dispersion[lkey],
                                                                             array_output=True,
                                                                             )
            # Update beta shape and check validity
            sim.people.contacts[lkey]['beta'] = np.ones(sim.people.contacts[lkey]['p1'].shape, dtype=cvd.default_float)
            sim.people.contacts[lkey].validate()

        return


def check_policy_changes(scenario: dict):
    if scenario.get('turn_off') is not None and scenario.get('replace') is not None:
        clash_off_replace_pols = {policy: p for p, policy in enumerate(scenario['turn_off']['off_pols']) if policy in scenario['replace'].keys()}
        if clash_off_replace_pols:
            off_dates = [scenario['turn_off']['dates'][p] for p in list(clash_off_replace_pols.values())]
            replace_dates = [scenario['replace'][policy]['dates'][0] for policy in clash_off_replace_pols]
            for o_date in off_dates:
                for r_date in replace_dates:
                    if o_date <= r_date:
                        date_index = scenario['turn_off']['dates'].index(o_date)
                        print('The torun dict had a clash between turning off and replacing policy %s. Replacement has been prioritised.' % scenario['turn_off']['off_pols'][date_index])
                        del scenario['turn_off']['off_pols'][date_index]
                        del scenario['turn_off']['dates'][date_index]
    if scenario.get('turn_on') is not None and scenario.get('replace') is not None:
        clash_on_replace_pols = {}
        for old_pol in scenario['replace']:
            clash_on_replace_pols[old_pol] = {policy: p for p, policy in enumerate(scenario['turn_on']) if policy in scenario['replace'][old_pol]['replacements']}
        for clash in clash_on_replace_pols:
            if clash_on_replace_pols[clash]:
                on_dates = [scenario['turn_on'][policy] for policy in clash_on_replace_pols[clash]]
                replace_dates = [scenario['replace'][clash]['dates'][p] for p in list(clash_on_replace_pols[clash].values())]
                for o_date in on_dates:
                    for r_date in replace_dates:
                        if len(o_date) > 1:
                            if o_date[0] <= r_date and o_date[1] > r_date:
                                for on_policy in scenario['turn_on']:
                                    if o_date == scenario['turn_on'][on_policy]:
                                        print('The torun dict had a clash between turning on and using policy %s as a replacement. Replacement has been prioritised.' % scenario['turn_on'][on_policy].key())
                                        del on_policy
                        elif len(o_date) == 1:
                            if o_date[0] <= r_date:
                                for on_policy in scenario['turn_on']:
                                    if o_date == scenario['turn_on'][on_policy]:
                                        print('The torun dict had a clash between turning on and using policy %s as a replacement. Replacement has been prioritised.' % scenario['turn_on'][on_policy].key())
                                        del on_policy
                        else:
                            for on_policy in scenario['turn_on']:
                                    if not o_date:
                                        print('The torun dict was not supplied with dates for policy %s, so it has been removed.' % scenario['turn_on'][on_policy].key())
                                        del on_policy
    return scenario


def turn_off_policies(scen, baseline_schedule, beta_policies, import_policies, clip_policies, trace_policies,  i_cases, n_days, policy_dates, imports_dict):

    adapt_beta_policies = beta_policies
    adapt_clip_policies = clip_policies
    adapt_trace_policies = trace_policies
    if len(scen['turn_off'])>0:
        for p, policy in enumerate(scen['turn_off']['off_pols']):
            relax_day = sc.dcp(scen['turn_off']['dates'][p])
            if policy in policy_dates:
                if len(policy_dates[policy]) % 2 == 0 and policy_dates[policy][-1] < relax_day:
                    print('Not turning off policy %s at day %s because it is already off.' % (policy, str(relax_day)))
                elif len(policy_dates[policy]) % 2 != 0 and policy_dates[policy][-1] > relax_day:
                    print('Not turning off policy %s at day %s because it is already off. It will be turned on again on day %s' % (policy, str(relax_day), str(policy_dates[policy][-1])))
                else:
                    if policy in adapt_beta_policies:
                        baseline_schedule.end(policy, relax_day)
                    if policy in import_policies:
                        imports_dict = dict(days=np.append(range(len(i_cases)), np.arange(relax_day, n_days)), vals=np.append(i_cases, [import_policies[policy]['n_imports']] * (n_days-relax_day)))
                    if policy in clip_policies:
                        adapt_clip_policies[policy]['dates'] = sc.dcp([policy_dates[policy][-1], relax_day])
                    if policy in trace_policies:
                        adapt_trace_policies[policy]['end_day'] = relax_day
                        adapt_trace_policies[policy]['coverage'] = [cov for c, cov in enumerate(adapt_trace_policies[policy]['coverage']) if adapt_trace_policies[policy]['dates'][ c] < relax_day]
                        adapt_trace_policies[policy]['dates'] = [day for day in adapt_trace_policies[policy]['dates'] if day < relax_day]
                    policy_dates[policy].append(relax_day)
            else:
                print('Not turning off policy %s at day %s because it was never on.' % (policy, str(relax_day)))
    return baseline_schedule, imports_dict, adapt_clip_policies, adapt_trace_policies,  policy_dates


def turn_on_policies(scen, baseline_schedule, beta_policies, import_policies, clip_policies, trace_policies, i_cases, n_days, policy_dates, imports_dict):
    adapt_beta_policies = beta_policies
    adapt_clip_policies = clip_policies
    adapt_trace_policies = trace_policies
    for policy in scen['turn_on']:
        new_pol_dates = scen['turn_on'][policy]
        date_trigger = False
        start_day = new_pol_dates[0]
        if policy in policy_dates:
            if len(policy_dates[policy]) % 2 != 0:
                print('Not turning on policy %s at day %s because it is already on.' % (policy, str(start_day)))
            elif policy_dates[policy][-1] > start_day:
                print('Not turning on policy %s at day %s because it is already on. It will be turned off again on day %s' % (policy, str(start_day), str(policy_dates[policy][-1])))
            else:
                if policy in adapt_beta_policies:
                    if len(new_pol_dates) > 1:
                        baseline_schedule.add(policy, new_pol_dates[0], new_pol_dates[1])
                        policy_dates[policy].extend(new_pol_dates)
                        date_trigger = True
                    else:
                        baseline_schedule.add(policy, new_pol_dates[0], n_days)
                        policy_dates[policy].extend([new_pol_dates[0], n_days])
                        date_trigger = True
                if policy in import_policies:
                    if len(new_pol_dates) > 1:
                        imports_dict = dict(days=np.append(range(len(i_cases)), np.arange(policy_dates[policy][0], policy_dates[policy][1]) + np.arange(new_pol_dates[0], new_pol_dates[1])),
                                            vals=np.append(i_cases, [import_policies[policy]['n_imports']] * (policy_dates[policy][1] - policy_dates[policy][0]) + [import_policies[policy]['n_imports']] * (new_pol_dates[1]-new_pol_dates[0])))
                        if not date_trigger:
                            policy_dates[policy].extend(new_pol_dates)
                            date_trigger = True
                    else:
                        imports_dict = dict(days=np.append(range(len(i_cases)), np.arange(policy_dates[policy][0], policy_dates[policy][1]) + np.arange(new_pol_dates[0], n_days)),
                                            vals=np.append(i_cases, [import_policies[policy]['n_imports']] * (policy_dates[policy][1] - policy_dates[policy][0]) + [import_policies[policy]['n_imports']] * (n_days-new_pol_dates[0])))
                        if not date_trigger:
                            policy_dates[policy].extend([new_pol_dates[0], n_days])
                            date_trigger = True
                if policy in clip_policies:
                    adapt_clip_policies[policy + 'v2'] = sc.dcp(adapt_clip_policies[policy])
                    if len(new_pol_dates) > 1:
                        adapt_clip_policies[policy + 'v2']['dates'] = new_pol_dates
                        if not date_trigger:
                            policy_dates[policy].extend(new_pol_dates)
                            date_trigger = True
                    else:
                        adapt_clip_policies[policy + 'v2']['dates'] = [new_pol_dates[0], n_days]
                        if not date_trigger:
                            policy_dates[policy].extend([new_pol_dates[0], n_days])
                            date_trigger = True
                if policy in trace_policies:
                    adapt_trace_policies[policy + 'v2']= sc.dcp(adapt_trace_policies[policy])
                    if len(new_pol_dates) > 1:
                        adapt_trace_policies[policy + 'v2']['start_day'] = start_day
                        adapt_trace_policies[policy + 'v2']['dates'] = [start_day]
                        adapt_trace_policies[policy + 'v2']['end_day'] = new_pol_dates[1]
                        adapt_trace_policies[policy + 'v2']['coverage'] = [adapt_trace_policies[policy]['coverage'][-1]] # Start coverage where it was when trace_app was first ended, not sure what best option is here
                        if not date_trigger:
                            policy_dates[policy].extend(new_pol_dates)
                    else:
                        adapt_trace_policies[policy + 'v2']['start_day'] = start_day
                        adapt_trace_policies[policy + 'v2']['dates'] = [start_day]
                        adapt_trace_policies[policy + 'v2']['end_day'] = None
                        adapt_trace_policies[policy + 'v2']['coverage'] = [adapt_trace_policies[policy]['coverage'][-1]]  # Start coverage where it was when trace_app was first ended, not sure what best option is here
                        if not date_trigger:
                            policy_dates[policy].append(start_day)
        else:
            if policy in adapt_beta_policies:
                if len(new_pol_dates) > 1:
                    baseline_schedule.add(policy, new_pol_dates[0], new_pol_dates[1])
                    policy_dates[policy] = new_pol_dates
                    date_trigger = True
                else:
                    baseline_schedule.add(policy, new_pol_dates[0], n_days)
                    policy_dates[policy] = [new_pol_dates[0], n_days]
                    date_trigger = True
            if policy in import_policies:
                if policy not in policy_dates:  # had not previously been turned on (i.e no start date in spreadsheet)
                    all_imports = np.zeros(n_days)
                    all_imports[:len(i_cases)] = i_cases  # fill with provided data
                    if len(new_pol_dates) == 1:  # only start date
                        day_turnon = new_pol_dates[0]
                        day_turnoff = n_days
                    elif len(new_pol_dates) == 2:  # start and end date
                        day_turnon = new_pol_dates[0]
                        day_turnoff = new_pol_dates[1]
                    else:
                        raise Exception(f'Dates for turning on policy "{policy}" are in the wrong format.\n Current format: {new_pol_dates}')
                    days_after_turnon = np.arange(day_turnon, day_turnoff)
                    new_imports = [import_policies[policy]['n_imports']] * len(days_after_turnon)
                    all_imports[days_after_turnon] += new_imports  # add the number of imports
                    imports_dict = dict(days=np.arange(n_days), vals=all_imports)

                # these conditions below are a bit sketchy
                elif len(new_pol_dates) > 1:
                    imports_dict = dict(days=np.append(range(len(i_cases)), np.arange(policy_dates[policy][0], policy_dates[policy][1]) + np.arange(new_pol_dates[0], new_pol_dates[1])),
                                        vals=np.append(i_cases, [import_policies[policy]['n_imports']] * (policy_dates[policy][1] - policy_dates[policy][0]) + [import_policies[policy]['n_imports']] * (new_pol_dates[1]-new_pol_dates[0])))
                    if not date_trigger:
                        policy_dates[policy] = new_pol_dates
                        date_trigger = True
                else:
                    imports_dict = dict(days=np.append(range(len(i_cases)), np.arange(policy_dates[policy][0], policy_dates[policy][1]) + np.arange(new_pol_dates[0], n_days)),
                                        vals=np.append(i_cases, [import_policies[policy]['n_imports']] * (policy_dates[policy][1] - policy_dates[policy][0]) + [import_policies[policy]['n_imports']] * (n_days-new_pol_dates[0])))
                    if not date_trigger:
                        policy_dates[policy] = [new_pol_dates[0], n_days]
                        date_trigger = True
            if policy in clip_policies:
                adapt_clip_policies[policy + 'v2'] = sc.dcp(adapt_clip_policies[policy])
                if len(new_pol_dates) > 1:
                    adapt_clip_policies[policy + 'v2']['dates'] = new_pol_dates
                    if not date_trigger:
                        policy_dates[policy] = new_pol_dates
                else:
                    adapt_clip_policies[policy + 'v2']['dates'] = [new_pol_dates[0], n_days]
                    if not date_trigger:
                        policy_dates[policy] = [new_pol_dates[0], n_days]
            if policy in trace_policies:
                adapt_trace_policies[policy + 'v2'] = sc.dcp(adapt_trace_policies[policy])
                if len(new_pol_dates) > 1:
                    adapt_trace_policies[policy + 'v2']['start_day'] = start_day
                    adapt_trace_policies[policy + 'v2']['dates'] = [start_day]
                    adapt_trace_policies[policy + 'v2']['end_day'] = new_pol_dates[1]
                    adapt_trace_policies[policy + 'v2']['coverage'] = [adapt_trace_policies[policy]['coverage'][-1]] # Start coverage where it was when trace_app was first ended, not sure what best option is here
                    if not date_trigger:
                        policy_dates[policy] = new_pol_dates
                else:
                    adapt_trace_policies[policy + 'v2']['start_day'] = start_day
                    adapt_trace_policies[policy + 'v2']['dates'] = [start_day]
                    adapt_trace_policies[policy + 'v2']['end_day'] = None
                    adapt_trace_policies[policy + 'v2']['coverage'] = [adapt_trace_policies[policy]['coverage'][-1]]  # Start coverage where it was when trace_app was first ended, not sure what best option is here
                    if not date_trigger:
                        policy_dates[policy] = [start_day]
    return baseline_schedule, imports_dict, adapt_clip_policies, adapt_trace_policies, policy_dates


def replace_policies(scen, baseline_schedule, beta_policies, import_policies, clip_policies, trace_policies, i_cases, n_days, policy_dates, imports_dict):
    adapt_beta_policies = beta_policies
    adapt_clip_policies = clip_policies
    adapt_trace_policies = trace_policies
    for old_pol in scen['replace']:
        old_pol_dates = scen['replace'][old_pol]['dates']
        old_pol_reps = scen['replace'][old_pol]['replacements']
        old_date_trigger = False
        if old_pol in policy_dates:
            if len(policy_dates[old_pol]) % 2 == 0 and policy_dates[old_pol][-1] < old_pol_dates[0]:
                print('Not replacing policy %s at day %s because it is already off.' % (old_pol, str(old_pol_dates[0])))
            elif len(policy_dates[old_pol]) % 2 != 0 and policy_dates[old_pol][-1] > old_pol_dates[0]:
                print('Not replacing policy %s at day %s because it is already off. It will be turned on again on day %s' % (old_pol, str(old_pol_dates[0]), str(policy_dates[old_pol][-1])))
            else:
                if old_pol in beta_policies:
                    baseline_schedule.end(old_pol, old_pol_dates[0])
                    policy_dates[old_pol].append(old_pol_dates[0])
                    old_date_trigger = True
                if old_pol in import_policies:
                    imports_dict = dict(days=np.append(range(len(i_cases)), np.arange(policy_dates[old_pol][0], old_pol_dates[0]),
                                        vals=np.append(i_cases, [import_policies[old_pol]['n_imports']] * (old_pol_dates[0] - policy_dates[old_pol][0]))))
                    if not old_date_trigger:
                        policy_dates[old_pol].append(old_pol_dates[0])
                        old_date_trigger = True
                if old_pol in clip_policies:
                    adapt_clip_policies[old_pol]['dates'][1] = old_pol_dates[0]
                    if not old_date_trigger:
                        policy_dates[old_pol].append(old_pol_dates[0])
                        old_date_trigger = True
                if old_pol in trace_policies:
                    adapt_trace_policies[old_pol]['end_day'] = old_pol_dates[0]
                    adapt_trace_policies[old_pol]['coverage'] = [cov for c, cov in enumerate(adapt_trace_policies[old_pol]['coverage']) if adapt_trace_policies[old_pol]['dates'][c] < old_pol_dates[0]]
                    adapt_trace_policies[old_pol]['dates'] = [day for day in adapt_trace_policies[old_pol]['dates'] if day < old_pol_dates[0]]
                    if not old_date_trigger:
                        policy_dates[old_pol].append(old_pol_dates[0])
                for n, new_policy in enumerate(old_pol_reps):
                    date_trigger = False
                    if new_policy in policy_dates:
                        if len(policy_dates[new_policy]) % 2 != 0:
                            print('Not turning on policy %s as a replacement at day %s because it is already on.' % (new_policy, str(old_pol_dates[0])))
                        elif policy_dates[new_policy][-1] > old_pol_dates[0]:
                            print('Not turning on policy %s as a replacement at day %s because it is already on. It will be turned off again on day %s' % (new_policy, str(old_pol_dates[0]), str(policy_dates[new_policy][-1])))
                        else:
                            if new_policy in adapt_beta_policies:
                                if n == 0:
                                    if len(old_pol_reps) > 1:
                                        baseline_schedule.add(new_policy, old_pol_dates[n])
                                        policy_dates[new_policy].append(old_pol_dates[n])
                                        date_trigger = True
                                    elif n == len(old_pol_reps) - 1 and len(old_pol_reps) < len(old_pol_dates):
                                        baseline_schedule.add(new_policy, old_pol_dates[n], old_pol_dates[n+1])
                                        policy_dates[new_policy].extend([old_pol_dates[n], old_pol_dates[n+1]])
                                        date_trigger = True
                                    else:
                                        baseline_schedule.add(new_policy, old_pol_dates[n])
                                        policy_dates[new_policy].append(old_pol_dates[n])
                                        date_trigger = True
                                else:
                                    baseline_schedule.end(old_pol_reps[n - 1], old_pol_dates[n])
                                    policy_dates[old_pol_reps[n - 1]].append(old_pol_dates[n])
                                    if n == len(old_pol_reps) - 1 and len(old_pol_reps) < len(old_pol_dates):
                                        baseline_schedule.add(new_policy, old_pol_dates[n], old_pol_dates[n+1])
                                        policy_dates[new_policy].extend([old_pol_dates[n], old_pol_dates[n+1]])
                                        date_trigger = True
                                    else:
                                        baseline_schedule.add(new_policy, old_pol_dates[n])
                                        policy_dates[new_policy].append(old_pol_dates[n])
                                        date_trigger = True
                            if new_policy in import_policies:
                                if len(old_pol_dates[n:]) > 1:
                                    imports_dict = dict(days=np.append(range(len(i_cases)), np.arange(policy_dates[new_policy][0], policy_dates[new_policy][1]) + np.arange(old_pol_dates[n], old_pol_dates[n+1])),
                                                        vals=np.append(i_cases, [import_policies[new_policy]['n_imports']] * (policy_dates[new_policy][1] - policy_dates[new_policy][0]) + [import_policies[new_policy]['n_imports']] * (old_pol_dates[n+1]-old_pol_dates[n])))
                                    if not date_trigger:
                                        policy_dates[new_policy].extend([old_pol_dates[n], old_pol_dates[n+1]])
                                        date_trigger = True
                                else:
                                    imports_dict = dict(days=np.append(range(len(i_cases)), np.arange(policy_dates[new_policy][0], policy_dates[new_policy][1]) + np.arange(old_pol_dates[n], n_days)),
                                                        vals=np.append(i_cases, [import_policies[new_policy]['n_imports']] * (policy_dates[new_policy][1] - policy_dates[new_policy][0]) + [import_policies[new_policy]['n_imports']] * (n_days-old_pol_dates[n])))
                                    if not date_trigger:
                                        policy_dates[new_policy].extend([old_pol_dates[n], n_days])
                                        date_trigger = True
                            if new_policy in adapt_clip_policies:
                                if n != 0:
                                    adapt_clip_policies[old_pol_reps[n - 1] + 'v2'][1] = old_pol_dates[n]
                                adapt_clip_policies[new_policy + 'v2'] = sc.dcp(adapt_clip_policies[new_policy])
                                if len(old_pol_dates) > 1:
                                    adapt_clip_policies[new_policy + 'v2']['dates'] = [old_pol_dates[n], old_pol_dates[n+1]]
                                    if not date_trigger:
                                        policy_dates[new_policy].extend([old_pol_dates[n], old_pol_dates[n+1]])
                                        date_trigger = True
                                else:
                                    adapt_clip_policies[new_policy + 'v2']['dates'] = [old_pol_dates[n], n_days]
                                    if not date_trigger:
                                        policy_dates[new_policy].extend([old_pol_dates[n], n_days])
                                        date_trigger = True
                            if new_policy in adapt_trace_policies:
                                if n != 0:
                                    adapt_trace_policies[old_pol_reps[n - 1] + 'v2'][1] = old_pol_dates[n]
                                adapt_trace_policies[new_policy + 'v2'] = sc.dcp(adapt_trace_policies[new_policy])
                                if len(old_pol_dates) > 1:
                                    adapt_trace_policies[new_policy + 'v2']['start_day'] = old_pol_dates[n]
                                    adapt_trace_policies[new_policy + 'v2']['end_day'] = old_pol_dates[n+1]
                                    adapt_trace_policies[new_policy + 'v2']['dates'] = [old_pol_dates[n]]
                                    adapt_trace_policies[new_policy + 'v2']['coverage'] = [adapt_trace_policies[new_policy]['coverage'][-1]] # Start coverage where it was when trace_app was first ended, not sure what best option is here
                                    if not date_trigger:
                                        policy_dates[new_policy].extend([old_pol_dates[n], old_pol_dates[n+1]])
                                else:
                                    adapt_trace_policies[new_policy + 'v2']['start_day'] = old_pol_dates[n]
                                    adapt_trace_policies[new_policy + 'v2']['end_day'] = None
                                    adapt_trace_policies[new_policy + 'v2']['dates'] = [old_pol_dates[n]]
                                    adapt_trace_policies[new_policy + 'v2']['coverage'] = [adapt_trace_policies[new_policy]['coverage'][-1]]  # Start coverage where it was when trace_app was first ended, not sure what best option is here
                                    if not date_trigger:
                                        policy_dates[new_policy].append(old_pol_dates[n])
                    else:
                        if new_policy in adapt_beta_policies:
                            if n == 0:
                                if len(old_pol_reps) > 1:
                                    baseline_schedule.add(new_policy, old_pol_dates[n])
                                    policy_dates[new_policy] = [old_pol_dates[n]]
                                    date_trigger = True
                                elif n == len(old_pol_reps) - 1 and len(old_pol_reps) < len(old_pol_dates):
                                    baseline_schedule.add(new_policy, old_pol_dates[n], old_pol_dates[n + 1])
                                    policy_dates[new_policy] = [old_pol_dates[n], old_pol_dates[n + 1]]
                                    date_trigger = True
                                else:
                                    baseline_schedule.add(new_policy, old_pol_dates[n])
                                    policy_dates[new_policy] = [old_pol_dates[n]]
                                    date_trigger = True
                            else:
                                baseline_schedule.end(old_pol_reps[n - 1], old_pol_dates[n])
                                policy_dates[old_pol_reps[n - 1]].append(old_pol_dates[n])
                                if n == len(old_pol_reps) - 1 and len(old_pol_reps) < len(old_pol_dates):
                                    baseline_schedule.add(new_policy, old_pol_dates[n], old_pol_dates[n + 1])
                                    policy_dates[new_policy] = [old_pol_dates[n], old_pol_dates[n + 1]]
                                    date_trigger = True
                                else:
                                    baseline_schedule.add(new_policy, old_pol_dates[n])
                                    policy_dates[new_policy] = [old_pol_dates[n]]
                                    date_trigger = True
                        if new_policy in import_policies:
                            if len(old_pol_dates[n:]) > 1:
                                imports_dict = dict(days=np.append(range(len(i_cases)), np.arange(policy_dates[new_policy][0], policy_dates[new_policy][1]) + np.arange(old_pol_dates[n], old_pol_dates[n+1])),
                                                    vals=np.append(i_cases, [import_policies[new_policy]['n_imports']] * (policy_dates[new_policy][1] - policy_dates[new_policy][0]) + [import_policies[new_policy]['n_imports']] * (old_pol_dates[n+1]-old_pol_dates[n])))
                                if not date_trigger:
                                    policy_dates[new_policy] = [old_pol_dates[n], old_pol_dates[n+1]]
                                    date_trigger = True
                            else:
                                imports_dict = dict(days=np.append(range(len(i_cases)), np.arange(policy_dates[new_policy][0], policy_dates[new_policy][1]) + np.arange(old_pol_dates[n], n_days)),
                                                    vals=np.append(i_cases, [import_policies[new_policy]['n_imports']] * (policy_dates[new_policy][1] - policy_dates[new_policy][0]) + [import_policies[new_policy]['n_imports']] * (n_days-old_pol_dates[n])))
                                if not date_trigger:
                                    policy_dates[new_policy] = [old_pol_dates[n], n_days]
                                    date_trigger = True
                        if new_policy in adapt_clip_policies:
                            if n != 0:
                                adapt_clip_policies[old_pol_reps[n - 1] + 'v2'][1] = old_pol_dates[n]
                            adapt_clip_policies[new_policy + 'v2'] = sc.dcp(adapt_clip_policies[new_policy])
                            if len(old_pol_dates) > 1:
                                adapt_clip_policies[new_policy + 'v2']['dates'] = [old_pol_dates[n], old_pol_dates[n+1]]
                                if not date_trigger:
                                    policy_dates[new_policy] = [old_pol_dates[n], old_pol_dates[n+1]]
                            else:
                                adapt_clip_policies[new_policy + 'v2']['dates'] = [old_pol_dates[n], n_days]
                                if not date_trigger:
                                    policy_dates[new_policy] = [old_pol_dates[n], n_days]
                        if new_policy in adapt_trace_policies:
                            if n != 0:
                                adapt_trace_policies[old_pol_reps[n - 1] + 'v2'][1] = old_pol_dates[n]
                            adapt_trace_policies[new_policy + 'v2'] = sc.dcp(adapt_trace_policies[new_policy])
                            if len(old_pol_dates) > 1:
                                adapt_trace_policies[new_policy + 'v2']['start_day'] = old_pol_dates[n]
                                adapt_trace_policies[new_policy + 'v2']['end_day'] = old_pol_dates[n+1]
                                adapt_trace_policies[new_policy + 'v2']['dates'] = [old_pol_dates[n]]
                                adapt_trace_policies[new_policy + 'v2']['coverage'] = [adapt_trace_policies[new_policy]['coverage'][-1]] # Start coverage where it was when trace_app was first ended, not sure what best option is here
                                if not date_trigger:
                                    policy_dates[new_policy] = [old_pol_dates[n], old_pol_dates[n+1]]
                            else:
                                adapt_trace_policies[new_policy + 'v2']['start_day'] = old_pol_dates[n]
                                adapt_trace_policies[new_policy + 'v2']['end_day'] = None
                                adapt_trace_policies[new_policy + 'v2']['dates'] = [old_pol_dates[n]]
                                adapt_trace_policies[new_policy + 'v2']['coverage'] = [adapt_trace_policies[new_policy]['coverage'][-1]]  # Start coverage where it was when trace_app was first ended, not sure what best option is here
                                if not date_trigger:
                                    policy_dates[new_policy] = [old_pol_dates[n]]
        else:
            print('Policy %s could not be replaced because it is not running.' % old_pol)
    return baseline_schedule, imports_dict, adapt_clip_policies, adapt_trace_policies, policy_dates


def make_tracing(trace_policies):
    if trace_policies.get('tracing_app') is not None:
        t_details = trace_policies['tracing_app']
        tracing_app = AppBasedTracing(name='tracing_app',
                                      days=t_details['days'],
                                      coverage=t_details['coverage'],
                                      layers=t_details['layers'],
                                      start_day=t_details['start_day'],
                                      end_day=t_details['end_day'],
                                      trace_time=t_details['trace_time'])
    else:
        tracing_app = None
    if trace_policies.get('id_checks') is not None:
        id_details = trace_policies['id_checks']
        id_checks = AppBasedTracing(name='id_checks',
                                    days=id_details['days'],
                                    coverage=id_details['coverage'],
                                    layers=id_details['layers'],
                                    start_day=id_details['start_day'],
                                    end_day=id_details['end_day'],
                                    trace_time=id_details['trace_time'])
    else:
        id_checks = None

    return tracing_app, id_checks