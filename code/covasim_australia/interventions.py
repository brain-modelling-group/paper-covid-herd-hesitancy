#!/usr/bin/env python
import numpy as np
import sciris as sc

# import utils, policy_updates
import covasim.defaults as cvd
import covasim.interventions as cvi
from collections import defaultdict

"""
Modified Class, derived from vaccine_num().
Has a parameter that will prioritise first doses over
scheduled doses.

"""


class vaccinate_num_priority(cvi.BaseVaccination):
    def __init__(self, vaccine, num_doses, priority_prop = [0.0, 1.0], sequence=None, **kwargs):
        """
        Sequence-based vaccination

        This vaccine intervention allocates vaccines in a pre-computed order of
        distribution, at a specified rate of doses per day. Second doses are prioritized
        each day.

        Args:
            vaccine (dict/str): which vaccine to use; see below for dict parameters
            label        (str): if vaccine is supplied as a dict, the name of the vaccine
            priority_prop (list/array): a two  element list/array with numbers between 0 and 1.0 
                                        first number represents the percentage of doses in num_doses that 
                                        are saved for "first doses", if there are any eleigible left
                                        second number represent the percentage of doses in num_doses
                                        that are saved for "scheduled doses" .
                                        Default values should reproduce beahviour of vaccinate_num()
            sequence: Specify the order in which people should get vaccinated. This can be

                - An array of person indices in order of vaccination priority
                - A callable that takes in `cv.People` and returns an ordered sequence. For example, to
                  vaccinate people in descending age order, ``def age_sequence(people): return np.argsort(-people.age)``
                  would be suitable.
                  If not specified, people will be randomly ordered.
            num_doses: Specify the number of doses per day. This can take three forms

                - A scalar number of doses per day
                - A dict keyed by day/date with the number of doses e.g. ``{2:10000, '2021-05-01':20000}``.
                  Any dates are convered to simulation days in `initialize()` which will also copy the
                  dictionary passed in.
                - A callable that takes in a ``cv.Sim`` and returns a scalar number of doses. For example,
                  ``def doses(sim): return 100 if sim.t > 10 else 0`` would be suitable
            **kwargs: Additional arguments passed to ``cv.BaseVaccination``

        **Example**::

            def age_sequence(people): return np.argsort(-people.age)
            pfizer = cv.vaccinate_num(vaccine='pfizer', sequence=age_sequence, num_doses=100)
            cv.Sim(interventions=pfizer, use_waning=True).run().plot()

        """
        super().__init__(vaccine,**kwargs) # Initialize the Intervention object
        self.sequence = sequence
        self.num_doses = num_doses
        self.priority_prop = priority_prop
        self._scheduled_doses = defaultdict(set)  # Track scheduled second doses
        return

    def initialize(self, sim):
        super().initialize(sim)

        # Convert any dates to simulation days
        if isinstance(self.num_doses, dict):
            self.num_doses = {sim.day(k):v for k, v in self.num_doses.items()}

        if callable(self.sequence):
            self.sequence = self.sequence(sim.people)
        elif self.sequence is None:
            self.sequence = np.random.permutation(sim.n)
        else:
            self.sequence = sc.promotetoarray(self.sequence)

        if self.p['doses'] > 2:
            raise NotImplementedError('Scheduling three or more doses not yet supported')

        return

    def select_people(self, sim):

        # Work out how many people to vaccinate today
        if sc.isnumber(self.num_doses):
            num_people_tovax = self.num_doses
        elif callable(self.num_doses):
            num_people_tovax = self.num_doses(sim)
        elif sim.t in self.num_doses:
            num_people_tovax = self.num_doses[sim.t]
        else:
            # If nobody gets vaccinated today, just return an empty list
            return np.array([])
        
        #Work out priority prop for 1st vs 2nd doses:
        if sc.checktype(self.priority_prop, 'arraylike', 'number') or sc.checktype(self.priority_prop, 'array', 'number'):
            priority_prop = self.priority_prop
        elif callable(self.priority_prop):
            priority_prop = self.priority_prop(sim)
        elif sim.t in self.priority_prop:
            priority_prop = self.priority_prop[sim.t]

        num_agents_tovax = int(np.round(num_people_tovax / sim["pop_scale"]))
        num_agents_first_vax = int(np.round((num_people_tovax / sim["pop_scale"])*priority_prop[0]))
        num_agents_sched_vax = int(np.round((num_people_tovax / sim["pop_scale"])*priority_prop[1]))

        # First, see how many scheduled doses we are going to deliver
        if self._scheduled_doses[sim.t]:
            scheduled = np.fromiter(self._scheduled_doses[sim.t], dtype=cvd.default_int) # Everyone scheduled today
            scheduled = scheduled[(sim.people.vaccinations[scheduled]<self.p['doses']) & ~sim.people.dead[scheduled]] # Remove fully vaccinated or dead

            # If there are more people due for a second dose than there are doses, vaccinate as many second doses
            # as possible, and add the remainder to tomorrow's vaccinations. At the moment, they don't get priority
            # because the order of the scheduling doesn't matter (so there is a chance someone could go for several days
            # before being allocated their second dose) but then there is some flexibility in the dosing schedules anyway
            # e.g. Pfizer being 3-6 weeks in some jurisdictions
            if len(scheduled) > num_agents_sched_vax:
                np.random.shuffle(scheduled) # Randomly pick who to defer
                self._scheduled_doses[sim.t+1].update(scheduled[num_agents_sched_vax:]) # Defer any extras
                return scheduled[:num_agents_sched_vax]
        else:
            scheduled = np.array([], dtype=cvd.default_int)

        # Next, work out who is eligible for a first dose
        # Anyone who has received at least one dose of a vaccine would have had subsequent doses scheduled
        # and therefore should not be selected here
        first_dose_eligible = self.sequence[~sim.people.vaccinated[self.sequence] & ~sim.people.dead[self.sequence]]

        if len(first_dose_eligible) == 0:
            return scheduled  # Just return anyone that is scheduled
        elif len(first_dose_eligible) > num_agents_first_vax:
            # Truncate it to the number of agents for performance when checking whether anyone scheduled overlaps with first doses to allocate
            first_dose_eligible = first_dose_eligible[:num_agents_first_vax] # This is the maximum number of people we could vaccinate this timestep, if there are no second doses allocated

        # It's *possible* that someone has been *scheduled* for a first dose by some other mechanism externally
        # Therefore, we need to check and remove them from the first dose list, otherwise they could be vaccinated
        # twice here (which would amount to wasting a dose)
        first_dose_eligible = first_dose_eligible[~np.in1d(first_dose_eligible, scheduled)]

        if (len(first_dose_eligible)) > num_agents_first_vax:
            first_dose_inds = first_dose_eligible[:num_agents_first_vax]
        else:
            first_dose_inds = first_dose_eligible
        # Schedule subsequent doses
        # For vaccines with >2 doses, scheduled doses will also need to be checked
        if self.p['doses'] > 1:
            self._scheduled_doses[sim.t+self.p['interval']].update(first_dose_inds)

        # NOTE: TODO: to change behaviour depending on the fix made to BaseVaccination class.
        vacc_inds = np.concatenate([scheduled, first_dose_inds])
        # Agents vaccinated at this time step
        self.vaccinated[sim.t] = vacc_inds
        # New agents vaccinated, that were unvaccinated before
        sim.people.flows['new_vaccinated'] += len(first_dose_inds)
        self._scheduled_doses[sim.t+self.p['interval']].update(first_dose_inds)

        return vacc_inds


class vaccinate_num_base(cvi.BaseVaccination):
    def __init__(self, vaccine, num_doses, sequence=None, **kwargs):
        """
        Sequence-based vaccination

        This vaccine intervention allocates vaccines in a pre-computed order of
        distribution, at a specified rate of doses per day. Second doses are prioritized
        each day.

        Args:
            vaccine (dict/str): which vaccine to use; see below for dict parameters
            label        (str): if vaccine is supplied as a dict, the name of the vaccine
            sequence: Specify the order in which people should get vaccinated. This can be

                - An array of person indices in order of vaccination priority
                - A callable that takes in `cv.People` and returns an ordered sequence. For example, to
                  vaccinate people in descending age order, ``def age_sequence(people): return np.argsort(-people.age)``
                  would be suitable.
                  If not specified, people will be randomly ordered.
            num_doses: Specify the number of doses per day. This can take three forms

                - A scalar number of doses per day
                - A dict keyed by day/date with the number of doses e.g. ``{2:10000, '2021-05-01':20000}``.
                  Any dates are convered to simulation days in `initialize()` which will also copy the
                  dictionary passed in.
                - A callable that takes in a ``cv.Sim`` and returns a scalar number of doses. For example,
                  ``def doses(sim): return 100 if sim.t > 10 else 0`` would be suitable
            **kwargs: Additional arguments passed to ``cv.BaseVaccination``

        **Example**::

            def age_sequence(people): return np.argsort(-people.age)
            pfizer = cv.vaccinate_num(vaccine='pfizer', sequence=age_sequence, num_doses=100)
            cv.Sim(interventions=pfizer, use_waning=True).run().plot()

        """
        super().__init__(vaccine,**kwargs) # Initialize the Intervention object
        self.sequence = sequence
        self.num_doses = num_doses
        self._scheduled_doses = defaultdict(set)  # Track scheduled second doses
        return

    def initialize(self, sim):
        super().initialize(sim)

        # Convert any dates to simulation days
        if isinstance(self.num_doses, dict):
            self.num_doses = {sim.day(k):v for k, v in self.num_doses.items()}

        if callable(self.sequence):
            self.sequence = self.sequence(sim.people)
        elif self.sequence is None:
            self.sequence = np.random.permutation(sim.n)
        else:
            self.sequence = sc.promotetoarray(self.sequence)

        if self.p['doses'] > 2:
            raise NotImplementedError('Scheduling three or more doses not yet supported')

        return

    def select_people(self, sim):

        # Work out how many people to vaccinate today
        if sc.isnumber(self.num_doses):
            num_people = self.num_doses
        elif callable(self.num_doses):
            num_people = self.num_doses(sim)
        elif sim.t in self.num_doses:
            num_people = self.num_doses[sim.t]
        else:
            # If nobody gets vaccinated today, just return an empty list
            return np.array([])

        num_agents = int(np.round(num_people / sim["pop_scale"]))

        # First, see how many scheduled doses we are going to deliver
        #import pdb; pdb.set_trace()
        if self._scheduled_doses[sim.t]:
            scheduled = np.fromiter(self._scheduled_doses[sim.t], dtype=cvd.default_int) # Everyone scheduled today
            scheduled = scheduled[(sim.people.vaccinations[scheduled]<self.p['doses']) & ~sim.people.dead[scheduled]] # Remove fully vaccinated or dead

            # If there are more people due for a second dose than there are doses, vaccinate as many second doses
            # as possible, and add the remainder to tomorrow's vaccinations. At the moment, they don't get priority
            # because the order of the scheduling doesn't matter (so there is a chance someone could go for several days
            # before being allocated their second dose) but then there is some flexibility in the dosing schedules anyway
            # e.g. Pfizer being 3-6 weeks in some jurisdictions
            if len(scheduled) > num_agents:
                np.random.shuffle(scheduled) # Randomly pick who to defer
                self._scheduled_doses[sim.t+1].update(scheduled[num_agents:]) # Defer any extras
                vacc_inds = scheduled[:num_agents]
                print("Delivering scheduled doses")
                return vacc_inds
        else:
            scheduled = np.array([], dtype=cvd.default_int)

        # Next, work out who is eligible for a first dose
        # Anyone who has received at least one dose of a vaccine would have had subsequent doses scheduled
        # and therefore should not be selected here
        first_dose_eligible = self.sequence[~sim.people.vaccinated[self.sequence] & ~sim.people.dead[self.sequence]]

        if len(first_dose_eligible) == 0:
            return scheduled  # Just return anyone that is scheduled
        elif len(first_dose_eligible) > num_agents:
            # Truncate it to the number of agents for performance when checking whether anyone scheduled overlaps with first doses to allocate
            first_dose_eligible = first_dose_eligible[:num_agents] # This is the maximum number of people we could vaccinate this timestep, if there are no second doses allocated

        # It's *possible* that someone has been *scheduled* for a first dose by some other mechanism externally
        # Therefore, we need to check and remove them from the first dose list, otherwise they could be vaccinated
        # twice here (which would amount to wasting a dose)
        first_dose_eligible = first_dose_eligible[~np.in1d(first_dose_eligible, scheduled)]

        if (len(first_dose_eligible)+len(scheduled)) > num_agents:
            first_dose_inds = first_dose_eligible[:(num_agents - len(scheduled))]
        else:
            first_dose_inds = first_dose_eligible

        # NOTE: TODO: to change behaviour depending on the fix made to BaseVaccination class.
        vacc_inds = np.concatenate([scheduled, first_dose_inds])
        # Agents vaccinated at this time step
        self.vaccinated[sim.t] = vacc_inds
        # New agents vaccinated, that were unvaccinated before
        print(len(first_dose_inds))
        sim.people.flows['new_vaccinated'] += len(first_dose_inds)
        self._scheduled_doses[sim.t+self.p['interval']].update(first_dose_inds)

        return vacc_inds