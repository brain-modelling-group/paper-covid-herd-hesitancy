"""
Analyzer to keep track of vax efficacy over time
Custom made for Hesitancy_SingleCorRun.py


Usage:
 sim_hesitance = cv.Sim(pars = pars, popfile=popfile, load_pop=True, analyzers=store_efficacy())


after sim is finished

ana = sim.pars['analyzers'][0]
ana.plot()
plt.show()

"""

from covasim.immunity import nab_to_efficacy
import sciris as sc
import covasim as cv
import matplotlib.pyplot as plt


class store_efficacy(cv.Analyzer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # This is necessary to initialize the class properly
        self.t = []
        self.eff_sus = []
        self.eff_symp = []
        self.eff_sev = []
        return

    def apply(self, sim):
        ppl = sim.people # Shorthand
        self.t.append(sim.t)

        nabs = ppl.nab
        # 
        vax_pars = sim.pars['interventions'][2].p['nab_eff']
        
        eff_susc = nab_to_efficacy(nabs, 'sus', vax_pars)
        self.eff_sus.append(eff_susc.mean())

        eff_symp = nab_to_efficacy(nabs, 'symp', vax_pars)
        self.eff_symp.append(eff_symp.mean())
        
        eff_sev = nab_to_efficacy(nabs, 'sev', vax_pars)
        self.eff_sev.append(eff_sev.mean())
        return

    def plot(self):
        plt.figure()
        plt.plot(self.t, self.eff_sus, label='eff_sus', color='blue')
        plt.plot(self.t, self.eff_symp, label='eff_symp', color='red')
        plt.plot(self.t, self.eff_sev, label='eff_sev', color='green')
        plt.legend()
        plt.xlabel('Day')
        plt.ylabel('vax_efficacy')
        sc.setylim() # Reset y-axis to start at 0
        sc.commaticks() # Use commas in the y-axis labels
        return