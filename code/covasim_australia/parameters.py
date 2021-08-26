class Parameters:
    """
    The Parameters class is a container for the various types of parameters
    required for an application of Covasim.
    Some parameter attributes are used directly by Covasim while others
    are used both prior to and during the simulation to provide greater
    resolution than Covasim currently provides.

    Args:
        pars (dict): parameters used directly by Covasim
        metapars (dict): meta-parameters used directly by Covasim
        extrapars (dict): additional parameters that allow greater layer resolution
        layerchars (dict): network characteristics that determine from which subset of the population the layers are generated
    """

    def __init__(self,
                 location=None,
                 pars=None,
                 metapars=None,
                 extrapars=None,
                 layerchars=None,
                 policies=None,
                 household_dist=None,
                 age_dist=None,
                 contact_matrix=None,
                 imported_cases=None,
                 daily_tests=None,
                 beta_vals=None,
                 all_lkeys=None,
                 default_lkeys=None,
                 custom_lkeys=None,
                 dynamic_lkeys=None):

        self.location = location
        self.pars = pars
        self.metapars = metapars
        self.extrapars = extrapars
        self.layerchars = layerchars
        self.policies = policies
        self.household_dist = household_dist
        self.age_dist = age_dist
        self.contact_matrix = contact_matrix
        self.imported_cases = imported_cases
        self.daily_tests = daily_tests
        self.beta_vals = beta_vals
        self.all_lkeys = all_lkeys
        self.default_lkeys = default_lkeys
        self.custom_lkeys = custom_lkeys
        self.dynamic_lkeys = dynamic_lkeys

    def update_pars(self, newpars, verbose=0):
        """Update values in self.pars with those in newpars"""
        if newpars is None:
            return

        if verbose:
            print(f'The following will be updated in the parameters dictionary...')
            for key in newpars.keys():
                print(f'- {key}')

        self.pars.update(newpars)
        return

    def update_metapars(self, new_metapars, verbose=0):
        """Update values in self.metapars with those in new_metapars"""
        if new_metapars is None:
            return

        if verbose:
            print("The following will be updated in the meta-parameters dictionary...")
            for key in new_metapars.keys():
                print(f'- {key}')

        self.metapars.update(new_metapars)
        return

    def print_pars(self):
        print("--- PARAMETERS ---")
        for key, value in sorted(self.pars.items()):
            print(f'- {key}: {value}')

    def print_metapars(self):
        print("--- METAPARAMETERS ---")
        for key, value in sorted(self.metapars.items()):
            print(f'- {key}: {value}')


def setup_params(location=None, loc_data=None, metapars=None, sim_pars=None):
    """Read in the required parameter types and put in container
    :return a Parameters container object"""

    pars = loc_data['pars']
    extrapars = loc_data['extrapars']
    layerchars = loc_data['layerchars']
    policies = loc_data['policies']
    household_dist = loc_data['household_dist']
    age_dist = loc_data['age_dist']
    contact_matrix = loc_data['contact_matrix']
    imported_cases = loc_data['imported_cases']
    daily_tests = loc_data['daily_tests']

    all_lkeys = loc_data['all_lkeys']
    default_lkeys = loc_data['default_lkeys']
    custom_lkeys = loc_data['custom_lkeys']
    dynamic_lkeys = loc_data['dynamic_lkeys']

    params = Parameters(location=location,
                        pars=pars,
                        metapars=metapars,
                        extrapars=extrapars,
                        layerchars=layerchars,
                        policies=policies,
                        household_dist=household_dist,
                        age_dist=age_dist,
                        contact_matrix=contact_matrix,
                        imported_cases=imported_cases,
                        daily_tests=daily_tests,
                        all_lkeys=all_lkeys,
                        default_lkeys=default_lkeys,
                        custom_lkeys=custom_lkeys,
                        dynamic_lkeys=dynamic_lkeys)

    # if specified by user, overwrite sim parameter values from the databook
    if sim_pars is not None:
        params.update_pars(sim_pars)
    return params
