import covasim_australia.clusters as cl
import collections
import covasim as cv
import covasim.defaults as cvd
import covasim.utils as cvu
import numpy as np
import sciris as sc
import numba as nb

def clusters_to_contacts(clusters):
    """
    Convert clusters to contacts

    cluster of people [1,2,3] would result in contacts
        1: [2,3]
        2: [1,3]
        3: [1,2]

    """
    contacts = collections.defaultdict(set)
    for cluster in clusters:
        for i in cluster:
            for j in cluster:
                if j <= i:
                    pass
                else:
                    contacts[i].add(j)
                    contacts[j].add(i)

    return {x: np.array(list(y)) for x, y in contacts.items()}


@nb.njit
def _get_contacts(include_inds, number_of_contacts):
    total_number_of_half_edges = np.sum(number_of_contacts)

    count = 0
    source = np.zeros((total_number_of_half_edges,), dtype=cvd.default_int)
    for i, person_id in enumerate(include_inds):
        n_contacts = number_of_contacts[i]
        source[count:count+n_contacts] = person_id
        count += n_contacts
    target = np.random.permutation(source)

    return source, target


def make_random_contacts(include_inds, mean_number_of_contacts, dispersion=None, array_output=False):
    """
    Makes the random contacts either by sampling the number of contacts per person from a Poisson or Negative Binomial distribution


    Parameters
    ----------
    include (array) array of person indexes (IDs) of everyone eligible for contacts in this layer
    mean_number_of_contacts (int) representing the mean number of contacts for each person
    dispersion (float) if not None, use a negative binomial distribution with this dispersion parameter instead of Poisson to make the contacts
    array_output (boolean) return contacts as arrays or as dicts

    Returns
    -------
    If array_output=False, return a contacts dictionary {1:[2,3,4],2:[1,5,6]} with keys for source person,
        and a values being a list of target contacts.

    If array_output=True, return arrays with `source` and `target` indexes. These could be interleaved to produce an edge list
        representation of the edges

    """

    n_people = len(include_inds)

    # sample the number of edges from a given distribution
    if dispersion is None:
        number_of_contacts = cvu.n_poisson(rate=mean_number_of_contacts, n=n_people)
    else:
        number_of_contacts = cvu.n_neg_binomial(rate=mean_number_of_contacts, dispersion=dispersion, n=n_people)

    source, target = _get_contacts(include_inds, number_of_contacts)

    if array_output:
        return source, target
    else:
        contacts = {}
        count = 0
        for i, person_id in enumerate(include_inds):
            n_contacts = number_of_contacts[i]
            contacts[person_id] = target[count:count+n_contacts]
            count += n_contacts
        return contacts


def make_hcontacts(n_households, pop_size, household_heads, uids, contact_matrix):
    """

    :param n_households:
    :param pop_size:
    :param household_heads:
    :return:
    """
    h_clusters, h_ages = cl.make_household_clusters(n_households, pop_size, household_heads, uids, contact_matrix)
    h_contacts = clusters_to_contacts(h_clusters)
    return h_contacts, h_ages


def make_scontacts(uids, ages, s_contacts):
    """Create school contacts, with children of each age clustered in groups"""
    class_cl = cl.make_sclusters(uids, ages, s_contacts)
    class_co = clusters_to_contacts(class_cl)
    return class_co


def make_lo_high_wcontacts(uids, ages, w_contacts, prop_high_risk):
    work_cl = cl.make_wclusters(uids, ages, w_contacts)

    is_high_risk = np.random.random(len(work_cl)) < prop_high_risk
    low_risk_cl = []
    high_risk_cl = []
    n_high_risk = 0
    n_low_risk = 0
    for cluster, high_risk in zip(work_cl, is_high_risk):
        if high_risk:
            high_risk_cl.append(cluster)
            n_high_risk += len(cluster)
        else:
            low_risk_cl.append(cluster)
            n_low_risk += len(cluster)

    print(f'Input high risk proportion = {prop_high_risk}')
    print(f'Assigned proportion high risk workplaces = {len(high_risk_cl)/len(work_cl):.4f}')
    print(f'Assigned proportion high risk workers = {n_high_risk/(n_low_risk+n_high_risk):.4f}')

    return clusters_to_contacts(low_risk_cl), clusters_to_contacts(high_risk_cl)


def make_wcontacts(uids, ages, w_contacts):
    work_cl = cl.make_wclusters(uids, ages, w_contacts)
    work_co = clusters_to_contacts(work_cl)
    return work_co    

def make_custom_contacts(uids, n_contacts, pop_size, ages, custom_lkeys, cluster_types, dispersion, pop_proportion, age_lb, age_ub):
    contacts = {}
    layer_members = {}
    for layer_key in custom_lkeys:
        cl_type = cluster_types[layer_key]
        num_contacts = n_contacts[layer_key]
        # get the uid of people in the layer
        n_people = int(pop_proportion[layer_key] * pop_size)
        # randomly choose people from right age
        agel = age_lb[layer_key]
        ageu = age_ub[layer_key]
        inds = np.random.choice(uids[(ages > agel) & (ages <= ageu)], n_people)
        layer_members[layer_key] = inds

        # handle the cluster types differently
        if cl_type == 'complete':   # number of contacts not used for complete clusters
            contacts[layer_key] = clusters_to_contacts([inds])
        elif cl_type == 'random':
            contacts[layer_key] = make_random_contacts(include_inds=inds, mean_number_of_contacts=num_contacts, dispersion=dispersion[layer_key])
            # contacts[layer_key] = random_contacts(in_layer, num_contacts)
        elif cl_type == 'cluster':
            miniclusters = []
            miniclusters.extend(cl.create_clustering(inds, num_contacts))
            contacts[layer_key] = clusters_to_contacts(miniclusters)
        else:
            raise Exception(f'Error: Unknown network structure: {cl_type}')

    return contacts, layer_members


def convert_contacts(contacts, uids, all_lkeys):
    """ Convert contacts structure to be compatible with Covasim

    :return a list of length pop_size, where each entry is a dictionary by layer,
            and each dictionary entry is the UIDs of the agent's contacts"""
    contacts_list = [None] * len(uids)
    for uid in uids:
        contacts_list[uid] = {}
        for layer_key in all_lkeys:
            layer_contacts = contacts[layer_key]
            if layer_contacts.get(uid) is not None:
                contacts_list[uid][layer_key] = layer_contacts[uid]
            else:
                contacts_list[uid][layer_key] = np.empty(0)
    return contacts_list


def get_uids(pop_size):
    people_id = np.arange(start=0, stop=pop_size, step=1)
    return people_id


def get_numhouseholds(household_dist, pop_size):
    """Calculates the number of households we need to create to meet the population size"""
    n_people = sum(household_dist.index * household_dist)  # n_people = household_size * n_households
    household_percent = household_dist / n_people
    n_households = (pop_size * household_percent).round().astype(int)
    n_households[1] += pop_size - sum(n_households * n_households.index)  # adjust single-person households to fill the gap
    return n_households


def get_household_heads(age_dist, n_households):
    """Selects the ages of the household heads by randomly selecting from the available ages"""
    # prevent anyone under the age of 18 being chosen
    age_dist.iloc[0:18] = 0
    # decrease probability of someone aged 18-28 being chosen
    age_dist.iloc[18:28] *= np.linspace(0.1, 1, 10)

    # randomly choose household head, given the number of people in each age
    age_prob = age_dist.values / sum(age_dist.values)
    household_heads = np.random.choice(age_dist.index, size=sum(n_households), p=age_prob)
    return household_heads


def make_lo_hi_contacts(params):
    contacts = {}

    pop_size = params.pars['pop_size']
    household_dist = params.household_dist
    age_dist = params.age_dist
    contact_matrix = params.contact_matrix
    n_contacts = params.pars['contacts']
    all_lkeys = params.all_lkeys

    # for custom layers
    custom_lkeys = params.custom_lkeys
    cluster_types = params.layerchars['cluster_type']
    dispersion = params.layerchars['dispersion']
    pop_proportion = params.layerchars['proportion']
    age_lb = params.layerchars['age_lb'] # todo: potentially confusing with the age_up in the contact matrix
    age_ub = params.layerchars['age_ub']

    layer_members = {}

    uids = get_uids(pop_size)

    # household contacts
    n_households = get_numhouseholds(household_dist, pop_size)
    household_heads = get_household_heads(age_dist, n_households)
    h_contacts, ages = make_hcontacts(n_households,
                                      pop_size,
                                      household_heads,
                                      uids,
                                      contact_matrix)
    contacts['H'] = h_contacts
    layer_members['H'] = uids # All people exist in the household layer

    # school contacts
    key = 'S'
    social_no = n_contacts[key]
    s_contacts = make_scontacts(uids, ages, social_no)
    contacts[key] = s_contacts
    layer_members['S'] = np.array(list(s_contacts.keys()))

    # workplace contacts
    key = 'low_risk_work'
    work_no = n_contacts[key]
    proportion_high_risk = params.extrapars["prop_high_risk_work"]
    low_risk_contacts, high_risk_contacts = make_wcontacts(uids, ages, work_no, proportion_high_risk)
    contacts['low_risk_work'] = low_risk_contacts
    contacts['high_risk_work'] = high_risk_contacts
    layer_members['low_risk_work'] = np.array(list(low_risk_contacts.keys()))
    layer_members['high_risk_work'] = np.array(list(high_risk_contacts.keys()))

    # random community contacts
    key = 'C'
    com_no = n_contacts[key]
    include = uids
    c_contacts = make_random_contacts(include_inds=include, mean_number_of_contacts=com_no, dispersion=dispersion['C'])
    contacts[key] = c_contacts
    layer_members['C'] = uids

    # Custom layers: those that are not households, work, school or community
    custom_contacts, custom_layer_members = make_custom_contacts(uids,
                                           n_contacts,
                                           pop_size,
                                           ages,
                                           custom_lkeys,
                                           cluster_types,
                                           dispersion,
                                           pop_proportion,
                                           age_lb,
                                           age_ub)
    contacts.update(custom_contacts)

    layer_members = sc.mergedicts(layer_members, custom_layer_members)

    # Initialize the new contacts
    cv_contacts = cv.Contacts(layer_keys=all_lkeys)
    for lkey in contacts.keys():
        p1 = []
        p2 = []
        for a1, b1 in contacts[lkey].items():
            p1.extend([a1]*len(b1))
            p2.extend(b1)

        cv_contacts[lkey]['p1'] = np.array(p1,dtype=cvd.default_int)
        cv_contacts[lkey]['p2'] = np.array(p2,dtype=cvd.default_int)
        cv_contacts[lkey]['beta'] = np.ones(len(p1),dtype=cvd.default_float)
        cv_contacts[lkey].validate()

    return cv_contacts, ages, uids, layer_members


def make_contacts(params):
    contacts = {}

    pop_size = params.pars['pop_size']
    household_dist = params.household_dist
    age_dist = params.age_dist
    contact_matrix = params.contact_matrix
    n_contacts = params.pars['contacts']
    all_lkeys = params.all_lkeys

    # for custom layers
    custom_lkeys = params.custom_lkeys
    cluster_types = params.layerchars['cluster_type']
    dispersion = params.layerchars['dispersion']
    pop_proportion = params.layerchars['proportion']
    age_lb = params.layerchars['age_lb'] # todo: potentially confusing with the age_up in the contact matrix
    age_ub = params.layerchars['age_ub']

    layer_members = {}

    uids = get_uids(pop_size)

    # household contacts
    n_households = get_numhouseholds(household_dist, pop_size)
    household_heads = get_household_heads(age_dist, n_households)
    h_contacts, ages = make_hcontacts(n_households,
                                      pop_size,
                                      household_heads,
                                      uids,
                                      contact_matrix)
    contacts['H'] = h_contacts
    layer_members['H'] = uids # All people exist in the household layer

    # school contacts
    key = 'S'
    social_no = n_contacts[key]
    s_contacts = make_scontacts(uids, ages, social_no)
    contacts[key] = s_contacts
    layer_members['S'] = np.array(list(s_contacts.keys()))

    # workplace contacts
    key = 'W'
    work_no = n_contacts[key]
    w_contacts = make_wcontacts(uids, ages, work_no)
    contacts[key] = w_contacts
    layer_members['W'] = np.array(list(w_contacts.keys()))


    # random community contacts
    key = 'C'
    com_no = n_contacts[key]
    include = uids
    c_contacts = make_random_contacts(include_inds=include, mean_number_of_contacts=com_no, dispersion=dispersion['C'])
    contacts[key] = c_contacts
    layer_members['C'] = uids

    # Custom layers: those that are not households, work, school or community
    custom_contacts, custom_layer_members = make_custom_contacts(uids,
                                           n_contacts,
                                           pop_size,
                                           ages,
                                           custom_lkeys,
                                           cluster_types,
                                           dispersion,
                                           pop_proportion,
                                           age_lb,
                                           age_ub)
    contacts.update(custom_contacts)

    layer_members = sc.mergedicts(layer_members, custom_layer_members)

    # Initialize the new contacts
    cv_contacts = cv.Contacts(layer_keys=all_lkeys)
    for lkey in contacts.keys():
        p1 = []
        p2 = []
        for a1, b1 in contacts[lkey].items():
            p1.extend([a1]*len(b1))
            p2.extend(b1)

        cv_contacts[lkey]['p1'] = np.array(p1,dtype=cvd.default_int)
        cv_contacts[lkey]['p2'] = np.array(p2,dtype=cvd.default_int)
        cv_contacts[lkey]['beta'] = np.ones(len(p1),dtype=cvd.default_float)
        cv_contacts[lkey].validate()

    return cv_contacts, ages, uids, layer_members

def make_people(params) -> cv.People:
    """
    Construct a cv.People object

    Uses a Parameters() object to construct a Covasim People object

    Args:
        params:

    Returns: - People object
             - A dictionary of layer members, {lkey: [indexes]}
    """
    cv_contacts, ages, uids, layer_members = make_contacts(params)
    people = cv.People(pars=params.pars, contacts=cv_contacts, age=ages, uid=uids)
    return people, layer_members
