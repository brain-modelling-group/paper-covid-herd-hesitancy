import covasim.utils as cvu
import numpy as np
import numba as nb

def create_clustering(people_to_cluster, mean_cluster_size):
    """
    Return random clustering of people

    Args:
        people_to_cluster: Indexes of people to cluster e.g. [1,5,10,12,13]
        mean_cluster_size: Mean cluster size (poisson distribution)

    Returns: List of lists of clusters e.g. [[1,5],[10,12,13]]
    """

    # people_to_cluster = np.random.permutation(people_to_cluster) # Optionally shuffle people to cluster - in theory not necessary?
    clusters = []
    n_people = len(people_to_cluster)
    n_remaining = n_people

    while n_remaining > 0:
        this_cluster = cvu.poisson(mean_cluster_size)  # Sample the cluster size
        if this_cluster > n_remaining:
            this_cluster = n_remaining
        clusters.append(people_to_cluster[(n_people-n_remaining)+np.arange(this_cluster)].tolist())
        n_remaining -= this_cluster

    return clusters

## Fast choice implementation
# From https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087
# by Jeremy Howard https://twitter.com/jeremyphoward/status/955136770806444032
@nb.njit
def sample(n, q, J, r1, r2):
    res = np.zeros(n, dtype=np.int32)
    lj = len(J)
    for i in range(n):
        kk = int(np.floor(r1[i]*lj))
        if r2[i] < q[kk]: res[i] = kk
        else: res[i] = J[kk]
    return res

class AliasSample():
    def __init__(self, probs):
        self.K=K= len(probs)
        self.q=q= np.zeros(K)
        self.J=J= np.zeros(K, dtype=np.int)

        smaller,larger  = [],[]
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0: smaller.append(kk)
            else: larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small,large = smaller.pop(),larger.pop()
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])
            if q[large] < 1.0: smaller.append(large)
            else: larger.append(large)

    def draw_one(self):
        K,q,J = self.K,self.q,self.J
        kk = int(np.floor(np.random.rand()*len(J)))
        if np.random.rand() < q[kk]: return kk
        else: return J[kk]

    def draw_n(self, n):
        r1,r2 = np.random.rand(n),np.random.rand(n)
        return sample(n,self.q,self.J,r1,r2)


def sample_household_cluster(sampler, bin_lower, bin_upper, reference_age, n):
    """
    Return list of ages in a household/location based on mixing matrix and reference person age
    """

    ages = [reference_age]  # The reference person is in the household/location

    if n > 1:
        idx = np.digitize(reference_age, bin_lower) - 1  # First, find the index of the bin that the reference person belongs to
        sampled_bins = sampler[idx].draw_n(n-1)

        for bin in sampled_bins:
            ages.append(int(round(np.random.uniform(bin_lower[bin]-0.5, bin_upper[bin]+0.5))))

    return np.array(ages)


def make_household_clusters(n_households, pop_size, household_heads, uids, contact_matrix):
    """

    :param n_households:
    :param pop_size:
    :param household_heads:
    :return:
        h_clusters: a list of lists in which each sublist contains
                    the IDs of the people who live in a specific household
        ages: flattened array of ages, corresponding to the UID positions
    """
    mixing_matrix = contact_matrix['matrix']
    mixing_matrix = mixing_matrix.div(mixing_matrix.sum(axis=1), axis=0)
    samplers = [AliasSample(mixing_matrix.iloc[i,:].values) for i in range(mixing_matrix.shape[0])] # Precompute samplers for each reference age bin

    age_lb = contact_matrix['age_lb']
    age_ub = contact_matrix['age_ub']

    h_clusters = []
    ages = np.zeros(pop_size, dtype=int)
    h_added = 0
    p_added = 0

    for h_size, h_num in n_households.iteritems():
        for household in range(h_num):
            head = household_heads[h_added]
            # get ages of people in household
            household_ages = sample_household_cluster(samplers,
                                                      age_lb,
                                                      age_ub,
                                                      head,
                                                      h_size)
            # add ages to ages array
            ub = p_added + h_size
            ages[p_added:ub] = household_ages
            # get associated UID that defines a household cluster
            h_ids = uids[p_added:ub]
            h_clusters.append(h_ids)
            # increment sliding windows
            h_added += 1
            p_added += h_size
    return h_clusters, ages


def make_sclusters(uids, ages, s_contacts):
    # filter UIDs array so that it only contains children between ages 5-18
    classrooms = []
    for age in range(5, 18):
        age_idx = ages == age
        children_thisage = uids[age_idx]
        classrooms.extend(create_clustering(children_thisage, s_contacts))
    for i in range(len(classrooms)):
        adult_idx = ages > 18
        adult_uid = [np.random.choice(uids[adult_idx])]
        classrooms[i].extend(adult_uid)
    return classrooms


def make_wclusters(uids, ages, w_contacts):
    work_idx = np.logical_and(ages > 18, ages <= 65)
    work_uids = uids[work_idx]
    workplace_clusters = create_clustering(work_uids, w_contacts)
    return workplace_clusters


def make_custom_clusters(uids, pop_size, ages, custom_lkeys, pop_proportion, age_lb, age_ub):
    for custom_key in custom_lkeys:
        agel = age_lb[custom_key]
        ageu = age_ub[custom_key]
        n_people = int(pop_proportion[custom_key] * pop_size)
        in_layer = np.logical_and(ages > agel, ages < ageu)
        layer_id = uids[in_layer]
        # sample with replacement so that we have enough people for the layer
        layer_pop = np.random.choice(layer_id, size=n_people)
    return layer_pop, in_layer
