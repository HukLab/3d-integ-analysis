dimport numpy as np

def rand_inds(population, k):
    if not hasattr(population, 'shape'):
        population = np.array(population)
    return np.random.randint(population.shape[0], size=k)

def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    if not hasattr(population, 'shape'):
        population = np.array(population)
    return population[rand_inds(population, k)]

def bootstrap(population, n, k=None):
    if not hasattr(population, 'shape'):
        population = np.array(population)
    if not k:
        k = len(population)
    return population[rand_inds(population, (n, k))]

def bootstrap_se(bootstrap_stats):
    """
    calculates the standard error of a collection of bootstrap statistics
    """
    # ddof=1 => divide by N-1
    return np.std(bootstrap_stats, ddof=1) if len(bootstrap_stats) > 1 else 0.0
