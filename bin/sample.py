import random

import numpy as np

def sample_wr(population, k):
    "Chooses k random elements (with replacement) from a population"
    n = len(population)
    _random, _int = random.random, int  # speed hack 
    result = [None] * k
    for i in xrange(k):
        j = _int(_random() * n)
        result[i] = population[j]
    return result

def bootstrap(population, n, m=None):
    if not m:
    	m = len(population)
	return [sample_wr(population, m) for i in xrange(n)]

def bootstrap_se(bootstrap_stats):
    """
    calculates the standard error of a collection of bootstrap statistics
    """
    # ddof=1 => divide by N-1
    return np.std(bootstrap_stats, ddof=1) if len(bootstrap_stats) > 1 else 0.0
