import random

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
