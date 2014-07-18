import numpy as np
from scipy.optimize import minimize

def params(theta, unfold=False):
    """
    a is scale
    b is shape
    minV is lower asymptote [default = 0.0 or 0.5]
    maxV is upper asymptote [default = 1.0]
    """
    if len(theta) == 2:
        a, b = theta
        minV = 0.0 if unfold else 0.5
        maxV = 1.0
    elif len(theta) == 3:
        a, b, maxV = theta
        minV = 0.0 if unfold else 0.5
    elif len(theta) == 4:
        a, b, minV, maxV = theta
    else:
        raise Exception("params must be length 2, 3, or 4: {0}".format(theta))
    return a, b, minV, maxV

def weibull(x, theta, unfold=False):
    """
    """
    a, b, minV, maxV = params(theta, unfold)
    return maxV - (maxV-minV) * np.exp(-pow(x/a, b))

def inv_weibull(theta, y, unfold=False):
    """
    the function calculates the inverse of a weibull function
    with given parameters (theta) for a given y value
    returns the x value
    """
    a, b, minV, maxV = params(theta, unfold)
    return a * pow(np.log((maxV-minV)/(maxV-y)), 1.0/b)

def log_likelihood(data, fcn, thetas, fcn_kwargs):
    """
    data is array of [(x0, y0), (x1, y1), ...], where each yi in {0, 1}
    fcn if function, and will be applied to each xi
    thetas is tuple, a set of parameters passed to fcn along with each xi

    calculates the sum of the log-likelihood of data
        = sum_i fcn(xi, *thetas)^(yi) * (1 - fcn(xi, *thetas))^(1-yi)
    """
    likelihood = lambda row: fcn(row[0], thetas, *fcn_kwargs) if row[1] else 1-fcn(row[0], thetas, *fcn_kwargs)
    log_likeli = lambda row: np.log(likelihood(row))
    val = sum(map(log_likeli, data))
    # if np.isnan(val):
    #     yp = lambda v: v*0.99 + 0.005
    #     likelihood = lambda row: fcn(row[0], thetas, *fcn_kwargs)**yp(row[1]) if row[1] else (1-fcn(row[0], thetas, *fcn_kwargs))**(1-yp(row[1]))
    #     val = sum(map(log_likeli, data))
    return val

def solve(xs, ys, unfold=False, guess=(0.5, 0.7, 0.5, 0.85), ntries=20, quick=True):
    guess = np.array(guess)
    APPROX_ZERO, APPROX_ONE = 0.00001, 0.99999
    bounds = [(APPROX_ZERO, None), (APPROX_ZERO, None)] + [(APPROX_ZERO, APPROX_ONE)] * (len(guess) - 2)
    pf = lambda th, d=zip(xs, ys): -log_likelihood(d, weibull, th, {'unfold': unfold})

    sol = None
    ymin = 100000
    method = 'L-BFGS-B' # 'SLSQP' 
    for i in xrange(ntries):
        if i > 0:
            guess = guess*np.random.uniform(0.95, 1.05)
        soln = minimize(pf, guess, method=method, bounds=bounds, constraints=[])
        if soln['success']:
            theta_hat = soln['x']
            if not quick and soln['fun'] < ymin:
                sol = theta_hat
            else:
                return theta_hat
        else:
            print soln
    return sol
