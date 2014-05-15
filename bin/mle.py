import logging

import numpy as np
from scipy.optimize import minimize

logging.basicConfig(level=logging.DEBUG)

APPROX_ZERO = 0.0001

def log_likelihood(arr, fcn, thetas):
    """
    arr is array of [[x0, y0], [x1, y1], ...]
        where each yi in {0, 1}
    fcn if function, and will be applied to each xi
    thetas is tuple, a set of parameters passed to fcn along with each xi

    calculates the sum of the log-likelihood of arr
        = sum_i fcn(xi, *thetas)^(yi) * (1 - fcn(xi, *thetas))^(1-yi)
    """
    fcn_x = lambda x: fcn(x, *thetas)
    likelihood = lambda row: fcn_x(row[0]) if row[1] else 1-fcn_x(row[0])
    log_likeli = lambda row: np.log(likelihood(row))
    val = sum(map(log_likeli, arr))
    return val

def pick_best_theta(thetas):
    close_enough = lambda x,y: abs(x-y) < APPROX_ZERO
    min_th = min(thetas, key=lambda d: d['fun'])
    if len(thetas) > 1:
        ths = [th for th in thetas if close_enough(th['fun'], min_th['fun'])]
        msg = '{0} out of {1} guesses found minima of {2}'.format(len(ths), len(thetas), min_th['fun'])
        # logging.info(msg)
    return min_th['x']

def keep_solution(theta, bnds, ymin):
    """
    theta is return value of scipy.optimize.minimize,
        where theta['x'] is list [t1, ...] of solution values
    bnds is list [(lb1, rb1), ...] of bounds for each ti in theta['x']
    ymin is previously-found minimum solution

    returns True iff theta is a success, lower than ymin, and has solutions not near its bounds
    """
    if not theta['success']:
        return False
    if theta['fun'] >= ymin:
        return False
    close_enough = lambda x, b: abs(x-b) < APPROX_ZERO*10
    at_left_bound = lambda x, lb: close_enough(x, lb) if lb else False
    at_right_bound = lambda x, rb: close_enough(x, rb) if rb else False
    at_bounds = lambda x, (lb, rb): at_left_bound(x, lb) or at_right_bound(x, rb)
    return not any([at_bounds(th, bnd) for th, bnd in zip(theta['x'], bnds)])

def mle(data, log_likelihood_fcn, guesses, bounds=None, constraints=None, quick=False, method='TNC'):
    """
    data is list [(dur, resp)]
        dur is float
        resp is bool
    quick is bool
        chooses the first solution not touching the bounds
    method is str
        bounds only for: L-BFGS-B, TNC, SLSQP
        constraints only for: COBYLA, SLSQP
        NOTE: SLSQP tends to give a lot of run-time errors...
    """
    if len(data) == 0 or len(guesses) == 0:
        return None
    if bounds is None:
        bounds = []
    if constraints is None:
        constraints = []

    thetas = []
    ymin = float('inf')
    for guess in guesses:
        theta = minimize(log_likelihood_fcn, guess, method=method, bounds=bounds, constraints=constraints)
        if keep_solution(theta, bounds, ymin):
            ymin = theta['fun']
            thetas.append(theta)
            if quick:
                return thetas
            msg = '{0}, {1}'.format(theta['x'], theta['fun'])
            logging.info(msg)
    return thetas
