import logging
import itertools

import numpy as np
from scipy.optimize import minimize

logging.basicConfig(level=logging.DEBUG)

APPROX_ZERO = 0.0001

add_dicts = lambda a, b: dict(a.items() + b.items())
make_dict = lambda key_order, thetas: dict(zip(key_order, thetas))

def theta_to_dict(thetas, theta_key_order, theta):
    thetas_lookup = make_dict(theta_key_order, thetas)
    thetas_preset = dict((key, val) for key, val in thetas_lookup.iteritems() if val is not None)
    keys_left = [key for key in theta_key_order if thetas_lookup[key] is None]
    return add_dicts(thetas_preset, make_dict(keys_left, theta))

def generic_fit(fcn, theta_key_order, thetas_to_fit, theta_defaults, quick, ts, bins, coh, guesses=None):
    """
    theta_key_order is e.g. ['A', 'B', 'T']
    theta_fit_defaults is e.g. (True, False, True)
    theta_defaults is e.g. {'A': 1.0, 'B': 0.5, 'T': 100.0}
    """
    fit_found = True
    thetas = [None if d else theta_defaults[t] for t, d in zip(theta_key_order, thetas_to_fit)]
    ths = fcn(ts, thetas, quick=quick, guesses=guesses)
    if ths:
        th = pick_best_theta(ths)
    else:
        msg = 'No fits found. Using {0}'.format(theta_defaults)
        # logging.warning(msg)
        fit_found = False
        th = [theta_defaults[t] for t, d in zip(theta_key_order, thetas_to_fit) if d]
    msg = '{0}% {2}: {1}'.format(int(coh*100), th, '[current fit]')
    # logging.info(msg)
    return theta_to_dict(thetas, theta_key_order, th), th, fit_found

def make_guesses(thetas, theta_key_order, guesses_lookup):
    thetas_lookup = make_dict(theta_key_order, thetas)
    guesses = []
    for key in theta_key_order:
        if thetas_lookup[key] is None:
            guesses.append(guesses_lookup[key])
    return list(itertools.product(*guesses)) # cartesian product

def make_bounds(thetas, theta_key_order, bounds_lookup):
    return [bounds_lookup[key] for key, val in zip(theta_key_order, thetas) if val is None]

def log_likelihood(arr, fcn, thetas):
    """
    arr is array of [[x0, y0], [x1, y1], ...]
        where each yi in {0, 1}
    fcn if function, and will be applied to each xi
    thetas is tuple, a set of parameters passed to fcn along with each xi

    calculates the sum of the log-likelihood of arr
        = sum_i fcn(xi, *thetas)^(yi) * (1 - fcn(xi, *thetas))^(1-yi)
    """
    if type(thetas) is dict:
        fcn_x = lambda x: fcn(x, **thetas)
    elif type(thetas) is list or type(thetas) is tuple:
        fcn_x = lambda x: fcn(x, *thetas)
    likelihood = lambda row: fcn_x(row[0]) if row[1] else 1-fcn_x(row[0])
    log_likeli = lambda row: np.log(likelihood(row))
    val = sum(map(log_likeli, arr))
    return val

def log_likelihood_factory(data, fcn, thetas, theta_key_order):
    thetas_lookup = make_dict(theta_key_order, thetas)
    thetas_preset = dict((key, val) for key, val in thetas_lookup.iteritems() if val is not None)
    keys_left = [key for key in theta_key_order if thetas_lookup[key] is None]
    return lambda theta: -log_likelihood(data, fcn, add_dicts(thetas_preset, make_dict(keys_left, theta)))

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

def fit_mle(data, inner_likelihood_fcn, thetas, theta_key_order, guesses_lookup, bounds_lookup, constraints, quick=False, guesses=None, method='TNC'):
    if guesses is None:
        guesses = make_guesses(thetas, theta_key_order, guesses_lookup)
    bounds = make_bounds(thetas, theta_key_order, bounds_lookup)
    return mle(data, log_likelihood_factory(data, inner_likelihood_fcn, thetas, theta_key_order), guesses, bounds, constraints, quick, method)
