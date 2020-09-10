import numpy as np
from . import basis_function as bf, utils

def validate_knots_and_data(p, knots, x):
    """ Checks the Schoenberg-Whitney conditions

    :param p:       The degree of the B-spline curve
    :param knots:   The knot sequence to use
    :param x:       The sites of data points to fit
    :raises ValueError:     if any condition is violated
    """
    utils.validate_knots(knots)
    numsites = len(x)
    if len(knots) < max(p+1, 1):
        raise ValueError('There must be at least {0} knots for a {1}-degree spline.'.format(p+1, p))
    if any(np.diff(x) < 0):
        raise ValueError('Sites must be non-decreasing.')
    if x[0] < knots[0] or x[-1] > knots[-1]:
        raise ValueError('At least one site falls outside the knot sequence.')
    if any([si == sipp for (si, sipp) in zip(x, x[p:])]):
        raise ValueError('Sites can be repeated at most {p} times.'.format(p=p))
    if utils.get_multiplicity(knots, knots[0]) != p+1 or utils.get_multiplicity(knots, knots[-1]) != p+1:
        raise ValueError('At least one of the first and last knots does not have multiplicity {0}'.format(p+1))
    if not (all(np.less(knots[1:numsites-1], x[1:-1])) and all(np.less(x[1:-1], knots[p+2:numsites+p]))):
        raise ValueError('The interior knots and data points do not meet the Schoenberg-Whitney conditions.')

def get_default_interior_knots(p, x):
    """  Gets an interior knot sequence that fulfills the Schoenberg-Whitney conditions (except end knot multiplicity)
    for the given sites

    :param p:   The degree of the B-spline curve
    :param x:   The data sites to be fit
    :return:    A vector of interior knots that meets the Schoenberg-Whitney conditions
    """
    return np.convolve(x[1:-1], [1.0/p]*p, 'valid')


def augment_knots(p, iknots, x):
    """ Adds beginning and ending knots to an interior knot sequence and checks Schoenberg-Whitney conditions

    :param p:       The degree of the B-spline curve
    :param iknots:  An internal knot sequence
    :param x:       The data sites to be fit
    :return:        The final knot sequence
    """
    knots = np.concatenate((np.repeat(x[0], p+1), iknots, np.repeat(x[-1], p+1)))
    validate_knots_and_data(p, knots, x)
    return knots


def get_default_knots(p, x):
    """ Gets a knot sequence that fulfills the Schoenberg-Whitney conditions for the given sites

    :param p:   The degree of the B-spline curve
    :param x:   The data sites to be fit
    :return:    A basic knot vector that meets the Schoenberg-Whitney conditions
    """
    return augment_knots(p, get_default_interior_knots(p, x), x)














