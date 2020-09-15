import re
from collections import Counter
import numpy as np
from numpy import linalg as la
from . import basis_function as bf, utils


class BsplineCurve(object):
    def __init__(self, p, knots, coefs):
        self._degree = p
        self._knots = knots
        self._coefs = coefs

    def __call__(self, x):
        A = bf.get_collocation_matrix(self._degree, self._knots, x)
        return A @ self._coefs


def validate_knots_and_data(p, knots, x):
    """ Checks the Schoenberg-Whitney conditions

    :param p:       The degree of the B-spline curve
    :param knots:   The knot sequence to use
    :param x:       The sites of data points to fit
    :raises ValueError:     if any condition is violated
    """
    utils.validate_knots(knots)
    numknots = len(knots)
    minknots = p+3  # ensures n > 0, where n+1 is the number of B-splines and n = numknots - p - 2
    if numknots < minknots:
        raise ValueError('There must be at least {0} knots for a degree {1} spline.'.format(minknots, p))
    X = np.sort(x)
    if X[0] < knots[0] or X[-1] > knots[-1]:
        raise ValueError('At least one site falls outside the knot sequence.')
    if max(Counter(knots).values()) > p+1:
        raise ValueError('At least one knot has multiplicity greater than {0}.'.format(p+1))
    lastspan = utils.get_last_knotspan(knots)
    if not all([np.any([utils.is_in_knotspan(a, (knots[i], knots[i+p+1]), i == lastspan) for a in X])
                for i in range(numknots-p-1) if knots[i] < knots[i+p+1]]): # only bother for non-zero length spans
        raise ValueError('At least one B-spline has no constraining data.')


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
    return np.concatenate((np.repeat(x[0], p+1), iknots, np.repeat(x[-1], p+1)))


def get_default_knots(p, x):
    """ Gets a knot sequence that fulfills the Schoenberg-Whitney conditions for the given sites

    :param p:   The degree of the B-spline curve
    :param x:   The data sites to be fit
    :return:    A basic knot vector that meets the Schoenberg-Whitney conditions
    """
    return augment_knots(p, get_default_interior_knots(p, x), x)


def get_spline(p, knots, x, y, **kwargs):
    """ An interpolating spline estimating the curve described by (x,y) pairs

    :param p:       The degree of the B-spline curve (int)
    :param knots:   Knots used to calculate the interpolating B-spline curve (iterable of floats)
    :param x:       Vector containing independent variable values for data points.  Must be same size as y
    :param y:       Vector containing dependent variable values for data points. Must be same size as x
    :key minimize_d[1|2|...|p]_x:  A list of x values at which the specified derivative should be
                    minimized (equal to zero).
                    For example, for keyword minimize_d1_x, the value would be an iterable of x values where the
                    first derivative can reasonably be expected to be close to 0.
                    Will ignore any derivative greater than p, since those would all be 0 anyway
    :return:        A namedtuple containing the degree (p), knots, and coefficients for the interpolating B-spline
    """
    if len(x) != len(y):
        raise ValueError('Parameters x and y must be the same length.')
    A = bf.get_collocation_matrix(p, knots, x)
    X = x
    d = y
    r = re.compile(r'minimize_d(\d)_x')
    for md in filter(r.match, kwargs.keys()):
        der = int(r.match(md)[1])
        if der <= p:  # ignore derivatives higher than the degree of the spline
            xx = kwargs[md]
            X = np.concatenate((X, xx))
            A = np.vstack((A, bf.get_collocation_matrix(p, knots, xx, der)))
            d = np.append(d, np.zeros(len(xx)))
    validate_knots_and_data(p, knots, X)
    coef, *_ = la.lstsq(A, d, rcond=None)
    return BsplineCurve(p, knots, coef)























