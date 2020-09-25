import re
from collections import Counter
import numpy as np
from numpy import linalg as la
from . import basis_function as bf, utils

set_kw = 'set'
minimize_kw = 'minimize'
re_deriv_constraints = re.compile(r'({0}|{1})_d(\d)_x'.format(set_kw, minimize_kw))

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


def _get_weighted_matrix(wt, mat):
    wt = np.asarray(wt); mat = np.asarray(mat)
    rc = mat.shape
    r = rc[0]
    ws = wt.size
    if ws == 1:
        wt = np.repeat(wt, r)
    elif ws != r:
        raise ValueError('Wrong number of weights provided')
    if len(rc) > 1:  # reshape to a column matrix if mat is not a vector
        wt.shape = (r, 1)
    return wt * mat


def _get_derivative_constraints(p, knots, **kwargs):
    sites = np.empty(0, dtype=float)
    d = np.empty(0, dtype=float)
    A = np.empty((0, len(knots) - 1 - p), dtype=float)
    for dc in filter(re_deriv_constraints.match, kwargs.keys()):
        constr_desc = re_deriv_constraints.match(dc)
        ctype = constr_desc[1]
        der = int(constr_desc[2])
        if 0 < der <= p:  # derivatives greater than the degree of the spline are already all 0
            if ctype == minimize_kw:
                xx = kwargs[dc]
                dd = np.zeros(len(xx))
            else:
                constr = np.array(kwargs[dc])
                xx = constr[:, 0]
                dd = constr[:, 1]
            dwp = '{ctype}_d{der}_w'.format(**locals())
            dw = 1 if dwp not in kwargs else kwargs[dwp]
            sites = np.append(sites, xx)
            A = np.vstack((A, _get_weighted_matrix(dw, bf.get_collocation_matrix(p, knots, xx, der))))
            d = np.append(d, _get_weighted_matrix(dw, dd))
    return sites, A, d


def get_bspline_fit(p, knots, x, y, w=1, **kwargs):
    """ An interpolating spline estimating the curve described by (x,y) pairs

    :param p:       The degree of the B-spline curve (int)
    :param knots:   Knots used to calculate the interpolating B-spline curve (iterable of floats)
    :param x:       Vector containing independent variable values for data points.  Must be same size as y
    :param y:       Vector containing dependent variable values for data points. Must be same size as x
    :param w:       Vector or scalar containing weights for (x,y) values.  A scalar value means all data points have
                    the same value.  Defaults to all (x,y) data points having weight 1.
    :key minimize_d[1|2|...|p]_x:  A list of x values at which the specified derivative should be
                    minimized (equal to zero). For example, for keyword minimize_d1_x, the value would be an iterable
                    of x values where the first derivative can reasonably be expected to be close to 0.
                    Will ignore any derivative greater than p, since those would all be 0 anyway
    :key minimize_d[1|2|...|p]_w:  An optional scalar or list of weights corresponding to the values of
                    minimize_d<deriv>_x with the same derivative.  If not provided, all weights are set to 1.
                    Ignored if minimize_d<deriv>_x for the same derivative is not provided.
    :key set_d[1|2|...|p]_x: A list of tuples (x, dx) tuples where the value of the specified derivative at x is equal
                    to dx. For example, for keyword set_d2_x, a tuple (x, dx) means the second derivative at x should
                    be equal to dx.  Ignores any derivative greater than p, since those will be equal to 0.
    :key set_d[1|2|...|p]_w: An optional scalar or list of weights corresponding to the tuples of set_d<deriv>_xdx.
                    If not provided, all weights are set to 1.  Ignored if set_d<deriv>_xdx for the same derivative is
                    not provided.
    :return:        A namedtuple containing the degree (p), knots, and coefficients for the interpolating B-spline
    """
    if len(x) == 0 or len(x) != len(y):
        raise ValueError('Parameters x and y must be the same length, and there must be at least one data point.')
    A = _get_weighted_matrix(w, bf.get_collocation_matrix(p, knots, x))
    X = x
    d = w*y  # already confirmed len(x)=len(y) and _get_weighted_matrix ensures len(x)=len(w) or w is scalar
    (dx, dA, dy) = _get_derivative_constraints(p, knots, **kwargs)
    # for md in filter(r.match, kwargs.keys()):
    #     der = int(r.match(md)[1])
    #     if der <= p:  # ignore derivatives higher than the degree of the spline
    #         xx = kwargs[md]
    #         X = np.concatenate((X, xx))
    #         # get weights for the minimization
    #         dwp = 'minimize_d{0}_w'.format(der)
    #         dw = 1 if dwp not in kwargs else kwargs[dwp]
    #         A = np.vstack((A, _get_weighted_matrix(dw, bf.get_collocation_matrix(p, knots, xx, der))))
    #         d = np.append(d, np.zeros(len(xx)))  # no need to multiply weights since all values are 0
    X = np.append(x, dx)
    validate_knots_and_data(p, knots, X)
    A = np.vstack((A, dA))
    d = np.append(w*y, dy)  # correct size of w already guaranteed by checks in _get_weighted_matrix and len(x)==len(y)
    coef, *_ = la.lstsq(A, d, rcond=None)
    return BsplineCurve(p, knots, coef)























