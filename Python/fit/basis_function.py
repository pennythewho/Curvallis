from functools import partial
import numpy as np
from numpy.polynomial.polynomial import Polynomial as poly
from .utils import get_num_ctrlpts, is_function_nonzero, get_knotspan_start_idx


def get_basis_function(icp, p, knots, iks, der=0):
    """ Returns a Polynomial for calculating basis function N_icp,degree in knot span [u_iks, u_iks+1)

    :param icp:     The index of the control point for which this function will be the coefficient (int i),
                    where 0 <= i <= n, where n = m-p-1
    :param p:       The degree of the B-spline (int p)
    :param knots:   The knot sequence used by the B-spline ([float] of length m)
    :param iks:     A knot span index indicating range of values [u_iks, u_iks+1)
    :param der:     The derivative to take (int >= 0)
    :return:        A Polynomial to calculate the basis function for u in the specified knot span
    """
    def get_basis_function_internal(i, p):
        if p == 0:
            return poly(1) if i == iks else poly(0)
        else:
            n_i_pm1 = get_basis_function_internal(i, p-1)
            term1coef = 0 if (knots[i+p]-knots[i]) == 0 else poly((-knots[i], 1))/(knots[i+p]-knots[i])
            n_ip1_pm1 = get_basis_function_internal(i+1, p-1)
            term2coef = 0 if (knots[i+p+1]-knots[i+1]) == 0 else poly((knots[i+p+1], -1))/(knots[i+p+1]-knots[i+1])
            return (term1coef * n_i_pm1) + (term2coef * n_ip1_pm1)
    n = get_num_ctrlpts(p, knots) - 1
    m = n + p + 1
    if (icp < 0) or (icp > n):
        raise ValueError('icp is out of range [0,{0}]'.format(n))
    if (iks < 0) or (iks > m):
        raise ValueError('iks is out of range [0,{0}]'.format(m))
    # knot span [u_i, u_i+1) is only nonzero on N_i-p,p, N_i-p+1,p, ..., N_i,p
    f = get_basis_function_internal(icp, p) if is_function_nonzero(p, knots, icp, iks) else poly(0)
    return f if der == 0 else f.deriv(der)


def get_basis_functions(p, knots):
    """ Returns a matrix of Polynomials that indicate how basis functions should be calculated for various knot spans,
    where out[j,i] = N_i,p for knot span [knots[j], knots[j+1]).

    :param p:       the degree of the B-spline (positive integer)
    :param knots:   the knot sequence to be used to divide the B-spline into pieces (iterable of floats)
    :return:        an (m x n+1) matrix of Polynomials, where m + 1 is the number of knots, and
                    n+1 is the number of control points and n=m-p-1.
    """
    m = len(knots) - 1
    n = m-p-1
    out = np.reshape(np.repeat(poly(0), m*(n+1)), (m, n+1))
    # for knot span [u_i, u_i+1), the only nonzero basis functions are N_{i-p,p}, N_{i-p+1,p}, ... N_{i,p}
    for j in range(m):  # for each knot span (row index)
        fnzi = max(j-p, 0)      # index of first non-zero function for this knot span
        lnzi = min(j+1, n+1)    # index of last non-zero function for this knot span
        out[j, fnzi:lnzi] = [get_basis_function(i, p, knots, j) for i in range(fnzi, lnzi)]
    return out


def get_collocation_matrix(p, knots, sites):
    """ Calculates collocation matrix given degree, knots, and a vector of sites

    :param p:       integer degree of the B-spline (p in the B-spline discussion)
    :param knots:   a float vector with the locations of knots for the B-spline (u_i in the B-spline discussion)
    :param sites:   a float vector with the sites (x-values) for which basis functions should be calculated
                    (u in the B-spline discussion)
    :return:        The len(sites) x get_num_ctrlpts(degree, knots) collocation matrix A,
                    where element A_{i,j} = N_{j,p}(x_i}
    """
    fn_matrix = get_basis_functions(p, knots)
    num_cp = fn_matrix.shape[1]
    out = np.empty((len(sites), num_cp), np.float)
    for r in range(len(sites)):
        iks = get_knotspan_start_idx(knots, sites[r])
        out[r, :] = [fn_matrix[iks, c](sites[r]) for c in range(num_cp)]
    return out



