from numpy.polynomial.polynomial import Polynomial as poly
import numpy as np
from .utils import get_num_ctrlpts


def get_basis_function(icp, degree, knots, iks):
    """ Returns a Polynomial for calculating basis function N_icp,degree in knot span [u_iks, u_iks+1)

    :param icp:     The index of the control point for which this function will be the coefficient (int i),
                    where 0 <= i <= n, where n = m-p-1
    :param degree:  The degree of the B-spline (int p)
    :param knots:   The knot sequence used by the B-spline ([float] of length m)
    :param iks:     A knot span index indicating range of values [u_iks, u_iks+1)
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
    n = get_num_ctrlpts(degree, knots) - 1
    m = n+degree+1
    if (icp < 0) or (icp > n):
        raise ValueError('icp is out of range [0,{0}]'.format(n))
    if (iks < 0) or (iks > m):
        raise ValueError('iks is out of range [0,{0}]'.format(m))
    # N_i,p is nonzero on [u_i, u_i+p+1)
    return get_basis_function_internal(icp, degree) if icp <= iks < icp+degree+1 else poly(0)
