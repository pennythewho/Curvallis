import numpy as np
from geomdl import helpers as hlp


def is_clamped_end_knot(degree, knots, site):
    """ True if the site has multiplicity degree+1 and is either the first or last knot.  Otherwise False
    """
    return (site == knots[0] or site == knots[-1]) and hlp.find_multiplicity(site, knots) == degree + 1


def get_num_ctrlpts(degree, knots):
    """" Returns n + 1, where n = m - p - 1,  m = len(knots) - 1, and p = degree
    """
    return len(knots) - 1 - degree


def get_knotspan_start_idx(knots, site):
    """ Finds the index of the knot that begins the half-open interval in which the site can be found

    :param knots: A non-decreasing finite vector of floats ({u_0, u_1, ..., u_m})
    :param site:  A float (u)
    :return:      The index i where knots[i] <= site < knots[i+1].
                  Returns -1 if site < knots[0] or site > knots[-1] or knots.size < 2 or all knot spans are zero length
                  If site == knots[-1], returns the index of the knot that starts the last interval of non-zero length
    """
    knots = np.asarray(knots)
    if site < knots[0] or site > knots[-1] or knots.size < 2:
        return -1
    elif site == knots[-1]:
        knot_span_len = np.nonzero(np.diff(knots))[0]
        return knot_span_len[-1] if knot_span_len.size > 0 else -1
    else:
        return next((idx for idx in range(knots.size - 1) if knots[idx] <= site < knots[idx+1]))


# def is_in_knotspan(knots, i, site):
#     """ True if knots[i] <= site < knots[i+1], or if site == knots[-1] and i represents the last knot span of nonzero length
#     Otherwise, False
#     """
#     return i == get_knotspan_start_idx(knots, site)



