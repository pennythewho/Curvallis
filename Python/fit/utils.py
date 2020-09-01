import numpy as np


def validate_knots(knots):
    """ Ensures knots is a non-decreasing sequence """
    kd = np.diff(knots)
    if any(kd < 0):
        raise ValueError('The knot sequence is non-decreasing.')
    return


def get_multiplicity(knots, knot):
    """ Returns multiplicity for given knot

    :param knots:   A non-decreasing finite vector of knots ([float])
    :param knot:    The knot for which multiplicity is determined
    """
    validate_knots(knots)
    return sum([1 for k in knots if k == knot])


def is_clamped_end_knot(p, knots, u):
    """ True if the site u has multiplicity p+1 and is either the first or last knot.  Otherwise False

    :param p:       The degree of the B-spline (int >= 1)
    :param knots:   A non-decreasing finite vector of knots ([float])
    :param u:       The site to evaluate (float)
    """
    return (u == knots[0] or u == knots[-1]) and get_multiplicity(knots, u) == p + 1


def get_num_ctrlpts(p, knots):
    """" Returns n + 1, where n = m - p - 1 and m = len(knots) - 1

    :param p:       The degree of the B-spline (int >= 1)
    :param knots:   A non-decreasing finite vector of knots ([float])
    """
    return len(knots) - 1 - p


def get_knotspan_start_idx(knots, u):
    """ Finds the index of the knot that begins the half-open interval in which the site can be found

    :param knots: A non-decreasing finite vector of knots ([float])
    :param u:     The site for which the knot span is being sought  (float)
    :return:      The index i where knots[i] <= site < knots[i+1].
                  Returns -1 if site < knots[0] or site > knots[-1] or knots.size < 2 or all knot spans are zero length
                  If site == knots[-1], returns the index of the knot that starts the last interval of non-zero length
    """
    knots = np.asarray(knots)
    if u < knots[0] or u > knots[-1] or knots.size < 2:
        return -1
    elif u == knots[-1]:
        knot_span_len = np.nonzero(np.diff(knots))[0]
        return knot_span_len[-1] if knot_span_len.size > 0 else -1
    else:
        return next((idx for idx in range(knots.size - 1) if knots[idx] <= u < knots[idx + 1]))


def is_function_nonzero(p, knots, icp, iks):
    """ True if N_icp,degree is nonzero on knot span [u_iks, u_iks+1) of non-zero length; False otherwise.
    This function does not support negative indexing.
    """
    return (0 <= iks < len(knots)) and (iks in range(icp, icp+p+1)) and (knots[iks + 1] - knots[iks] > 0)






