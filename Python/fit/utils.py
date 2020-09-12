import numpy as np


def is_nondecreasing(vals):
    """ True if the iterable vals is non-decreasing; False otherwise
    """
    return all(a <= b for a, b in zip(vals, vals[1:]))


def validate_knots(knots):
    """ Checks conditions required for a valid knot sequence and throws a ValueError if any are violated
    Conditions include
        - knots is a non-decreasing sequence
        - there is at least one index i for which knots[i+1] > knots[i]

    :raises ValueError:     if any condition is violated
    """
    if not is_nondecreasing(knots):
        raise ValueError('The knot sequence is not non-decreasing.')
    if knots[0] == knots[-1]:   # already know sequence is non-decreasing
        raise ValueError('There must be at least one knot span with length > 0.')
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


def get_last_knotspan(knots):
    """ Returns the starting index of the last nonempty knot span in the knot sequence
    """
    validate_knots(knots)
    return np.max(np.nonzero(np.diff(knots)))


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
        return get_last_knotspan(knots)
    else:
        return next((idx for idx in range(knots.size - 1) if knots[idx] <= u < knots[idx + 1]))


def find_sites_in_span(knots, iks, sites):
    """ Returns indices for sites that fall in the specified knot span [knots[iks], knots[iks+1])
    For sites that equal knots[-1], the last non-empty knot span will be returned

    :param knots:   The knot sequence (iterable of floats)
    :param iks:     The index of the knot span for which sites should be found (int where 0 <= iks < len(knots)-1)
                    Does not support negative knot indexing
    :param sites:   The nondecreasing list of sites to search.  (iterable of floats)
    :return:        A list of indexes for sites that fall in knot span
    """
    if iks < 0 or iks >= len(knots) - 1:
        return np.empty((0,), dtype=int)
    else:
        lastspan = get_last_knotspan(knots)
        # nonzero returns a tuple of arrays (each treating a different axis) so need to just get the 0th entry for 1D
        return np.nonzero(((knots[iks] <= sites) & (sites < knots[iks+1])) | ((iks == lastspan) & (sites == knots[-1])))[0]


def is_function_nonzero(p, knots, icp, iks):
    """ True if N_icp,degree is nonzero on knot span [u_iks, u_iks+1) of non-zero length; False otherwise.
    This function does not support negative indexing.
    """
    return (0 <= iks < len(knots)) and (iks in range(icp, icp+p+1)) and (knots[iks + 1] - knots[iks] > 0)









