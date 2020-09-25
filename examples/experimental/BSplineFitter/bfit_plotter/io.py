import csv
from collections import namedtuple
import numpy as np

dy_template = 'd{0}y'
dyw_template = 'd{0}w'

def load_data(path, p, with_weights=[]):
    """  Loads and parses data from a specified file and returns parameters suitable for calls to bfit.fit.get_bspline_fit

    :param path:    absolute or relative path to csv data file
                    columns should be x, y, dy/dx, d2y/dx2, d3y/dx3, ...
                    unless with_weights indicate weight will be provided for a particular derivative
                    To ignore the x for a particular value or derivative, provide an empty value or white space,
                    as 0 is a valid value for all columns.
                    E.g., without weights
                        1,,2    =>      indicates that dy/dx=2 at x=1   (no value for y provided)
                        2,1     =>      indicates that y=1 at x=2
                        3,4,5   =>      indicates that at x=3, y=4 and dy/dx=5
    :param p:       target degree of the b-spline fit.  Any derivatives higher than this value will be ignored.
    :param with_weights:  an iterable of non-negative integers indicating the derivatives for which weights are
                    provided in the input file (list need not be sorted). Default value assumes no weights are provided.
                        0 indicates that weights are provided for y values,
                        1 indicates that weights are provided for dy/dx values, and
                        2 indicates that weights are provided for d2y/dx2 values, etc.
                    For example, with_weights=[0,2] means that columns in the csv will be interpreted as follows:
                        x, y, y_weight, dy/dx, d2y/dx2, d2y/dx2_weight, d3y/dx3, d4y/dx4, ...
    :return:        a namedtuples containing args for bfit.get_bspline_fit
                        .x: a 1D float array of x values corresponding to values in y
                        .y: a 1D float array of y values corresponding to values in x
                        .w: (optional) a 1D float array of weights corresponding to y values
                        .kwargs: (optional) a dictionary suitable for passing to bfit.fit.get_bspline_fit using the keyword
                            expansion operator (**).  contains keys and values for passing derivative constraints
                            as documented in bfit.fit.get_bspline_fit
    """
    # determine expected columns and initialize dictionary with expected keys

    # is file present?
    # load file as csv
    #
    # line by line
        # column by column
    # remove empty keys
    pass

def _get_expected_cols(p, with_weights):
    """ returns an ordered list of columns in the data file """
    cols = ['x', 'y'] + (['w'] if 0 in with_weights else [])
    for d in range(1, p+1):
        cols = cols + [dy_template.format(d)] + ([dyw_template.format(d)] if d in with_weights else [])
    return cols
