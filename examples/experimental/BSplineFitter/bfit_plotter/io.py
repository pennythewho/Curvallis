import csv, re
from collections import namedtuple
import numpy as np
from bfit import fit

xcol = 'x'
ycol = 'y'
wcol = 'w'
dsig = 'd'
dy_template = 'd{0}y'
dyw_template = 'd{0}w'
dre = re.compile(r'd(\d+)({0}|{1})'.format(ycol, wcol))

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
    cols = _get_expected_cols(p, with_weights)
    out = _initialize_output(cols)
    try:
        with open(path) as f:
            ln = f.readline()
            while ln:
                out = _parse_line(out, cols, ln)
                ln = f.readline()
    except FileNotFoundError as fe:
        raise fe
    except Exception as e:
        raise IOError('Cannot load data from '+path+' -- '+str(e))
    out = _remove_empty_keys(out)
    return out

def _get_expected_cols(p, with_weights):
    """ returns an ordered list of columns in the data file """
    cols = [xcol, ycol]
    if 0 in with_weights:
        cols = cols + [wcol]
    for d in range(1, p+1):
        cols = cols + [dy_template.format(d)] + ([dyw_template.format(d)] if d in with_weights else [])
    return cols


def _get_kw(deriv, is_weight=False):
    return '_'.join([fit.set_kw, dsig+deriv, wcol if is_weight else xcol])


def _initialize_output(cols):
    def _get_empty_fvect():
        return np.empty((0,), dtype=float)
    out = dict()
    for c in cols:
        if c[0] in [xcol, ycol, wcol]:
            out[c] = _get_empty_fvect()
        elif c[0] == dsig:
            cm = dre.match(c)
            is_weight = cm[2] == wcol
            out[_get_kw(cm[1], is_weight)] = _get_empty_fvect() if is_weight else np.empty((0, 2), dtype=float)
    return out


def _parse_line(out, cols, line, delim=','):
    vals = line.split(delim)
    nv = {c: float(v) for c, v in zip(cols, vals) if bool(v and v.strip())}
    try:
        x = nv[xcol]
    except KeyError:
        raise ValueError('Every line must have an x value, but one is missing from line {0}.'.format(line))
    if ycol in nv:
        out[xcol] = np.append(out[xcol], x)
        out[ycol] = np.append(out[ycol], nv[ycol])
        if wcol in cols:
            try:
                out[wcol] = np.append(out[wcol], nv[wcol])
            except KeyError:
                raise ValueError('The spec for this file indicates that if a y value is provided, a weight is also required.')
    for dm in [dre.match(cn) for cn in nv.keys() if dre.match(cn)]:
        cn, deriv, ct = [dm[i] for i in range(3)]  # column name, deriv, type
        if ct == ycol:   # handle weights only with y match
            kw = _get_kw(deriv)
            out[kw] = np.vstack((out[kw], (x, nv[cn])))
            assoc_wt_col = dyw_template.format(deriv)
            if assoc_wt_col in cols:
                kww = _get_kw(deriv, True)
                try:
                    out[kww] = np.append(out[kww], nv[assoc_wt_col])
                except KeyError:
                    raise ValueError('The spec for this file indicates that if a value for the {0} derivative is provided, a weight is also required.'.format(deriv))
    return out


def _remove_empty_keys(out):
    # removes unused derivatives
    return {k: v for (k, v) in out.items() if len(v) > 0}





