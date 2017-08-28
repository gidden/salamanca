from __future__ import division

import itertools
import math

import numpy as np
import pandas as pd
import pyomo.environ as mo


from salamanca import ineq
from salamanca.models.utils import below_threshold, model_T_b, model_T_w

#
# Constraints
#


def gdp_sum_rule(m):
    return sum(m.data['n'][idx] * m.i[idx] for idx in m.idxs) == m.data['G']


def threshold_lo_rule(m, idx, f=1.0, b=0.95, relative=True):
    i = m.data['I']
    x = f * i if relative else f
    rhs = m.data['N'] * below_threshold(x, i, m.data['T'])
    lhs = sum(m.data['n'][idx] * below_threshold(x, m.i[idx], m.t[idx])
              for idx in m.idxs)
    return lhs >= b * rhs


def threshold_hi_rule(m, idx, f=1.0, b=1.05, relative=True):
    i = m.data['I']
    x = f * i if relative else f
    rhs = m.data['N'] * below_threshold(x, i, m.data['T'])
    lhs = sum(m.data['n'][idx] * below_threshold(x, m.i[idx], m.t[idx])
              for idx in m.idxs)
    return lhs <= b * rhs


def theil_diff_hi_rule(m, idx, b=0.9):
    """
    \frac{t^{t+1} - t^t}{t^{t}} \geq -0.1

    @TODO: is 10% in 10 years (or other timeperiod) reasonable?
    """
    return m.t[idx] >= b * m.data['t']


def theil_diff_lo_rule(m, idx, b=1.1):
    """
    \frac{t^{t+1} - t^t}{t^{t}} \leq 0.1

    @TODO: is 10% in 10 years (or other timeperiod) reasonable?
    """
    return m.t[idx] <= b * m.data['t']


def income_diff_hi_rule(m, idx, b=0.9):
    """
    \frac{s^{t+1} - s^t}{s^{t}} \geq -0.1

    s^t = \frac{i^t}{I^t}

    @TODO: is 10% in 10 years (or other timeperiod) reasonable?
    """
    return m.i[idx] / m.data['I'] >= b * m.data['i'] / m.data['I_old']


def income_diff_lo_rule(m, idx, b=1.1):
    """
    \frac{s^{t+1} - s^t}{s^{t}} \leq -0.1

    s^t = \frac{i^t}{I^t}

    @TODO: is 10% in 10 years (or other timeperiod) reasonable?
    """
    return m.i[idx] / m.data['I'] <= b * m.data['i'] / m.data['I_old']


#
# Objectives
#


def theil_total_sum_obj(m):
    T_b = model_T_b(m)
    T_w = model_T_w(m, income_from_data=False)
    return (m.data['T_w'] - T_b - T_w) ** 2
