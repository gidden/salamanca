from __future__ import division

import itertools
import math

import numpy as np
import pandas as pd
import pyomo.environ as mo


from salamanca import ineq
from salamanca.models.utils import below_threshold

#
# Constraints
#


#
# Objectives
#

def theil_total_sum_obj(m):
    _t_w = lambda m, idx: m.i[idx] * \
        m.data['n'][idx] / m.data['G'] * m.t[idx]
    T_w = sum(_t_w(m, idx) for idx in m.idxs)
    T_b = sum(_t_b(m, idx) for idx in m.idxs)
    return (m.data['T'] - T_w - T_b) ** 2
