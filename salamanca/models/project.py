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
    return (m.data['T'] - T_b - T_w) ** 2


class Model(object):
    """Base class for Projection Models"""

    def __init__(self, natdata, subdata, empirical=False):
        self.natdata = natdata
        self.subdata = subdata
        self.empirical = empirical

        self._setup_model_data(natdata, subdata)
        self._check_model_data()

    def _setup_model_data(self, natdata, subdata):
        required = (n, i, gini) = 'n', 'i', 'gini'
        ndf, sdf = self.natdata.copy(), self.subdata.copy()
        msg = 'Must include all of {} in {} data'
        if any(x not in ndf for x in required):
            raise ValueError(msg.format(required, 'national'))
        if any(x not in sdf for x in required):
            raise ValueError(msg.format(required, 'subnational'))
        if len(ndf.index) != 2:
            raise ValueError('National data does not have 2 entries')

        self.histidx = histidx = ndf.index[0]
        self.modelidx = modelidx = ndf.index[1]

        # correct population
        sdf.loc[modelidx, n] *= ndf.loc[modelidx, n] / \
            sdf.loc[modelidx, n].sum()

        # # save index of sorted values
        # self.orig_idx = sdf.index
        # sdf = sdf.sort_values(by=gini)
        # self.sorted_idx = sdf.index
        # sdf = sdf.reset_index()
        # self.model_idx = sdf.index  # must be ordered

        # save model data
        self.model_data = {
            'idxs': self.model_idx.values,
            'n': sdf.loc[modelidx, n].values,
            'i': sdf.loc[histidx, i].values,
            't': ineq.gini_to_theil(sdf.loc[histidx, gini].values,
                                    empirical=self.empirical),
            'N': ndf.loc[modelidx, n],
            'I': ndf.loc[modelidx, i],
            'I_old': ndf.loc[histidx, i],
            'G': ndf.loc[modelidx, n] * ndf.loc[modelidx, i],
            'T': ineq.gini_to_theil(ndf.loc[modelidx, gini],
                                    empirical=self.empirical),
        }

    def _check_model_data(self):
        obs = self.model_data.loc[self.modelidx, 'N']
        exp = np.sum(self.model_data.loc[self.modelidx, 'n'])
        if not np.isclose(obs, exp):
            raise ValueError('Population values do not sum to national')

    def construct(self):
        raise NotImplementedError()

    def solve(self):
        m = self.model
        solver = mo.SolverFactory('ipopt')
        result = solver.solve(m)  # , tee=True)
        result.write()
        m.solutions.load_from(result)
        self.solution = pd.Series(m.t.get_values().values(), name='thiel')
        return self

    # def result(self):
    #     n, g, i, gini = 'n', 'g', 'i', 'gini'
    #     df = pd.DataFrame({
    #         i: self.model_data[g] / self.model_data[n],
    #         n: self.model_data[n],
    #         gini: ineq.theil_to_gini(self.solution, empirical=self.empirical),
    #     })
    #     df.index = self.sorted_idx

    #     df = df.loc[self.orig_idx]
    #     df['i_orig'] = self.subdata[i]
    #     df['n_orig'] = self.subdata[n]
    #     df['gini_orig'] = self.subdata[gini]
    #     return df
