from __future__ import division

import itertools
import math

import numpy as np
import pandas as pd
import pyomo.environ as mo


from salamanca import ineq
from salamanca.models.utils import below_threshold, model_T_b, model_T_w

# quiet pandas setwithcopy warnings
pd.options.mode.chained_assignment = None  # default='warn'

#
# Constraints
#


def gdp_sum_rule(m):
    return sum(m.data['n_frac'][idx] * m.i[idx] for idx in m.idxs) == m.data['I']


def threshold_lo_rule(m, f=1.0, b=0.95, relative=True):
    i = m.data['I']
    x = f * i if relative else f
    rhs = below_threshold(x, i, m.data['T'])
    lhs = sum(m.data['n_frac'][idx] * below_threshold(x, m.i[idx], m.t[idx])
              for idx in m.idxs)
    return lhs >= b * rhs


def threshold_hi_rule(m, f=1.0, b=1.05, relative=True):
    i = m.data['I']
    x = f * i if relative else f
    rhs = below_threshold(x, i, m.data['T'])
    lhs = sum(m.data['n_frac'][idx] * below_threshold(x, m.i[idx], m.t[idx])
              for idx in m.idxs)
    return lhs <= b * rhs


def theil_diff_hi_rule(m, idx, b=0.9):
    """
    \frac{t^{t+1} - t^t}{t^{t}} \geq -0.1

    @TODO: is 10% in 10 years (or other timeperiod) reasonable?
    """
    return m.t[idx] >= b * m.data['t'][idx]


def theil_diff_lo_rule(m, idx, b=1.1):
    """
    \frac{t^{t+1} - t^t}{t^{t}} \leq 0.1

    @TODO: is 10% in 10 years (or other timeperiod) reasonable?
    """
    return m.t[idx] <= b * m.data['t'][idx]


def income_diff_hi_rule(m, idx, b=0.8):
    """
    \frac{\iota^{t+1} - \iota^t}{\iota^{t}} \geq -0.2

    \iota^t = \frac{i^t}{I^t}

    @TODO: is 20% in 10 years (or other timeperiod) reasonable?
    """
    return m.i[idx] >= b * m.data['i'][idx] * m.data['I'] / m.data['I_old']


def income_diff_lo_rule(m, idx, b=1.2):
    """
    \frac{\iota^{t+1} - \iota^t}{\iota^{t}} \leq 0.2

    \iota^t = \frac{i^t}{I^t}

    @TODO: is 20% in 10 years (or other timeperiod) reasonable?
    """
    return m.i[idx] <= b * m.data['i'][idx] * m.data['I'] / m.data['I_old']


def share_diff_hi_rule(m, idx, b=0.8):
    """
    \frac{s^{t+1} - s^t}{s^{t}} \geq -0.2

    s^t = \frac{i^t n^t}{I^t N^t}

    @TODO: is 20% in 10 years (or other timeperiod) reasonable?
    """
    lhs = m.i[idx] * m.data['n_frac'][idx] / m.data['I']
    rhs = m.data['i'][idx] * m.data['n_frac_old'][idx] / m.data['I_old']
    return lhs >= b * rhs


def share_diff_lo_rule(m, idx, b=1.2):
    """
    \frac{s^{t+1} - s^t}{s^{t}} \leq 0.2

    s^t = \frac{i^t n^t}{I^t N^t}

    @TODO: is 20% in 10 years (or other timeperiod) reasonable?
    """
    lhs = m.i[idx] * m.data['n_frac'][idx] / m.data['I']
    rhs = m.data['i'][idx] * m.data['n_frac_old'][idx] / m.data['I_old']
    return lhs <= b * rhs


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

        # save t-1 and t indicies
        self._histidx = histidx = ndf.index[0]
        self._modelidx = modelidx = ndf.index[1]

        # correct population
        ratio = ndf.loc[modelidx][n] / sdf.loc[modelidx][n].sum()
        sdf.loc[modelidx][n] *= ratio
        assert(sdf.loc[modelidx][n].sum() == ndf.loc[modelidx][n])

        # # save index of sorted values
        self.orig_idx = sdf.loc[modelidx].index
        # sdf = sdf.sort_values(by=gini)
        # self.sorted_idx = sdf.index
        # sdf = sdf.reset_index()
        self.model_idx = list(range(len(self.orig_idx)))  # must be ordered

        # save model data
        ginis = sdf.loc[histidx][gini].values
        gini_min = min(0.15, np.min(ginis))
        gini_max = max(0.85, np.max(ginis))
        self.model_data = {
            'idxs': self.model_idx,
            'n_frac_old': sdf.loc[histidx][n].values / ndf.loc[histidx][n],
            'n_frac': sdf.loc[modelidx][n].values / ndf.loc[modelidx][n],
            'i': sdf.loc[histidx][i].values,
            'i_min': 0.1 * np.min(sdf.loc[histidx][i].values),
            'i_max': 10 * np.max(sdf.loc[histidx][i].values),
            't': ineq.gini_to_theil(ginis,
                                    empirical=self.empirical),
            't_min': ineq.gini_to_theil(gini_min,
                                        empirical=self.empirical),
            't_max': ineq.gini_to_theil(gini_max,
                                        empirical=self.empirical),
            'I': ndf.loc[modelidx][i],
            'I_old': ndf.loc[histidx][i],
            'T': ineq.gini_to_theil(ndf.loc[modelidx][gini],
                                    empirical=self.empirical),
        }

    def _check_model_data(self):
        obs = 1
        exp = np.sum(self.model_data['n_frac'])
        if not np.isclose(obs, exp):
            raise ValueError('Population values do not sum to national')

    def construct(self):
        raise NotImplementedError()

    def debug(self, pth=''):
        skeys = ['idxs', 'n_frac_old', 'n_frac', 'i', 't',
                 't_min', 't_max', 'i_min', 'i_max']
        sdf = pd.DataFrame({s: self.model_data[s] for s in skeys},
                           index=self.orig_idx)
        sdf.to_csv(pth + 'sdf.csv')

        nkeys = ['I', 'I_old', 'T']
        ndf = pd.Series({n: self.model_data[n] for n in nkeys})
        ndf.to_csv(pth + 'ndf.csv')

    def solve(self, options={}, **kwargs):
        m = self.model
        solver = mo.SolverFactory('ipopt')
        for k, v in options.items():
            solver.options[k] = v
        result = solver.solve(m, **kwargs)
        result.write()
        m.solutions.load_from(result)
        self.solution = pd.DataFrame({
            'i': m.i.get_values().values(),
            't': m.t.get_values().values(),
        }, index=self.orig_idx)
        return self

    def result(self):
        nfrac, n, i, gini = 'n_frac', 'n', 'i', 'gini'
        df = pd.DataFrame({
            i: self.solution['i'],
            n: self.model_data[nfrac] * self.natdata.loc[self._modelidx][n],
            gini: ineq.theil_to_gini(self.solution['t'], empirical=self.empirical),
        }, index=self.orig_idx)

        df['n_orig'] = self.subdata.loc[self._modelidx][n]
        return df


class Model1(Model):
    """
    Minimize L2 norm of Theil difference under:

    - GDP sum
    - constrained CDF

    with optionally (assuming 10 year time steps):

    - maximum theil diffusion of 1% per year
    - maximum income share diffusion of 2% per year
    """

    def construct(self, with_diffusion=False):
        self.model = m = mo.ConcreteModel()
        # Model Data
        m.data = self.model_data
        # Sets
        m.idxs = mo.Set(initialize=m.data['idxs'])
        # Variables
        m.i = mo.Var(m.idxs, within=mo.PositiveReals,
                     bounds=(m.data['i_min'], m.data['i_max']))
        m.t = mo.Var(m.idxs, within=mo.PositiveReals,
                     bounds=(m.data['t_min'], m.data['t_max']))
        # Constraints
        m.gdp_sum = mo.Constraint(rule=gdp_sum_rule,
                                  doc='gdp sum = gdp')
        m.cdf_lo = mo.Constraint(rule=threshold_lo_rule,
                                 doc='Population under threshold within 5%')
        m.cdf_hi = mo.Constraint(rule=threshold_hi_rule,
                                 doc='Population under threshold within 5%')
        if with_diffusion:
            m.t_hi = mo.Constraint(m.idxs, rule=theil_diff_hi_rule,
                                   doc='theil within 10% from past')
            m.t_lo = mo.Constraint(m.idxs, rule=theil_diff_lo_rule,
                                   doc='theil within 10% from past')
            m.i_hi = mo.Constraint(m.idxs, rule=share_diff_hi_rule,
                                   doc='income share within 20% from past')
            m.i_lo = mo.Constraint(m.idxs, rule=share_diff_lo_rule,
                                   doc='income share within 20% from past')
        # Objective
        m.obj = mo.Objective(rule=theil_total_sum_obj, sense=mo.minimize)
        return self


class Runner(object):
    """Simple class that runs all projection values for a given full projection
    instance

    Example
    -------

    ```
    runner = Runner(natdata, subnatdata, Model1)
    for time1, time2 in runner.horizon():
        runner.project(time1, time2)
    result = runner.result()
    ```
    """

    def __init__(self, natdata, subdata, model, empirical=False):
        self.natdata = natdata.copy()
        self.subdata = subdata.copy()
        self._orig_idx = subdata.index
        self._result = subdata.copy().sort_index()
        self.Model = model
        self.empirical = empirical

        if natdata.isnull().values.any():
            raise ValueError('Null values found in national data')
        if subdata['n'].isnull().values.any():
            raise ValueError('Null values found in subnational data')

    def horizon(self):
        return zip(self.natdata.index[:-1], self.natdata.index[1:])

    def result(self):
        return self._result.loc[self._orig_idx]

    def _data(self, t1, t2):
        ndf = self.natdata.loc[t1:t2]
        sdf = self._result.loc[t1:t2]
        return ndf, sdf

    def _update(self, t, df):
        df = df.sort_index()
        idx = (t, list(df.index.values))
        self._result.loc[idx, 'i'] = df['i'].values
        self._result.loc[idx, 'gini'] = df['gini'].values

    def make_model(self, t1, t2, *args, **kwargs):
        self.solve_t = t2
        ndf, sdf = self._data(t1, t2)
        self.model = self.Model(ndf, sdf, empirical=self.empirical)
        self.model.construct(*args, **kwargs)
        return self

    def solve(self, *args, **kwargs):
        self.model.solve(*args, **kwargs)
        df = self.model.result()
        self._update(self.solve_t, df)
        return self
