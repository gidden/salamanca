from __future__ import division

import copy
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


def theil_lo_rule(m, b=0.05):
    lhs = model_T_b(m) + model_T_w(m, income_from_data=False)
    rhs = m.data['T']
    return lhs >= (1 - b) * rhs


def theil_hi_rule(m, b=0.05):
    lhs = model_T_b(m) + model_T_w(m, income_from_data=False)
    rhs = m.data['T']
    return lhs <= (1 + b) * rhs


def threshold_lo_rule(m, f=1.0, b=0.05, relative=True):
    i = m.data['I']
    x = f * i if relative else f
    rhs = below_threshold(x, i, m.data['T'])
    lhs = sum(m.data['n_frac'][idx] * below_threshold(x, m.i[idx], m.t[idx])
              for idx in m.idxs)
    return lhs >= (1 - b) * rhs


def threshold_hi_rule(m, f=1.0, b=0.05, relative=True):
    i = m.data['I']
    x = f * i if relative else f
    rhs = below_threshold(x, i, m.data['T'])
    lhs = sum(m.data['n_frac'][idx] * below_threshold(x, m.i[idx], m.t[idx])
              for idx in m.idxs)
    return lhs <= (1 + b) * rhs


def theil_diff_hi_rule(m, idx, b):
    """
    \frac{t^{t+1} - t^t}{t^{t}} \geq -0.1

    @TODO: is 10% in 10 years (or other timeperiod) reasonable?
    """
    return m.t[idx] >= (1 - b) * m.data['t'][idx]


def theil_diff_lo_rule(m, idx, b):
    """
    \frac{t^{t+1} - t^t}{t^{t}} \leq 0.1

    @TODO: is 10% in 10 years (or other timeperiod) reasonable?
    """
    return m.t[idx] <= (1 + b) * m.data['t'][idx]


def income_diff_hi_rule(m, idx, b):
    """
    \frac{\iota^{t+1} - \iota^t}{\iota^{t}} \geq -0.2

    \iota^t = \frac{i^t}{I^t}

    @TODO: is 20% in 10 years (or other timeperiod) reasonable?
    """
    return m.i[idx] >= (1 - b) * m.data['i'][idx] * m.data['I'] / m.data['I_old']


def income_diff_lo_rule(m, idx, b):
    """
    \frac{\iota^{t+1} - \iota^t}{\iota^{t}} \leq 0.2

    \iota^t = \frac{i^t}{I^t}

    @TODO: is 20% in 10 years (or other timeperiod) reasonable?
    """
    return m.i[idx] <= (1 + b) * m.data['i'][idx] * m.data['I'] / m.data['I_old']


def share_diff_hi_rule(m, idx, b):
    """
    \frac{s^{t+1} - s^t}{s^{t}} \geq -0.2

    s^t = \frac{i^t n^t}{I^t N^t}

    @TODO: is 20% in 10 years (or other timeperiod) reasonable?
    """
    lhs = m.i[idx] * m.data['n_frac'][idx] / m.data['I']
    rhs = m.data['i'][idx] * m.data['n_frac_old'][idx] / m.data['I_old']
    return lhs >= (1 - b) * rhs


def share_diff_lo_rule(m, idx, b):
    """
    \frac{s^{t+1} - s^t}{s^{t}} \leq 0.2

    s^t = \frac{i^t n^t}{I^t N^t}

    @TODO: is 20% in 10 years (or other timeperiod) reasonable?
    """
    lhs = m.i[idx] * m.data['n_frac'][idx] / m.data['I']
    rhs = m.data['i'][idx] * m.data['n_frac_old'][idx] / m.data['I_old']
    return lhs <= (1 + b) * rhs


def income_direction_rule(m, idx):
    dI = m.data['I_new'] - m.data['I_old']
    if dI >= 0:
        return m.i[idx] >= m.data['i'][idx] / m.data['I_new']
    else:
        return m.i[idx] <= m.data['i'][idx] / m.data['I_new']


def theil_direction_rule(m, idx):
    dT = m.data['T'] - m.data['T_old']
    if dT >= 0:
        return m.t[idx] >= m.data['t'][idx]
    else:
        return m.t[idx] <= m.data['t'][idx]

#
# Objectives
#


def theil_total_sum_obj(m):
    T_b = model_T_b(m)
    T_w = model_T_w(m, income_from_data=False)
    return (m.data['T'] - T_b - T_w) ** 2


def population_below_obj(m, f=1.0, relative=True):
    i = m.data['I']
    x = f * i if relative else f
    p_nat = below_threshold(x, i, m.data['T'])
    p_sum = sum(m.data['n_frac'][idx] * below_threshold(x, m.i[idx], m.t[idx])
                for idx in m.idxs)
    return (p_nat - p_sum) ** 2


def combined_obj(m, theil_weight=1.0, pop_weight=1.0, **pop_kwargs):
    return \
        theil_weight * theil_total_sum_obj(m) + \
        pop_weight * population_below_obj(m, **pop_kwargs)


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
        n_s = sdf.loc[modelidx][n].sum()
        n_n = ndf.loc[modelidx][n]
        if not np.isclose(n_s, n_n):
            msg = 'Subnational ({}) != national ({}) population using ratio: {}'
            raise RuntimeError(msg.format(n_s, n_n, ratio))

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
        self.scale_I = ndf.loc[modelidx][i]
        self.model_data = {
            'idxs': self.model_idx,
            'n_frac_old': sdf.loc[histidx][n].values / ndf.loc[histidx][n],
            'n_frac': sdf.loc[modelidx][n].values / ndf.loc[modelidx][n],
            'i': sdf.loc[histidx][i].values,
            'i_min': 0.1 * np.min(sdf.loc[histidx][i].values) / ndf.loc[histidx][i],
            'i_max': 10 * np.max(sdf.loc[histidx][i].values) / ndf.loc[histidx][i],
            't': ineq.gini_to_theil(ginis,
                                    empirical=self.empirical),
            't_min': ineq.gini_to_theil(gini_min,
                                        empirical=self.empirical),
            't_max': ineq.gini_to_theil(gini_max,
                                        empirical=self.empirical),
            'I': 1.0,
            'I_new': ndf.loc[modelidx][i],
            'I_old': ndf.loc[histidx][i],
            'T': ineq.gini_to_theil(ndf.loc[modelidx][gini],
                                    empirical=self.empirical),
            'T_old': ineq.gini_to_theil(ndf.loc[histidx][gini],
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
            i: self.solution[i] * self.scale_I,
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

    with optionally (assuming equally spaced time steps):

    - maximum theil diffusion % per year
    - maximum income share diffusion % per year
    - incomes track with national income
    - theils track with national theil
    """

    def construct(self, diffusion={}, direction={}):
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
        # optional constraints
        b = diffusion.pop('income', False)
        if b:
            b = 0.2 if b is True else b
            m.i_hi = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: income_diff_hi_rule(m, idx, b),
                doc='income within 20% from past',
            )
            m.i_lo = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: income_diff_lo_rule(m, idx, b),
                doc='income within 20% from past',
            )
        b = diffusion.pop('share', False)
        if b:
            b = 0.2 if b is True else b
            m.s_hi = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: share_diff_hi_rule(m, idx, b),
                doc='income share within 20% from past',
            )
            m.s_lo = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: share_diff_lo_rule(m, idx, b),
                doc='income share within 20% from past',
            )
        b = diffusion.pop('theil', False)
        if b:
            b = 0.1 if b is True else b
            m.t_hi = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: theil_diff_hi_rule(m, idx, b),
                doc='income share within 20% from past',
            )
            m.t_lo = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: theil_diff_lo_rule(m, idx, b),
                doc='income share within 20% from past',
            )
        if direction.pop('income', False):
            m.i_dir = mo.Constraint(
                m.idxs,
                rule=income_direction_rule,
                doc='income must track with national values',
            )
        if direction.pop('theil', False):
            m.t_dir = mo.Constraint(
                m.idxs,
                rule=theil_direction_rule,
                doc='theil must track with national values',
            )
        # Objective
        m.obj = mo.Objective(rule=theil_total_sum_obj, sense=mo.minimize)
        return self


class Model2(Model):
    """
    Minimize L2 norm of population under income threshold subject to:

    - GDP sum
    - constrained Theil


    with optionally (assuming equally spaced time steps):

    - maximum theil diffusion % per year
    - maximum income share diffusion % per year
    - incomes track with national income
    - theils track with national theil
    """

    def construct(self, diffusion={}, direction={},
                  theil_within=0.05, threshold=1.0):
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
        m.gdp_sum = mo.Constraint(rule=gdp_sum_rule, doc='gdp sum = gdp')
        m.theil_lo = mo.Constraint(
            rule=lambda m: theil_lo_rule(m, b=theil_within),
            doc='Theil within 5%'
        )
        m.theil_hi = mo.Constraint(
            rule=lambda m: theil_hi_rule(m, b=theil_within),
            doc='Theil within 5%'
        )
        # optional constraints
        b = diffusion.pop('income', False)
        if b:
            b = 0.2 if b is True else b
            m.i_hi = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: income_diff_hi_rule(m, idx, b),
                doc='income within 20% from past',
            )
            m.i_lo = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: income_diff_lo_rule(m, idx, b),
                doc='income within 20% from past',
            )
        b = diffusion.pop('share', False)
        if b:
            b = 0.2 if b is True else b
            m.s_hi = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: share_diff_hi_rule(m, idx, b),
                doc='income share within 20% from past',
            )
            m.s_lo = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: share_diff_lo_rule(m, idx, b),
                doc='income share within 20% from past',
            )
        b = diffusion.pop('theil', False)
        if b:
            b = 0.1 if b is True else b
            m.t_hi = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: theil_diff_hi_rule(m, idx, b),
                doc='income share within 20% from past',
            )
            m.t_lo = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: theil_diff_lo_rule(m, idx, b),
                doc='income share within 20% from past',
            )
        if direction.pop('income', False):
            m.i_dir = mo.Constraint(
                m.idxs,
                rule=income_direction_rule,
                doc='income must track with national values',
            )
        if direction.pop('theil', False):
            m.t_dir = mo.Constraint(
                m.idxs,
                rule=theil_direction_rule,
                doc='theil must track with national values',
            )
        # Objective
        m.obj = mo.Objective(
            rule=lambda m: population_below_obj(m, f=threshold),
            sense=mo.minimize
        )
        return self


class Model3(Model):
    """
    Minimize sum of L2 norms of 
    - theil
    - population under income threshold 

    subject to:

    - GDP sum

    with optionally (assuming equally spaced time steps):

    - maximum theil diffusion % per year
    - maximum income share diffusion % per year
    - incomes track with national income
    - theils track with national theil
    """

    def construct(self, diffusion={}, direction={},
                  pop_weight=1.0, theil_weight=1.0, threshold=1.0):
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
        m.gdp_sum = mo.Constraint(rule=gdp_sum_rule, doc='gdp sum = gdp')
        # optional constraints
        b = diffusion.pop('income', False)
        if b:
            b = 0.2 if b is True else b
            m.i_hi = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: income_diff_hi_rule(m, idx, b),
                doc='income within 20% from past',
            )
            m.i_lo = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: income_diff_lo_rule(m, idx, b),
                doc='income within 20% from past',
            )
        b = diffusion.pop('share', False)
        if b:
            b = 0.2 if b is True else b
            m.s_hi = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: share_diff_hi_rule(m, idx, b),
                doc='income share within 20% from past',
            )
            m.s_lo = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: share_diff_lo_rule(m, idx, b),
                doc='income share within 20% from past',
            )
        b = diffusion.pop('theil', False)
        if b:
            b = 0.1 if b is True else b
            m.t_hi = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: theil_diff_hi_rule(m, idx, b),
                doc='income share within 20% from past',
            )
            m.t_lo = mo.Constraint(
                m.idxs,
                rule=lambda m, idx: theil_diff_lo_rule(m, idx, b),
                doc='income share within 20% from past',
            )
        if direction.pop('income', False):
            m.i_dir = mo.Constraint(
                m.idxs,
                rule=income_direction_rule,
                doc='income must track with national values',
            )
        if direction.pop('theil', False):
            m.t_dir = mo.Constraint(
                m.idxs,
                rule=theil_direction_rule,
                doc='theil must track with national values',
            )
        # Objective
        m.obj = mo.Objective(
            rule=lambda m: combined_obj(m, theil_weight=theil_weight,
                                        pop_weight=pop_weight, f=threshold),
            sense=mo.minimize
        )
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

    def __init__(self, natdata, subdata, model, constructor_kwargs={}, model_kwargs={}):
        self.natdata = natdata.copy()
        self.subdata = subdata.copy()
        self._orig_idx = subdata.index
        self._result = subdata.copy().sort_index()
        self.Model = model
        self.constructor_kwargs = constructor_kwargs
        self.model_kwargs = model_kwargs

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

    def make_model(self, t1, t2):
        self.solve_t = t2
        ndf, sdf = self._data(t1, t2)
        # explicitly copies so pop can be used
        kwargs = copy.deepcopy(self.constructor_kwargs)
        self.model = self.Model(ndf, sdf, **kwargs)
        kwargs = copy.deepcopy(self.model_kwargs)
        self.model.construct(**kwargs)
        return self

    def solve(self, *args, **kwargs):
        self.model.solve(*args, **kwargs)
        df = self.model.result()
        self._update(self.solve_t, df)
        return self
