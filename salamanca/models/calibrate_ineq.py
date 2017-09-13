from __future__ import division

import itertools
import math

import numpy as np
import pandas as pd
import pyomo.environ as mo


from salamanca import ineq
from salamanca.models.utils import below_threshold, model_T_w, t_std


#
# Constraints
#


def position_rule(m, idx):
    r"""|pos|

    .. |pos| replace:: :math:`t_{r-1} \leq t_{r} \forall r \in {r_2 \ldots r_N}`
    """
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx] >= m.t[idx - 1]


def diff_hi_rule(m, idx, b=0.2):
    r"""|diff_hi|

    .. |diff_hi| replace:: :math:`t_r - t_{r-1} \geq 0.8 (t^*_r - t^*_{r-1}) \forall r \in {r_2 \ldots r_N}`
    """
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx] - m.t[idx - 1] >= (1 - b) * (m.data['t'][idx] - m.data['t'][idx - 1])


def diff_lo_rule(m, idx, b=0.2):
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx] - m.t[idx - 1] <= (1 + b) * (m.data['t'][idx] - m.data['t'][idx - 1])


def spacing_rule(m, idx):
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx] - m.t[idx - 1] >= 0.5 * (m.data['t'][idx] - m.data['t'][idx - 1])


def threshold_lo_rule(m, idx, f=1.0, b=0.9, relative=True):
    i = m.data['i'][idx]
    x = f * i if relative else f
    rhs = below_threshold(x, i, m.data['t'][idx])
    lhs = below_threshold(x, i, m.t[idx])
    return lhs >= b * rhs


def threshold_hi_rule(m, idx, f=1.0, b=1.1, relative=True):
    i = m.data['i'][idx]
    x = f * i if relative else f
    rhs = below_threshold(x, i, m.data['t'][idx])
    lhs = below_threshold(x, i, m.t[idx])
    return lhs <= b * rhs


def theil_sum_lo_rule(m, b=0.1):
    lhs = sum(m.t[idx] * m.data['g'][idx] for idx in m.idxs)
    rhs = m.data['T_w'] * m.data['G']
    return lhs >= (1 - b) * rhs


def theil_sum_hi_rule(m, b=0.1):
    lhs = sum(m.t[idx] * m.data['g'][idx] for idx in m.idxs)
    rhs = m.data['T_w'] * m.data['G']
    return lhs <= (1 + b) * rhs


def theil_sum_rule(m):
    return sum(m.t[idx] * m.data['g'][idx] for idx in m.idxs) == \
        m.data['T_w'] * m.data['G']


def std_diff_hi_rule(m, b=0.1):
    rhs = t_std(m, from_data=True)
    lhs = t_std(m, from_data=False)
    return lhs >= (1 - b) * rhs


def std_diff_lo_rule(m, b=0.1):
    rhs = t_std(m, from_data=True)
    lhs = t_std(m, from_data=False)
    return lhs <= (1 + b) * rhs

#
# Objectives
#


def l2_norm_obj(m):
    return sum((m.t[idx] - m.data['t'][idx]) ** 2
               for idx in m.idxs)


def theil_sum_obj(m):
    T_w = model_T_w(m, income_from_data=True)
    return (m.data['T_w'] - T_w) ** 2


def threshold_obj(m, factors=[1.0], weights=[1.0], relative=True):
    i = m.data['i']
    if relative:
        x = lambda m, idx, f: below_threshold(
            f * i[idx], i[idx], m.data['t'][idx])
        y = lambda m, idx, f: below_threshold(f * i[idx], i[idx], m.t[idx])
    else:
        x = lambda m, idx, f: below_threshold(f, i[idx], m.data['t'][idx])
        y = lambda m, idx, f: below_threshold(f, i[idx], m.t[idx])

    n = m.data['n']
    return sum(
        (w * n[idx] * (x(m, idx, f) - y(m, idx, f))) ** 2
        for (f, w), idx in itertools.product(zip(factors, weights), m.idxs))


def national_threshold_obj(m, f=1.0, relative=True):
    I = m.data['I']
    x = f * I if relative else f
    lhs = sum(
        m.data['n_frac'][idx] * below_threshold(x, m.data['i'][idx], m.t[idx])
        for idx in m.idxs
    )
    rhs = below_threshold(x, I, m.data['T'])
    return (lhs - rhs) ** 2


class Model(object):
    """Base class for Inequality Calibration Models"""

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

        # correct income by scaling
        sdf[n] *= ndf[n] / sdf[n].sum()
        assert(np.isclose(ndf[n], sdf[n].sum()))
        sdf[i] *= (ndf[n] * ndf[i]) / (sdf[n] * sdf[i]).sum()
        assert(np.isclose(ndf[n] * ndf[i], (sdf[n] * sdf[i]).sum()))

        # calculate national within theil
        T = ineq.gini_to_theil(ndf[gini], empirical=self.empirical)
        gfrac = (sdf[n] * sdf[i]) / (ndf[n] * ndf[i])
        nfrac = sdf[n] / ndf[n]
        T_b = np.sum(gfrac * np.log(gfrac / nfrac))
        T_w = T - T_b
        assert(T_w > 0 and T > T_w)
        assert(T_b > 0 and T > T_b)

        # save index of sorted values
        self.orig_idx = sdf.index
        sdf = sdf.sort_values(by=gini)
        self.sorted_idx = sdf.index
        sdf = sdf.reset_index()
        self.model_idx = sdf.index  # must be ordered

        # save model data
        self.model_data = {
            'idxs': self.model_idx.values,
            'n': sdf[n].values,
            'n_frac': sdf[n].values / ndf[n],
            'i': sdf[i].values,
            'g': (sdf[n] * sdf[i]).values,
            't': ineq.gini_to_theil(sdf[gini].values, empirical=self.empirical),
            'N': ndf[n],
            'I': ndf[i],
            'G': ndf[n] * ndf[i],
            'T_w': T_w,
            'T': T,
        }

    def _check_model_data(self):
        obs = self.model_data['N']
        exp = np.sum(self.model_data['n'])
        if not np.isclose(obs, exp):
            raise ValueError('Population values do not sum to national')

        obs = self.model_data['G']
        exp = np.sum(self.model_data['g'])
        if not np.isclose(obs, exp):
            raise ValueError('GDP values do not sum to national')

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

    def result(self):
        n, g, i, gini = 'n', 'g', 'i', 'gini'
        df = pd.DataFrame({
            i: self.model_data[g] / self.model_data[n],
            n: self.model_data[n],
            gini: ineq.theil_to_gini(self.solution, empirical=self.empirical),
        })

        # set up index
        df.index = self.sorted_idx
        df = df.loc[self.orig_idx]

        df['i_orig'] = self.subdata[i]
        df['n_orig'] = self.subdata[n]
        df['gini_orig'] = self.subdata[gini]
        return df


class Model1(Model):
    """
    Minimize L2-norm under position and constrainted relative difference and Theil sum.

    Comprised of

    | |pos|
    | |diff_hi|

    """

    def construct(self):
        self.model = m = mo.ConcreteModel()
        # Model Data
        m.data = self.model_data
        # Sets
        m.idxs = mo.Set(initialize=m.data['idxs'])
        # Variables
        m.t = mo.Var(m.idxs, within=mo.PositiveReals,
                     bounds=(1e-5, ineq.MAX_THEIL))
        # Constraints
        m.position = mo.Constraint(m.idxs, rule=position_rule,
                                   doc='ordering between provinces must be maintained')
        m.diff_hi = mo.Constraint(m.idxs, rule=diff_hi_rule,
                                  doc='difference between values should be about the same')
        m.diff_lo = mo.Constraint(m.idxs, rule=diff_lo_rule,
                                  doc='difference between values should be about the same')
        m.theil_sum = mo.Constraint(rule=theil_sum_rule, doc='factor ordering')
        # Objective
        m.obj = mo.Objective(rule=l2_norm_obj, sense=mo.minimize)
        return self


class Model2(Model):
    """
    Minimize L2-norm under position and constrainted CDF (relative income) and Theil sum.
    """

    def construct(self):
        self.model = m = mo.ConcreteModel()
        # Model Data
        m.data = self.model_data
        # Sets
        m.idxs = mo.Set(initialize=m.data['idxs'])
        # Variables
        m.t = mo.Var(m.idxs, within=mo.PositiveReals,
                     bounds=(1e-5, ineq.MAX_THEIL))
        # Constraints
        m.position = mo.Constraint(m.idxs, rule=position_rule,
                                   doc='ordering between provinces must be maintained')
        m.thresh_hi = mo.Constraint(m.idxs, rule=threshold_hi_rule,
                                    doc='')
        m.thresh_lo = mo.Constraint(m.idxs, rule=threshold_lo_rule,
                                    doc='')
        m.theil_sum = mo.Constraint(rule=theil_sum_rule, doc='')
        # Objective
        m.obj = mo.Objective(rule=l2_norm_obj, sense=mo.minimize)
        return self


class Model3(Model):
    """
    Minimize Theil sum under position and constrainted CDF.

    This is subject to having multiple optima within the constrained threshold space.
    """

    def construct(self):
        self.model = m = mo.ConcreteModel()
        # Model Data
        m.data = self.model_data
        # Sets
        m.idxs = mo.Set(initialize=m.data['idxs'])
        # Variables
        m.t = mo.Var(m.idxs, within=mo.PositiveReals,
                     bounds=(1e-5, ineq.MAX_THEIL))
        # Constraints
        m.position = mo.Constraint(m.idxs, rule=position_rule,
                                   doc='ordering between subgroups must be maintained')
        m.thresh_hi = mo.Constraint(m.idxs, rule=threshold_hi_rule,
                                    doc='')
        m.thresh_lo = mo.Constraint(m.idxs, rule=threshold_lo_rule,
                                    doc='')
        # Objective
        m.obj = mo.Objective(rule=theil_sum_obj, sense=mo.minimize)
        return self


class Model4(Model):
    """
    Minimize population difference below threshold under position and constrained Theil sum.
    """

    def construct(self):
        self.model = m = mo.ConcreteModel()
        # Model Data
        m.data = self.model_data
        # Sets
        m.idxs = mo.Set(initialize=m.data['idxs'])
        # Variables
        m.t = mo.Var(m.idxs, within=mo.PositiveReals,
                     bounds=(1e-5, ineq.MAX_THEIL))
        # Constraints
        m.position = mo.Constraint(m.idxs, rule=position_rule,
                                   doc='ordering between provinces must be maintained')
        m.theil_sum = mo.Constraint(rule=theil_sum_rule, doc='')
        # Objective
        m.obj = mo.Objective(rule=threshold_obj, sense=mo.minimize)
        return self


class Model4b(Model):
    """
    Minimize population difference below threshold under position and constrained Theil sum.
    """

    def construct(self):
        self.model = m = mo.ConcreteModel()
        # Model Data
        m.data = self.model_data
        # Sets
        m.idxs = mo.Set(initialize=m.data['idxs'])
        # Variables
        m.t = mo.Var(m.idxs, within=mo.PositiveReals,
                     bounds=(1e-5, ineq.MAX_THEIL))
        # Constraints
        m.position = mo.Constraint(m.idxs, rule=position_rule,
                                   doc='ordering between provinces must be maintained')
        m.theil_sum = mo.Constraint(rule=theil_sum_rule, doc='')
        # Objective
        factors = [10 * 365, m.data['I']]
        weights = [2, 1]
        rule = lambda m: threshold_obj(m, factors=factors, weights=weights)
        m.obj = mo.Objective(rule=rule, sense=mo.minimize)
        return self


class Model5(Model):
    """
    Minimize L2-norm under position and constrainted CDF (relative income) and Theil sum.
    """

    def construct(self):
        self.model = m = mo.ConcreteModel()
        # Model Data
        m.data = self.model_data
        # Sets
        m.idxs = mo.Set(initialize=m.data['idxs'])
        # Variables
        m.t = mo.Var(m.idxs, within=mo.PositiveReals,
                     bounds=(1e-5, ineq.MAX_THEIL))
        # Constraints
        m.position = mo.Constraint(m.idxs, rule=position_rule,
                                   doc='ordering between provinces must be maintained')
        m.thresh_hi = mo.Constraint(m.idxs, rule=threshold_hi_rule,
                                    doc='')
        m.thresh_lo = mo.Constraint(m.idxs, rule=threshold_lo_rule,
                                    doc='')
        m.theil_sum = mo.Constraint(rule=theil_sum_rule, doc='')
        m.spacing = mo.Constraint(m.idxs, rule=spacing_rule,
                                  doc='')
        m.std_lo = mo.Constraint(m.idxs, rule=std_diff_lo_rule,
                                 doc='')
        m.std_hi = mo.Constraint(m.idxs, rule=std_diff_hi_rule,
                                 doc='')
        # Objective
        m.obj = mo.Objective(rule=l2_norm_obj, sense=mo.minimize)
        return self


class Model6(Model):
    """
    Minimize low threshold L2-norm under position, std, and approx theil sum.
    """

    def construct(self):
        self.model = m = mo.ConcreteModel()
        # Model Data
        m.data = self.model_data
        # Sets
        m.idxs = mo.Set(initialize=m.data['idxs'])
        # Variables
        m.t = mo.Var(m.idxs, within=mo.PositiveReals,
                     bounds=(1e-5, ineq.MAX_THEIL))
        # Constraints
        m.position = mo.Constraint(m.idxs, rule=position_rule,
                                   doc='ordering between provinces must be maintained')
        m.theil_sum_hi = mo.Constraint(rule=theil_sum_hi_rule,
                                       doc='')
        m.theil_sum_lo = mo.Constraint(rule=theil_sum_lo_rule,
                                       doc='')
        m.spacing = mo.Constraint(m.idxs, rule=spacing_rule,
                                  doc='')
        m.std_lo = mo.Constraint(m.idxs, rule=std_diff_lo_rule,
                                 doc='')
        m.std_hi = mo.Constraint(m.idxs, rule=std_diff_hi_rule,
                                 doc='')
        # Objective
        m.obj = mo.Objective(
            rule=lambda m: national_threshold_obj(m, f=0.25),
            sense=mo.minimize
        )
        return self
