from __future__ import division

import numpy as np
import pandas as pd
import pyomo.environ as mo


from salamanca import ineq

#
# Constraints
#


def position_rule(m, idx):
    r"""|pos|

    .. |pos| replace:: :math:`t_{r-1} \leq t_{r} \forall r \in {r_2 \ldots r_N}`
    """
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx - 1] <= m.t[idx]


def diff_hi_rule(m, idx, b=0.8):
    r"""|diff_hi|

    .. |diff_hi| replace:: :math:`t_r - t_{r-1} \geq 0.8 (t^*_r - t^*_{r-1}) \forall r \in {r_2 \ldots r_N}`
    """
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx] - m.t[idx - 1] >= b * (m.data['t'][idx] - m.data['t'][idx - 1])


def diff_lo_rule(m, idx, b=1.2):
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx] - m.t[idx - 1] <= b * (m.data['t'][idx] - m.data['t'][idx - 1])


def theil_sum_rule(m):
    return sum(m.t[idx] * m.data['g'][idx] for idx in m.idxs) == \
        m.data['T_w'] * m.data['G']


def threshold_lo_rule(m, idx, f=0.5, b=0.9):
    dist = ineq.LogNormal()
    emp = m.data['empirical']
    i = m.data['i'][idx]
    rhs = dist.below_threshold(f * i, theil=m.data['t'][idx], empirical=emp)
    lhs = dist.below_threshold(f * i, theil=m.t[idx], empirical=emp, opt=True)
    return lhs >= b * rhs


def threshold_hi_rule(m, idx, f=0.5, b=1.1):
    dist = ineq.LogNormal(pyomo=True, mean=False)
    emp = m.data['empirical']
    i = m.data['i'][idx]
    rhs = dist.below_threshold(f * i, theil=m.data['t'][idx], empirical=emp)
    print(f, i, m.t[idx], emp, m.data['t'][idx], rhs)
    lhs = dist.below_threshold(f * i, theil=m.t[idx], empirical=emp, opt=True)
    return lhs <= b * rhs

#
# Objectives
#


def min_diff_obj(m):
    return sum((m.t[idx] - m.data['t'][idx]) ** 2
               for idx in m.idxs)


class Model(object):
    """Base class for Inequality Calibration Models"""

    def __init__(self, natdata, subdata, empirical=False):
        self.natdata = natdata
        self.subdata = subdata
        self.empirical = empirical

        self._setup_model_data(natdata, subdata)
        self._check_model_data()

    def _setup_model_data(self, natdata, subdata):
        n, i, gini = 'n', 'i', 'gini'
        ndf, sdf = self.natdata, self.subdata

        # correct income by scaling
        sdf[n] *= ndf[n] / sdf[n].sum()
        sdf[i] *= (ndf[n] * ndf[i]) / (sdf[n] * sdf[i]).sum()

        # calculate national within theil
        T = ineq.gini_to_theil(ndf[gini], empirical=self.empirical)
        gfrac = (sdf[n] * sdf[i]) / (ndf[n] * ndf[i])
        nfrac = sdf[n] / ndf[n]
        T_b = np.sum(gfrac * np.log(gfrac / nfrac))
        T_w = T - T_b

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
            'i': sdf[i].values,
            'g': (sdf[n] * sdf[i]).values,
            't': ineq.gini_to_theil(sdf[gini].values, empirical=self.empirical),
            'N': ndf[n],
            'I': ndf[i],
            'G': ndf[n] * ndf[i],
            'T_w': T_w,
            'empirical': self.empirical,
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
        df.index = self.sorted_idx
        df = df.loc[self.orig_idx]
        return df


class Model1(Model):
    """
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
        m.t = mo.Var(m.idxs)
        # Constraints
        m.position = mo.Constraint(m.idxs, rule=position_rule,
                                   doc='ordering between provinces must be maintained')
        m.diff_hi = mo.Constraint(m.idxs, rule=diff_hi_rule,
                                  doc='difference between values should be about the same')
        m.diff_lo = mo.Constraint(m.idxs, rule=diff_lo_rule,
                                  doc='difference between values should be about the same')
        m.theil_sum = mo.Constraint(rule=theil_sum_rule, doc='factor ordering')
        m.obj = mo.Objective(rule=min_diff_obj, sense=mo.minimize)
        return self


class Model2(Model):
    """
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
        m.t = mo.Var(m.idxs)
        # Constraints
        m.thresh_hi = mo.Constraint(m.idxs, rule=threshold_hi_rule,
                                    doc='')
        m.thresh_lo = mo.Constraint(m.idxs, rule=threshold_lo_rule,
                                    doc='')
        m.theil_sum = mo.Constraint(rule=theil_sum_rule, doc='')
        m.obj = mo.Objective(rule=min_diff_obj, sense=mo.minimize)
        return self
