from __future__ import division

import numpy as np
import pandas as pd
import pyomo.environ as mo


from salamanca import ineq
from salamanca.models.utils import below_threshold

#
# Constraints
#


def theil_sum_rule(m):
    return sum(m.t[idx] * m.data['g'][idx] for idx in m.idxs) == \
        m.data['T_w'] * m.data['G']


def ratio_hi_rule(m, b=1.1):
    return m.t[m.data['idxs'][0]] <= b * m.t[m.data['idxs'][1]]


def ratio_lo_rule(m, b=0.9):
    return m.t[m.data['idxs'][0]] >= b * m.t[m.data['idxs'][1]]


#
# Objectives
#
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
        if any(x not in sdf for x in required[:-1]):
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

        # save model data
        self.orig_idx = sdf.index
        sdf = sdf.reset_index()
        self.model_idx = sdf.index
        self.model_data = {
            'idxs': self.model_idx.values,
            'n': sdf[n].values,
            'n_frac': sdf[n].values / ndf[n],
            'i': sdf[i].values,
            'g': (sdf[n] * sdf[i]).values,
            'N': ndf[n],
            'I': ndf[i],
            'G': ndf[n] * ndf[i],
            'T_w': T_w,
            'T_b': T_b,
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

        obs = list(self.orig_idx)
        exp = ['u', 'r']
        if not obs == exp:
            raise ValueError('Subnational indicies are not [u, r]')

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
        m.theil_sum = mo.Constraint(rule=theil_sum_rule, doc='')
        m.ratio_hi = mo.Constraint(rule=ratio_hi_rule, doc='')
        m.ratio_lo = mo.Constraint(rule=ratio_lo_rule, doc='')
        # Objective
        m.obj = mo.Objective(
            rule=lambda m: national_threshold_obj(m, f=0.25),
            sense=mo.minimize
        )
        return self

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
            i: self.model_data[i],
            n: self.model_data[n],
            gini: ineq.theil_to_gini(self.solution, empirical=self.empirical),
        })
        df.index = self.orig_idx

        df['i_orig'] = self.subdata[i]
        df['n_orig'] = self.subdata[n]
        return df
