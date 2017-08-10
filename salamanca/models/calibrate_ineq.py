import pyomo.environ as mo


def position_rule(m, idx):
    r"""|pos|

    .. |pos| replace:: :math:`t_{r-1} \leq t_{r} \forall r \in {r_2 \ldots r_N}`
    """
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx - 1] <= m.t[idx]


def diff_hi_rule(m, idx):
    r"""|diff_hi|

    .. |diff_hi| replace:: :math:`t_r - t_{r-1} \geq 0.8 (t^*_r - t^*_{r-1}) \forall r \in {r_2 \ldots r_N}`
    """
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx] - m.t[idx - 1] >= 0.8 * (m.data['t'][idx] - m.data['t'][idx - 1])


def diff_lo_rule(m, idx):
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx] - m.t[idx - 1] <= 1.2 * (m.data['t'][idx] - m.data['t'][idx - 1])


def theil_sum_rule(m):
    return sum(m.t[idx] * m.data['g'][idx] for idx in m.idxs) == \
        m.data['T_w'] * m.data['G']


def min_diff_obj(m):
    return sum((m.t[idx] - m.data['t'][idx]) ** 2
               for idx in m.idxs)


class Model(object):
    """Base class for Inequality Calibration Models"""

    def __init__(self, data):
        self.data = data

    def construct(self):
        raise NotImplementedError()

    def solve(self):
        m = self.model
        solver = mo.SolverFactory('ipopt')
        result = solver.solve(m)  # , tee=True)
        result.write()
        m.solutions.load_from(result)
        t = pd.Series(m.t.get_values().values(),
                      index=m.data['orig_idx'], name='thiel')
        return t


class Model1(Model):
    """
    Comprised of

    | |pos|
    | |diff_hi|

    """

    def construct(self):
        self.model = m = mo.ConcreteModel()
        # Model Data
        m.data = data
        # Sets
        m.idxs = mo.Set(initialize=m.data['idxs'])
        # Variables
        m.t = mo.Var(m.idxs)
        # Constraints
        m.position = mo.Constraint(m.idxs, rule=position_rule,
                                   doc='ordering between provinces must be maintained')
        m.diff = mo.Constraint(m.idxs, rule=diff_rule,
                               doc='difference between values should be about the same')
        m.diff2 = mo.Constraint(m.idxs, rule=diff_rule2,
                                doc='difference between values should be about the same')
        m.theil_sum = mo.Constraint(rule=theil_sum_rule, doc='factor ordering')
        m.obj = mo.Objective(rule=obj_rule, sense=mo.minimize)
        return self
