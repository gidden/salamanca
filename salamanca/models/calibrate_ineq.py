import pyomo.environ as mo


def position_rule(m, idx):
    r"""Returns |matha|

    .. |matha| replace:: :math:`\sigma = 2 * erf^{-1}(g)`
    """
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx - 1] <= m.t[idx]


def diff_rule(m, idx):
    r"""Returns |mathb|

    .. |mathb| replace:: :math:`\sigma = 2 * erf^{-1}(g)`
    """
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx] - m.t[idx - 1] >= 0.8 * (m.data['t'][idx] - m.data['t'][idx - 1])


def diff_rule2(m, idx):
    if idx == 0:
        return mo.Constraint.Skip
    return m.t[idx] - m.t[idx - 1] <= 1.2 * (m.data['t'][idx] - m.data['t'][idx - 1])


def theil_sum_rule(m):
    return sum(m.t[idx] * m.data['g'][idx] for idx in m.idxs) == \
        m.data['T_w'] * m.data['G']


def obj_rule(m):
    return sum((m.t[idx] - m.data['t'][idx]) ** 2
               for idx in m.idxs)


class Model(object):
    """
    Comprised of

    | |matha|
    | |mathb|

    | Subject to
    | |mathb|
    """

    def __init__(self):
        pass
