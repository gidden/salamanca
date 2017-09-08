import math

import pyomo.environ as mo


#
# Pyomo-enabled helper functions
#


def below_threshold(x, i, t, mean=True):
    """Compute the CDF of the lognormal distribution at x using an approximation of
    the error function:

    .. math::

        erf(x) \approx \tanh(\sqrt \pi \log x)

    Parameters
    ----------
    x : numeric
       threshold income
    i : numeric
       mean income (per capita)
    t : numeric or Pyomo variable
       theil coefficient
    mean : bool, default: True
       treat income as mean income
    """
    # see
    # https://math.stackexchange.com/questions/321569/approximating-the-error-function-erf-by-analytical-functions
    sigma2 = 2 * t  # t is var
    # f(var), adjust for mean income vs. median
    mu = mo.log(i)
    if mean:
        mu -= sigma2 / 2
    # f(var), argument for error function
    arg = (mo.log(x) - mu) / mo.sqrt(2 * sigma2)
    # coefficient for erf approximation
    k = math.pi ** 0.5 * math.log(2)
    # definition of cdf with tanh(kx) approximating erf(x)
    return 0.5 + 0.5 * mo.tanh(k * arg)


def model_T_w(m, income_from_data=True):
    def _t_w(m, idx):
        i = m.data['i'] if income_from_data else m.i
        return i[idx] * m.data['n_frac'][idx] / m.data['I'] * m.t[idx]
    T_w = sum(_t_w(m, idx) for idx in m.idxs)
    return T_w


def model_T_b(m, from_data=False):
    i = m.data['i'] if from_data else m.i
    I = m.data['I_old'] if from_data else m.data['I']
    nfracs = m.data['n_frac_old'] if from_data else m.data['n_frac']

    def _t_b(m, idx):
        ifrac = i[idx] / I
        nfrac = nfracs[idx]
        return nfrac * ifrac * mo.log(ifrac)
    T_b = sum(_t_b(m, idx) for idx in m.idxs)
    return T_b


def i_std(m, from_data=True):
    i = m.data['i'] / m.data['I_old'] if from_data else m.i
    mu = sum(i[idx] for idx in m.idxs) / len(m.idxs)
    return sum((i[idx] - mu) ** 2 for idx in m.idxs) / len(m.idxs)


def t_std(m, from_data=True):
    t = m.data['t'] if from_data else m.t
    mu = sum(t[idx] for idx in m.idxs) / len(m.idxs)
    return sum((t[idx] - mu) ** 2 for idx in m.idxs) / len(m.idxs)
