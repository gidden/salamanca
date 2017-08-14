import math

import numpy as np

from scipy.stats import norm, lognorm
from scipy.special import erf, erfinv

from salamanca.utils import AttrObject

# empirical limits with gini = 0.99
MAX_THEIL = 6.64


def gini_to_std(g):
    r"""Returns :math:`\sigma = 2 * erf^{-1}(g)`"""
    return 2.0 * erfinv(g)


def std_to_gini(s):
    r"""Returns :math:`g = erf( \frac{\sigma}{2} )`"""
    return erf(0.5 * s)


def theil_to_std(t):
    r"""Returns :math:`\sigma = \sqrt{2t}`"""
    return (2 * t) ** 0.5


def std_to_theil(s):
    r"""Returns :math:`t = \frac{\sigma^2}{2}`"""
    return s ** 2 / 2.0


def _theil_empirical_constants():
    """
    These quadractic constants were established by Narasimha to translate between
    Thiel values calculated from deciles to those calculated from Ginis
    """
    return 0.216, 0.991, 0.003


def gini_to_theil(g, empirical=False):
    r"""Translate gini to theil

    .. math::

       t(g) = \sqrt{2} \Phi^{-1} \left( \frac{1 + g}{2} \right)

    Where :math:`\Phi` is cumulative distribution function (CDF) of the standard
    normal distribution.

    Parameters
    ----------
    g : numeric or array-like
        gini coefficient(s)
    empirical : bool, optional, default: False
        whether to use empirical relationship for theil

    """
    if not (np.all(g > 0) and np.all(g < 1)):
        raise ValueError('Gini not within (0, 1)')

    s = gini_to_std(g)
    t = std_to_theil(s)
    if empirical:
        # quadratic method root finding
        a, b, c = _theil_empirical_constants()
        t = (-b + (b ** 2 - 4 * a * (c - t)) ** 0.5) / (2 * a)

    if not (np.all(t < MAX_THEIL) and np.all(t > 0)):
        raise ValueError('Theil not within (0, 2.88): {}'.format(t))
    return t


def theil_to_gini(t, empirical=False):
    r"""Translate theil to gini

    .. math::

       t(g) = \sqrt{2} \Phi^{-1} \left( \frac{1 + g}{2} \right)

    Where :math:`\Phi` is cumulative distribution function (CDF) of the standard
    normal distribution.

    Parameters
    ----------
    t : numeric or array-like
        theil coefficient(s)
    empirical : bool, optional, default: False
        whether to use empirical relationship for theil
    """
    if not (np.all(t < MAX_THEIL) and np.all(t > 0)):
        raise ValueError('Theil not within (0, 2.88): {}'.format(t))

    if empirical:
        a, b, c = _theil_empirical_constants()
        t = a * t ** 2 + b * t + c
    s = theil_to_std(t)
    g = std_to_gini(s)

    if not (np.all(g > 0) and np.all(g < 1)):
        raise ValueError('Gini not within (0, 1)')
    return g


class LogNormalData(AttrObject):
    """Object for storing and updating LogNormal distribution data"""

    def _has(self, x):
        return getattr(self, x, None) is None

    def add_defaults(self, copy=True):
        # add new defaults for kwargs here
        defaults = {
            'inc': 1,
            'mean': True,
            'gini': None,
            'theil': None,
        }
        return self.update(copy=copy, override=False, **defaults)

    def check(self):
        x = ('theil', 'gini')
        bad = all(self._has(_) for _ in x)
        if bad:
            raise ValueError('Must use either theil or gini')
        bad = all(not self._has(_) for _ in x)
        if bad:
            raise ValueError('Cannot use both theil and gini')

        x = ('inc', 'mean')
        for _ in x:
            if self._has(_):
                raise ValueError('Must declare value for ' + _)
        return self


class LogNormal(object):
    """
    .. _lognorm-docs:

    An interfrace to the log-normal distribution.

    For mathematical descriptions see ScipyLogNorm_.

    Parameters
    ----------
    inc  : numeric, optional, default: 1
    mean : bool, optional, default: True
        whether income is representative of mean (True) or median (False)
    gini : numeric, optional
    theil : numeric, optional
    """

    def __init__(self, **kwargs):
        """See lognorm-docs_ for possible arguments"""
        self.init_data = LogNormalData(**kwargs)

    def params(self, **kwargs):
        """Returns (shape, scale) tuple for use in scipy.stats.lognorm"""
        data = (
            self.init_data
            .update(**kwargs)
            .add_defaults(copy=True)
            .check()
        )

        shape = gini_to_std(data.gini) if data.theil is None \
            else theil_to_std(data.theil)
        # scale assumes a median value, adjust is made if income is mean income
        scale = np.exp(np.log(data.inc) - shape ** 2 / 2) if data.mean \
            else data.inc
        return shape, scale

    def pdf(self, x, **kwargs):
        """See ScipyLogNorm_"""
        shape, scale = self.params(**kwargs)
        return lognorm.pdf(x, shape, scale=scale)

    def ppf(self, x, **kwargs):
        """See ScipyLogNorm_"""
        shape, scale = self.params(**kwargs)
        return lognorm.ppf(x, shape, scale=scale)

    def cdf(self, x, **kwargs):
        """See ScipyLogNorm_"""
        shape, scale = self.params(**kwargs)
        return lognorm.cdf(x, shape, scale=scale)

    def below_threshold(self, threshold, **kwargs):
        """Returns the fraction of population below a given income threshold (equivalent
        to `cdf`).

        Parameters
        ----------
        threshold : numeric
            income threshold
        """
        return self.cdf(threshold, **kwargs)

    def lorenz(self, x, **kwargs):
        r"""The Lorenz curve for log-normal distributions is defined as:

        .. math::

           L(x) = \Phi \left( \Phi^{-1} (x) - \sigma \right)

        Where :math:`\Phi` is cumulative distribution function (CDF) of the standard
        normal distribution.
        """
        shape, scale = self.params(**kwargs)
        return norm.cdf(norm.ppf(x) - shape)

    def mean(self, **kwargs):
        """See ScipyLogNorm_"""
        shape, scale = self.params(**kwargs)
        return lognorm.mean(shape, scale=scale)

    def median(self, **kwargs):
        """See ScipyLogNorm_"""
        shape, scale = self.params(**kwargs)
        return lognorm.median(shape, scale=scale)

    def var(self, **kwargs):
        """See ScipyLogNorm_"""
        shape, scale = self.params(**kwargs)
        return lognorm.var(shape, scale=scale)

    def std(self, **kwargs):
        """See ScipyLogNorm_"""
        shape, scale = self.params(**kwargs)
        return lognorm.std(shape, scale=scale)
