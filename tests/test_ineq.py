import pytest

import numpy as np

from salamanca import ineq as iq

from utils import assert_almost_equal


def test_gini_to_std():
    obs = iq.gini_to_std(0.5)
    exp = 0.95387
    assert_almost_equal(obs, exp, eps=1e-5)


def test_std_to_gini():
    obs = iq.std_to_gini(0.95387)
    exp = 0.5
    assert_almost_equal(obs, exp, eps=1e-5)


def test_theil_to_std():
    obs = iq.theil_to_std(0.75)
    exp = 1.22474
    assert_almost_equal(obs, exp, eps=1e-5)


def test_std_to_theil():
    obs = iq.std_to_theil(1.22474)
    exp = 0.75
    assert_almost_equal(obs, exp, eps=1e-5)


def test_gini_to_theil():
    obs = iq.gini_to_theil(0.5)
    exp = 0.45493
    assert_almost_equal(obs, exp, eps=1e-5)


def test_theil_to_gini():
    obs = iq.theil_to_gini(0.45493)
    exp = 0.5
    assert_almost_equal(obs, exp, eps=1e-5)


def test_gini_to_theil_empirical():
    obs = iq.gini_to_theil(0.5, empirical=True)
    exp = 0.41796
    assert_almost_equal(obs, exp, eps=1e-5)


def test_theil_to_gini_empirical():
    obs = iq.theil_to_gini(0.41796, empirical=True)
    exp = 0.5
    assert_almost_equal(obs, exp, eps=1e-5)


def test_gini_to_theil_close_lo():
    # just shouldn't through
    iq.gini_to_theil(0.01)
    iq.gini_to_theil(0.05, empirical=True)  # emp lowers theil


def test_gini_to_theil_close_hi():
    # just shouldn't through
    iq.gini_to_theil(0.99)
    iq.gini_to_theil(0.99, empirical=True)


def test_gini_to_theil_nan():
    obs = iq.gini_to_theil(np.nan, ignorenan=True)
    assert np.isnan(obs)


def test_theil_to_gini_nan():
    obs = iq.theil_to_gini(np.nan, ignorenan=True)
    assert np.isnan(obs)


def test_gini_to_theil_error():
    with pytest.raises(ValueError):
        iq.gini_to_theil(-1)

    with pytest.raises(ValueError):
        iq.gini_to_theil(0.0)

    with pytest.raises(ValueError):
        iq.gini_to_theil(1.0)

    with pytest.raises(ValueError):
        iq.gini_to_theil(1.1)

    with pytest.raises(ValueError):
        iq.gini_to_theil(np.nan)


def test_theil_to_gini_error():
    with pytest.raises(ValueError):
        iq.theil_to_gini(-1)

    with pytest.raises(ValueError):
        iq.theil_to_gini(0.0)

    with pytest.raises(ValueError):
        iq.theil_to_gini(iq.MAX_THEIL)

    with pytest.raises(ValueError):
        iq.theil_to_gini(7.0)

    with pytest.raises(ValueError):
        iq.theil_to_gini(np.nan)


def test_lndata_init():
    data = iq.LogNormalData(inc=4.2, gini=0.5)
    assert data.inc == 4.2
    assert data.gini == 0.5
    assert not hasattr(data, 'mean')


def test_lndata_defaults():
    data = iq.LogNormalData(inc=4.2)
    data.add_defaults(copy=False)
    assert data.inc == 4.2
    assert data.mean is True


def test_lndata_defaults_copy():
    data = iq.LogNormalData(inc=4.2)
    newdata = data.add_defaults(copy=True)
    assert data.inc == 4.2
    assert not hasattr(data, 'mean')
    assert newdata.inc == 4.2
    assert newdata.mean is True


def test_lndata_raises_none():
    data = iq.LogNormalData(inc=4.2)
    data.add_defaults()
    with pytest.raises(ValueError):
        data.check()


def test_lndata_raises_both():
    data = iq.LogNormalData(inc=4.2, theil=1.0, gini=0.5)
    data.add_defaults()
    with pytest.raises(ValueError):
        data.check()


def test_lndata_raises_missing():
    data = iq.LogNormalData(gini=0.5, inc=1)
    with pytest.raises(ValueError):
        data.check()

    data = iq.LogNormalData(gini=0.5, mean=False)
    with pytest.raises(ValueError):
        data.check()


def test_ln_params_mean():
    dist = iq.LogNormal(inc=1000, gini=0.5, mean=True)
    obs_shape, obs_scale = dist.params()
    exp_shape = 0.953872
    exp_scale = 634.48830545
    assert_almost_equal(obs_shape, exp_shape)
    assert_almost_equal(obs_scale, exp_scale)


def test_ln_params_median():
    dist = iq.LogNormal(inc=1000, gini=0.5, mean=True)
    obs_shape, obs_scale = dist.params(mean=False)
    exp_shape = 0.953872
    exp_scale = 1000
    assert_almost_equal(obs_shape, exp_shape)
    assert_almost_equal(obs_scale, exp_scale)


def test_ln_lorenz_gini():
    dist = iq.LogNormal()
    obs = dist.lorenz(0.4, gini=0.5)
    exp = 0.11367378
    assert_almost_equal(obs, exp)


def test_ln_lorenz_theil():
    dist = iq.LogNormal()
    obs = dist.lorenz(0.4, theil=0.5)
    exp = 0.1050397
    assert_almost_equal(obs, exp)


def test_ln_threshold():
    # equivalent to cdf function
    dist = iq.LogNormal()
    exp = dist.cdf(0.4, gini=0.5)
    obs = dist.below_threshold(0.4, gini=0.5)
    assert_almost_equal(exp, obs)


def test_ln_scipy_funcs():
    # these just need to run without error, they pass directly to scipy
    dist = iq.LogNormal()
    dist.ppf(0.4, gini=0.5)
    dist.cdf(0.4, gini=0.5)
    dist.mean(gini=0.5)
    dist.median(gini=0.5)
    dist.var(gini=0.5)
    dist.std(gini=0.5)
