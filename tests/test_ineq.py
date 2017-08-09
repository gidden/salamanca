import pytest

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


def test_gini_to_theil_error():
    with pytest.raises(ValueError):
        iq.gini_to_theil(-1)

    with pytest.raises(ValueError):
        iq.gini_to_theil(0.0)

    with pytest.raises(ValueError):
        iq.gini_to_theil(1.0)

    with pytest.raises(ValueError):
        iq.gini_to_theil(1.1)


def test_theil_to_gini_error():
    with pytest.raises(ValueError):
        iq.theil_to_gini(-1)

    with pytest.raises(ValueError):
        iq.theil_to_gini(0.0)

    with pytest.raises(ValueError):
        iq.theil_to_gini(iq.MAX_THEIL)

    with pytest.raises(ValueError):
        iq.theil_to_gini(7.0)


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
