import pytest

from salamanca.currency import Translator


def assert_almost_equal(x, y, eps=1e-6):
    assert abs(x - y) < eps


def test_currency_same_year():
    xltr = Translator()

    obs = xltr.exchange(20, iso='AUT', yr=2010)
    exp = 20 / 0.75504495198999999
    assert_almost_equal(obs, exp)

    obs = xltr.exchange(20, iso='AUT', yr=2010, units='PPP')
    exp = 20 / 0.75504495198999999 / 1.1137920529799998
    assert_almost_equal(obs, exp)

    obs = xltr.exchange(20, fromiso='AUT', toiso='USA', yr=2010)
    exp = 20 / 0.75504495198999999
    assert_almost_equal(obs, exp)

    obs = xltr.exchange(20, fromiso='USA', toiso='AUT', yr=2010)
    exp = 20 * 0.75504495198999999
    assert_almost_equal(obs, exp)

    obs = xltr.exchange(20, fromiso='CAN', toiso='AUT', yr=2010)
    exp = 20 / 1.03016278295 * 0.75504495198999999
    assert_almost_equal(obs, exp)


def test_currency_same_country():
    xltr = Translator()
    cpi_aut_2005 = 91.34538211520001
    gdef_aut_2005 = 91.573719723099998

    obs = xltr.exchange(20, iso='AUT', fromyr=2010, toyr=2005,
                        inflation_method='cpi')
    exp = 20 * cpi_aut_2005 / 100.
    assert_almost_equal(obs, exp)

    obs = xltr.exchange(20, iso='AUT', fromyr=2010, toyr=2005)
    exp = 20 * gdef_aut_2005 / 100.
    assert_almost_equal(obs, exp)

    obs = xltr.exchange(20, iso='AUT', fromyr=2005, toyr=2010)
    exp = 20 / (gdef_aut_2005 / 100.)
    assert_almost_equal(obs, exp)


def test_currency_us_exchange():
    xltr = Translator()
    gdef_usa_2005 = 90.877573015799996
    gdef_aut_2005 = 91.573719723099998
    xr_aut_2010 = 0.75504495198999999

    obs = xltr.exchange(20, fromiso='AUT', toiso='USA', fromyr=2010, toyr=2005)
    exp = 20 / xr_aut_2010 / (100. / gdef_usa_2005)
    assert_almost_equal(obs, exp)

    obs = xltr.exchange(20, fromiso='USA', toiso='AUT', fromyr=2010, toyr=2005)
    exp = 20 * xr_aut_2010 / (100. / gdef_aut_2005)
    assert_almost_equal(obs, exp)


def test_currency_bilateral_exchange():
    xltr = Translator()
    gdef_aut_2005 = 91.573719723099998
    gdef_can_2005 = 90.371842335099998
    xr_aut_2005 = 0.80411999999999995
    xr_can_2005 = 1.21176333333

    obs = xltr.exchange(20, fromiso='AUT', toiso='CAN', fromyr=2005, toyr=2010)
    exp = 20 / (xr_aut_2005 / xr_can_2005) / (gdef_can_2005 / 100.)
    assert_almost_equal(obs, exp)

    obs = xltr.exchange(20, fromiso='CAN', toiso='AUT', fromyr=2005, toyr=2010)
    exp = 20 * (xr_aut_2005 / xr_can_2005) / (gdef_aut_2005 / 100.)
    assert_almost_equal(obs, exp)

    ppp_to_mer_can_2005 = 1.0015217032500001
    ppp_to_mer_aut_2005 = 1.1023554284299999

    obs = xltr.exchange(20, fromiso='CAN', toiso='AUT', fromyr=2005, toyr=2010,
                        units='PPP')
    exp = 20 * (xr_aut_2005 * ppp_to_mer_aut_2005) / \
        (xr_can_2005 * ppp_to_mer_can_2005) / \
        (gdef_aut_2005 / 100.)
    assert_almost_equal(obs, exp)


def test_currency_inusd():
    xltr = Translator()
    gdef_aut_2005 = 91.573719723099998
    xr_aut_2005 = 0.80411999999999995
    xr_aut_2010 = 0.75504495198999999
    ppp_to_mer_aut_2005 = 1.1023554284299999
    ppp_to_mer_aut_2010 = 1.1023554284299999

    obs = xltr.exchange(20, iso='AUT', fromyr=2005,
                        toyr=2010, inusd=True)
    exp = (20 * xr_aut_2005) * (100. / gdef_aut_2005) / xr_aut_2010
    assert_almost_equal(obs, exp)

    ppp_to_mer_aut_2005 = 1.1023554284299999
    ppp_to_mer_aut_2010 = 1.1137920529799998

    obs = xltr.exchange(20, iso='AUT', fromyr=2005,
                        toyr=2010, inusd=True, units='PPP')
    exp = 20 * (xr_aut_2005 * ppp_to_mer_aut_2005) * \
        (100. / gdef_aut_2005) / \
        (xr_aut_2010 * ppp_to_mer_aut_2010)
    assert_almost_equal(obs, exp)

    # error, function, args
    with pytest.raises(ValueError):
        xltr.exchange(20, fromiso='AUT', toiso='USA',
                      fromyr=2005, toyr=2010, inusd=True)
