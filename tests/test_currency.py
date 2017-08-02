import pytest

from salamanca import cli
from salamanca.currency import Translator


def assert_almost_equal(x, y, eps=1e-6):
    assert abs(x - y) < eps


US_AUT_2010 = 0.75504495198999999
CAN_US_2010 = 1.03016278295
MER_PPP_AUT_2010 = 1.1133549039499999  # changed recently in world bank dataset


def test_2010_aut_mer():
    xltr = Translator()
    obs = xltr.exchange(20, iso='AUT', yr=2010)
    exp = 20 / US_AUT_2010
    assert_almost_equal(obs, exp)


def test_2010_aut_ppp():
    xltr = Translator()
    obs = xltr.exchange(20, iso='AUT', yr=2010, units='PPP')
    exp = 20 / US_AUT_2010 / MER_PPP_AUT_2010
    assert_almost_equal(obs, exp)


def test_2010_aut_usa_mer():
    xltr = Translator()
    obs = xltr.exchange(20, fromiso='AUT', toiso='USA', yr=2010)
    exp = 20 / US_AUT_2010
    assert_almost_equal(obs, exp)


def test_2010_usa_aut_mer():
    xltr = Translator()
    obs = xltr.exchange(20, fromiso='USA', toiso='AUT', yr=2010)
    exp = 20 * US_AUT_2010
    assert_almost_equal(obs, exp)


def test_2010_can_aut_mer():
    xltr = Translator()
    obs = xltr.exchange(20, fromiso='CAN', toiso='AUT', yr=2010)
    exp = 20 / CAN_US_2010 * US_AUT_2010
    assert_almost_equal(obs, exp)


cpi_aut_2005 = 91.34538211520001
gdef_aut_2005 = 91.573719723099998


def test_same_country_2010_2005_cpi():
    xltr = Translator()
    obs = xltr.exchange(20, iso='AUT', fromyr=2010, toyr=2005,
                        inflation_method='cpi')
    exp = 20 * cpi_aut_2005 / 100.
    assert_almost_equal(obs, exp)


def test_same_country_2010_2005_def():
    xltr = Translator()
    obs = xltr.exchange(20, iso='AUT', fromyr=2010, toyr=2005)
    exp = 20 * gdef_aut_2005 / 100.
    assert_almost_equal(obs, exp)


def test_same_country_2005_2010_def():
    xltr = Translator()
    obs = xltr.exchange(20, iso='AUT', fromyr=2005, toyr=2010)
    exp = 20 / (gdef_aut_2005 / 100.)
    assert_almost_equal(obs, exp)


gdef_usa_2005 = 90.877573015799996
gdef_aut_2005 = 91.573719723099998
xr_aut_2010 = 0.75504495198999999


def test_exchange_aut_usa():
    xltr = Translator()
    obs = xltr.exchange(20, fromiso='AUT', toiso='USA', fromyr=2010, toyr=2005)
    exp = 20 / xr_aut_2010 / (100. / gdef_usa_2005)
    assert_almost_equal(obs, exp)


def test_exchange_usa_aut():
    xltr = Translator()
    obs = xltr.exchange(20, fromiso='USA', toiso='AUT', fromyr=2010, toyr=2005)
    exp = 20 * xr_aut_2010 / (100. / gdef_aut_2005)
    assert_almost_equal(obs, exp)


gdef_can_2005 = 90.226979220900006  # changed recently in world bank dataset
xr_aut_2005 = 0.80411999999999995
xr_can_2005 = 1.21176333333


def test_aut_2005_can_2010_mer():
    xltr = Translator()
    obs = xltr.exchange(20, fromiso='AUT', toiso='CAN', fromyr=2005, toyr=2010)
    exp = 20 / (xr_aut_2005 / xr_can_2005) / (gdef_can_2005 / 100.)
    assert_almost_equal(obs, exp)


gdef_aut_2005 = 91.573719723099998


def test_can_2005_aut_2010_mer():
    xltr = Translator()
    obs = xltr.exchange(20, fromiso='CAN', toiso='AUT', fromyr=2005, toyr=2010)
    exp = 20 * (xr_aut_2005 / xr_can_2005) / (gdef_aut_2005 / 100.)
    assert_almost_equal(obs, exp)


ppp_to_mer_aut_2005 = 1.0968026364899999  # changed recently in wb stats
ppp_to_mer_can_2005 = 1.0015217032500001


def test_can_2005_aut_2010_ppp():
    xltr = Translator()
    obs = xltr.exchange(20, fromiso='CAN', toiso='AUT', fromyr=2005, toyr=2010,
                        units='PPP')
    exp = 20 * (xr_aut_2005 / xr_can_2005) / (gdef_aut_2005 / 100.)
    exp *= ppp_to_mer_aut_2005 / ppp_to_mer_can_2005
    assert_almost_equal(obs, exp)


ppp_to_mer_aut_2010 = 1.11335490395  # changed recently in wb stats


def test_aut_2005_2010_usd_mer():
    xltr = Translator()
    obs = xltr.exchange(20, iso='AUT', fromyr=2005,
                        toyr=2010, inusd=True)
    exp = (20 * xr_aut_2005) * (100. / gdef_aut_2005) / xr_aut_2010
    assert_almost_equal(obs, exp)


def test_aut_2005_2010_usd_ppp():
    xltr = Translator()
    obs = xltr.exchange(20, iso='AUT', fromyr=2005,
                        toyr=2010, inusd=True, units='PPP')
    exp = 20 * (xr_aut_2005 * ppp_to_mer_aut_2005) * \
        (100. / gdef_aut_2005) / \
        (xr_aut_2010 * ppp_to_mer_aut_2010)
    assert_almost_equal(obs, exp)


def test_cli():
    obs = cli.exchange(command='exchange', amt=20, iso='AUT', fromyr=2005,
                       toyr=2010, inusd=True, units='PPP')
    exp = 20 * (xr_aut_2005 * ppp_to_mer_aut_2005) * \
        (100. / gdef_aut_2005) / \
        (xr_aut_2010 * ppp_to_mer_aut_2010)
    assert_almost_equal(obs, exp)


def test_iso_use_error():
    xltr = Translator()
    with pytest.raises(ValueError):
        xltr.exchange(20, fromiso='AUT', toiso='USA',
                      fromyr=2005, toyr=2010, inusd=True)
