import pytest
import os

import pandas as pd

from pandas.util.testing import assert_frame_equal

from salamanca import data

# decorator for test requiring internet
remote = pytest.mark.skipif(
    not pytest.config.getoption("--remote"),
    reason="need --remote option to run"
)

# decorator for slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run"
)


def assert_almost_equal(x, y, eps=1e-6):
    assert abs(x - y) < eps


def logging_on():
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def test_wb_query_url():
    wb = data.WorldBank()

    obs = wb._query_url('DPANUSIFS')
    exp = 'http://api.worldbank.org/en/countries/all/indicators/DPANUSIFS?format=json&per_page=1000'
    assert obs == exp

    obs = wb._query_url('DPANUSIFS', iso='br')
    exp = 'http://api.worldbank.org/en/countries/br/indicators/DPANUSIFS?format=json&per_page=1000'
    assert obs == exp

    obs = wb._query_url(
        'DPANUSIFS', iso='ind;chn', MRV=5, frequency='M')
    exp = 'http://api.worldbank.org/en/countries/ind;chn/indicators/DPANUSIFS?format=json&per_page=1000&MRV=5&frequency=M'
    assert obs == exp


@remote
def test_wb_db_query():
    wb = data.WorldBank()
    q = wb._query_url('DPANUSSPF', iso='ind;chn', MRV=5, frequency='M')
    result = wb._do_query(q)
    for d in result:
        if d['date'] == '2017M06':
            if d['country']['value'] == 'China':
                assert d['value'] == '6.80702272727'
            if d['country']['value'] == 'India':
                assert d['value'] == '64.44736363636'


@remote
def test_wb_query():
    wb = data.WorldBank()
    df = wb.query('DPANUSSPF', iso='ind;chn', MRV=5,
                  frequency='M', use_cache=False)
    obs = df[df.date == '2017M06'].set_index('country')
    exp = pd.DataFrame({
        'country': ['CHN', 'IND'],
        'date': ['2017M06', '2017M06'],
        'value': [6.80702272727, 64.44736363636],
    }).set_index('country')
    assert_frame_equal(obs, exp)


@remote
@slow
def test_wb_query_cache():
    db = data.backend()
    source = 'wb'
    ind = 'DPANUSSPF'

    if db.exists(source, 'DPANUSSPF'):
        os.remove(db.full_path(source, 'DPANUSSPF'))
    assert not db.exists(source, 'DPANUSSPF')

    wb = data.WorldBank()
    df = wb.query(ind)
    assert db.exists(source, ind)

    obs = float(df[(df.country == 'AFG') & (df.date == 1992)]['value'])
    exp = 43.507299
    assert_almost_equal(obs, exp)


def test_csv_backend_fname():
    db = data.CSVBackend()
    obs = db.fname('foo', 'bar')
    exp = 'foo_bar.csv'
    assert obs == exp


@remote
@slow
def test_wb_exchange_rate():
    wb = data.WorldBank()
    df = wb.exchange_rate()
    obs = float(df[(df.country == 'AUT') & (df.date == 2015)]['value'])
    exp = 0.902
    assert_almost_equal(obs, exp, eps=1e-3)
