import pytest

import pandas as pd

from pandas.util.testing import assert_frame_equal

from salamanca import data

# decorator for test requiring internet
remote = pytest.mark.skipif(
    not pytest.config.getoption("--remote"),
    reason="need --remote option to run"
)


def test_wb_query_url():
    wb = data.WorldBank()

    obs = wb._query_url('DPANUSIFS')
    exp = 'http://api.worldbank.org/en/countries/all/indicators/DPANUSIFS?format=json'
    assert obs == exp

    obs = wb._query_url('DPANUSIFS', iso='br')
    exp = 'http://api.worldbank.org/en/countries/br/indicators/DPANUSIFS?format=json'
    assert obs == exp

    obs = wb._query_url(
        'DPANUSIFS', iso='ind;chn', MRV=5, frequency='M')
    exp = 'http://api.worldbank.org/en/countries/ind;chn/indicators/DPANUSIFS?format=json&MRV=5&frequency=M'
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
    df = wb.query('DPANUSSPF', iso='ind;chn', MRV=5, frequency='M')
    obs = df[df.date == '2017M06'].set_index('country')
    exp = pd.DataFrame({
        'country': ['CN', 'IN'],
        'date': ['2017M06', '2017M06'],
        'decimal': [0.0, 0.0],
        'indicator': ['DPANUSSPF', 'DPANUSSPF'],
        'value': [6.80702272727, 64.44736363636],
    }).set_index('country')
    assert_frame_equal(obs, exp)
