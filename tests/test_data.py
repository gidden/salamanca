import pytest
from pytest import approx
import os

import pandas as pd

from pandas.util.testing import assert_frame_equal

from salamanca import data


@pytest.mark.remote
def test_wb_db_query():
    wb = data.WorldBank()
    result = wb._do_query(
        'DPANUSSPF',
        dict(iso='ind;chn', date="2017M05:2017M07", frequency='M')
    )
    for d in result:
        if d['date'] == '2017M06':
            if d['country']['value'] == 'China':
                assert d['value'] == 6.80702272727
            if d['country']['value'] == 'India':
                assert d['value'] == 64.44736363636


@pytest.mark.remote
def test_wb_query():
    wb = data.WorldBank()
    df = wb.query('DPANUSSPF', iso='ind;chn', date="2017M05:2017M07",
                  frequency='M', use_cache=False)
    obs = df[df.date == '2017M06'].set_index('country')
    exp = pd.DataFrame({
        'country': ['CHN', 'IND'],
        'date': ['2017M06', '2017M06'],
        'value': [6.80702272727, 64.44736363636],
    }).set_index('country')
    assert_frame_equal(obs, exp)


@pytest.mark.remote
@pytest.mark.slow
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
    assert obs == approx(exp)


@pytest.mark.remote
@pytest.mark.slow
def test_wb_exchange_rate():
    wb = data.WorldBank()
    df = wb.exchange_rate()
    obs = float(df[(df.country == 'AUT') & (df.date == 2015)]['value'])
    exp = 0.902
    assert obs == approx(exp, rel=1e-3)
