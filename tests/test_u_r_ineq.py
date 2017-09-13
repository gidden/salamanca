import pytest

import numpy as np
import pandas as pd

from salamanca import ineq
from salamanca.models.u_r_ineq import Model

from utils import assert_almost_equal, assert_array_almost_equal


def data():
    natdata = pd.Series({
        'n': 20,
        'i': 105,
        'gini': 0.4,
    })
    subdata = pd.DataFrame({
        'n': [5, 10],
        'i': [10, 5],
    }, index=['u', 'r'])
    return natdata, subdata


def test_model_data_pop():
    natdata, subdata = data()
    model = Model(natdata, subdata)

    # pop
    obs = model.model_data['N']
    exp = natdata['n']
    assert_almost_equal(obs, exp)
    obs = model.model_data['n']
    exp = subdata['n'] * natdata['n'] / subdata['n'].sum()
    assert_array_almost_equal(obs, exp)


def test_model_data_inc():
    # note all subdata order is swapped in model_data due to sorting by gini
    natdata, subdata = data()
    model = Model(natdata, subdata)

    # inc
    obs = model.model_data['G']
    exp = natdata['n'] * natdata['i']
    assert_almost_equal(obs, exp)
    obs = model.model_data['g']
    expn = subdata['n'] * natdata['n'] / subdata['n'].sum()
    exp = (subdata['i'] * expn) * (natdata['i'] * natdata['n']) \
        / (subdata['i'] * expn).sum()
    assert_array_almost_equal(obs, exp)


def test_Model_full():
    natdata, subdata = data()
    model = Model(natdata, subdata, empirical=True)
    model.construct()
    model.solve()

    obs = model.solution
    exp = np.array([0.190298, 0.211437])
    assert_array_almost_equal(obs, exp)
    theil_exp = exp

    df = model.result()
    obs = sorted(df.columns)
    exp = ['gini', 'i', 'i_orig', 'n', 'n_orig']
    assert obs == exp

    obs = df['i_orig']
    exp = subdata['i']
    assert_array_almost_equal(obs, exp)

    obs = df['n_orig']
    exp = subdata['n']
    assert_array_almost_equal(obs, exp)


def test_Model_result():
    natdata, subdata = data()
    model = Model(natdata, subdata, empirical=True)
    model.construct()
    model.solve()

    # ginis in original order
    obs = model.result()['gini'].values
    exp = [0.34480044, 0.36262428]
    assert_array_almost_equal(obs, exp)


if __name__ == '__main__':
    n, s = data()
    print(n)
    print(s)
    test_model_data_pop()
    test_model_data_inc()
