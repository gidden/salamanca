import pytest

import numpy as np
import pandas as pd

from salamanca import ineq
from salamanca.models.calibrate_ineq import Model, Model1, Model2, Model3, Model4

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
        'gini': [0.5, 0.3],
    }, index=['foo', 'bar'])
    return natdata, subdata


def test_model_data_pop():
    # note all subdata order is swapped in model_data due to sorting by gini
    natdata, subdata = data()
    model = Model(natdata, subdata)

    # pop
    obs = model.model_data['N']
    exp = natdata['n']
    assert_almost_equal(obs, exp)
    obs = model.model_data['n']
    exp = subdata['n'] * natdata['n'] / subdata['n'].sum()
    assert_array_almost_equal(obs, exp[::-1])


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
    assert_array_almost_equal(obs, exp[::-1])


def test_model_data_ineq():
    # note all subdata order is swapped in model_data due to sorting by gini
    natdata, subdata = data()
    model = Model(natdata, subdata)

    # ineq
    obs = model.model_data['t']
    exp = ineq.gini_to_theil(subdata['gini'].values, empirical=False)
    assert_array_almost_equal(obs, exp[::-1])


def test_model_data_idx():
    # note all subdata order is swapped in model_data due to sorting by gini
    natdata, subdata = data()
    model = Model(natdata, subdata)

    # indicies
    obs = model.orig_idx.values
    exp = np.array(['foo', 'bar'])
    assert (obs == exp).all()
    obs = model.sorted_idx.values
    exp = np.array(['bar', 'foo'])
    assert (obs == exp).all()
    obs = model.model_idx.values
    exp = np.array([0, 1])
    assert (obs == exp).all()


def test_model_data_error():
    natdata, subdata = data()
    ndf = natdata.copy().drop('n')
    with pytest.raises(ValueError):
        model = Model(ndf, subdata)

    sdf = natdata.copy().drop('n')
    with pytest.raises(ValueError):
        model = Model(natdata, sdf)


def test_Model1_full():
    natdata, subdata = data()
    model = Model1(natdata, subdata)
    model.construct()
    model.solve()

    # this is the theil result
    # equivalent ginis are: 0.19798731, 0.45663392
    obs = model.solution
    # solution is ordered small to large
    exp = np.array([0.062872, 0.369337])
    assert_array_almost_equal(obs, exp)
    theil_exp = exp

    df = model.result()
    obs = sorted(df.columns)
    exp = ['gini', 'gini_orig', 'i', 'i_orig', 'n', 'n_orig']
    assert obs == exp

    obs = df['gini_orig']
    exp = subdata['gini']
    assert_array_almost_equal(obs, exp)

    obs = df['i_orig']
    exp = subdata['i']
    assert_array_almost_equal(obs, exp)

    obs = df['n_orig']
    exp = subdata['n']
    assert_array_almost_equal(obs, exp)


def test_Model1_result():
    natdata, subdata = data()
    model = Model1(natdata, subdata)
    model.construct()
    model.solve()

    # ginis in original order
    obs = model.result()['gini'].values
    exp = [0.45663392, 0.19798731]
    assert_array_almost_equal(obs, exp)


def test_Model2_result():
    natdata, subdata = data()
    model = Model2(natdata, subdata)
    model.construct()
    model.solve()

    # ginis in original order
    obs = model.result()['gini'].values
    exp = [0.45663392, 0.19798731]
    assert_array_almost_equal(obs, exp)


def test_Model3_result():
    natdata, subdata = data()
    model = Model3(natdata, subdata)
    model.construct()
    model.solve()

    # ginis in original order
    obs = model.result()['gini'].values
    exp = [0.45209117, 0.21030433]
    assert_array_almost_equal(obs, exp)


def test_Model4_result():
    natdata, subdata = data()
    model = Model4(natdata, subdata)
    model.construct()
    model.solve()

    # ginis in original order
    obs = model.result()['gini'].values
    exp = [0.41473959, 0.28608873]
    assert_array_almost_equal(obs, exp)
