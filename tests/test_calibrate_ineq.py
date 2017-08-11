import numpy as np
import pandas as pd

from salamanca import ineq
from salamanca.models.calibrate_ineq import Model, Model1

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


def test_Model1_solution():
    natdata, subdata = data()
    model = Model1(natdata, subdata)
    model.construct()
    model.solve()

    # this is the theil result
    obs = model.solution
    exp = np.array([0.062872, 0.369337])  # solution is ordered small to large
    assert_array_almost_equal(obs, exp)
    theil_exp = exp


def test_Model1_result():
    natdata, subdata = data()
    model = Model1(natdata, subdata)
    model.construct()
    model.solve()

    df = model.result()
    assert sorted(df.columns) == ['gini', 'i', 'n']
    obs = df['gini'].values
    exp = ineq.theil_to_gini(model.solution, empirical=False)
    assert_array_almost_equal(obs, exp[::-1])  # back to the right order
