import pytest

import numpy as np
import pandas as pd

from salamanca import ineq
from salamanca.models.project import Model, Model1

from pandas.testing import assert_frame_equal

from utils import assert_almost_equal, assert_array_almost_equal


def data():
    natdata = pd.DataFrame({
        'n': [20, 25],
        'i': [135 / 20., 175 / 25.],
        'gini': [0.4, 0.35],
    }, index=pd.Index([2010, 2020], name='year'))
    subdata = pd.DataFrame({
        'n': [7, 13, 9, 16],
        'i': [10, 5, np.nan, np.nan],
        'gini': [0.5, 0.3, np.nan, np.nan],
    },
        index=pd.MultiIndex.from_product([[2010, 2020], ['foo', 'bar']],
                                         names=['year', 'name'])
    )
    return natdata, subdata


if __name__ == '__main__':
    n, s = data()
    print('data')
    print(n)
    print(s)
    print('first idx')
    print(n.iloc[0])
    print(s.loc[n.index[0]])
    print(s.loc[n.index[1]])
    print(s.loc[2020]['n'])
    print(s.loc[(2020,), 'n'])


def test_model_data_pop():
    # note all subdata order is swapped in model_data due to sorting by gini
    natdata, subdata = data()
    model = Model(natdata, subdata)

    # pop
    obs = model.model_data['N']
    exp = natdata.loc[2020]['n']
    assert_almost_equal(obs, exp)
    obs = model.model_data['n']
    exp = subdata.loc[2020]['n'] * \
        natdata.loc[2020]['n'] / subdata.loc[2020]['n'].sum()
    assert_array_almost_equal(obs, exp)


def test_model_data_error():
    natdata, subdata = data()
    ndf = natdata.copy().drop('n', axis=1)
    with pytest.raises(ValueError):
        model = Model(ndf, subdata)

    sdf = natdata.copy().drop('n', axis=1)
    with pytest.raises(ValueError):
        model = Model(natdata, sdf)


def test_Model1_solution():
    natdata, subdata = data()
    model = Model1(natdata, subdata)
    model.construct()
    model.solve()

    # this is a regression test, results tested 08-29-17
    obs = model.solution
    exp = pd.DataFrame({
        'i': np.array([7.007136, 6.995986]),
        't': np.array([0.241891, 0.185623]),
    }, index=pd.Index(['foo', 'bar'], name='name'))
    assert_frame_equal(obs, exp)


def test_Model1_result():
    natdata, subdata = data()
    model = Model1(natdata, subdata)
    model.construct()
    model.solve()
    df = model.result()

    # test columns
    obs = sorted(df.columns)
    exp = ['gini', 'i', 'n', 'n_orig']
    assert obs == exp

    # test original data
    obs = df['n_orig']
    exp = subdata.loc[2020]['n']
    assert_array_almost_equal(obs, exp)
    obs = df['n']
    exp = subdata.loc[2020]['n']
    assert_array_almost_equal(obs, exp)

    # this is a regression test, results tested 08-29-17
    obs = df[['gini', 'i']]
    exp = pd.DataFrame({
        'i': np.array([7.007136, 6.995986]),
        'gini': ineq.theil_to_gini(np.array([0.241891, 0.185623]),
                                   empirical=False)
    }, index=pd.Index(['foo', 'bar'], name='name'))
    assert_frame_equal(obs, exp)


def test_Model1_diffusion_result():
    natdata, subdata = data()
    model = Model1(natdata, subdata)
    model.construct(with_diffusion=True)
    model.solve()
    df = model.result()

    # this is a regression test, results tested 08-29-17
    obs = df[['gini', 'i']]
    exp = pd.DataFrame({
        'i': np.array([8.38271666486, 6.22222187602]),
        'gini': np.array([0.477747557316, 0.285296967258]),
    }, index=pd.Index(['foo', 'bar'], name='name'))
    assert_frame_equal(obs, exp)
