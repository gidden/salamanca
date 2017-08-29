import pytest

import numpy as np
import pandas as pd

from salamanca import ineq
from salamanca.models.project import Model

from utils import assert_almost_equal, assert_array_almost_equal


def data():
    natdata = pd.DataFrame({
        'n': [20, 25],
        'i': [105, 110],
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


# def test_Model1_full():
#     natdata, subdata = data()
#     model = Model1(natdata, subdata)
#     model.construct()
#     model.solve()

#     # this is the theil result
#     # equivalent ginis are: 0.19798731, 0.45663392
#     obs = model.solution
#     # solution is ordered small to large
#     exp = np.array([0.062872, 0.369337])
#     assert_array_almost_equal(obs, exp)
#     theil_exp = exp

#     df = model.result()
#     obs = sorted(df.columns)
#     exp = ['gini', 'gini_orig', 'i', 'i_orig', 'n', 'n_orig']
#     assert obs == exp

#     obs = df['gini_orig']
#     exp = subdata['gini']
#     assert_array_almost_equal(obs, exp)

#     obs = df['i_orig']
#     exp = subdata['i']
#     assert_array_almost_equal(obs, exp)

#     obs = df['n_orig']
#     exp = subdata['n']
#     assert_array_almost_equal(obs, exp)


# def test_Model1_result():
#     natdata, subdata = data()
#     model = Model1(natdata, subdata)
#     model.construct()
#     model.solve()

#     # ginis in original order
#     obs = model.result()['gini'].values
#     exp = [0.45663392, 0.19798731]
#     assert_array_almost_equal(obs, exp)
