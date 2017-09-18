import pytest

import numpy as np
import pandas as pd

from salamanca import ineq
from salamanca.models.project import Model, Model1, Model2, Model3
from salamanca.models.project import Runner

from pandas.testing import assert_frame_equal

from utils import assert_almost_equal, assert_array_almost_equal


def data():
    natdata = pd.DataFrame({
        'n': [20, 25],
        'i': [135 / 20., 175 / 25.],
        'gini': [0.4, 0.35],
    }, index=pd.Index([2010, 2020], name='year'))
    subdata = pd.DataFrame({
        'n': np.array([7, 13, 9, 16]) * 0.75,
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
    print('idx groups')
    print(n.loc[n.index[0]:n.index[1]])
    print(s.loc[n.index[0]:n.index[1]])


def test_model_data_pop():
    # note all subdata order is swapped in model_data due to sorting by gini
    natdata, subdata = data()
    model = Model(natdata, subdata)

    # pop, fraction
    obs = model.model_data['n_frac']
    exp = subdata.loc[2020]['n'] / subdata.loc[2020]['n'].sum()
    assert_array_almost_equal(obs, exp)

    # pop, absolute
    obs = model.model_data['n_frac'] * model.model_data['N']
    exp = subdata.loc[2020]['n'] * \
        natdata.loc[2020]['n'] / subdata.loc[2020]['n'].sum()
    assert_array_almost_equal(obs, exp)


def test_model_data_pop_override():
    # note all subdata order is swapped in model_data due to sorting by gini
    natdata, subdata = data()
    model = Model(natdata, subdata, override_national={'n': True})

    # pop, fraction
    obs = model.model_data['n_frac']
    exp = subdata.loc[2020]['n'] / subdata.loc[2020]['n'].sum()
    assert_array_almost_equal(obs, exp)

    # pop, absolute
    obs = model.model_data['n_frac'] * model.model_data['N']
    exp = subdata.loc[2020]['n']
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

    # this is a regression test, results tested 09-04-17
    obs = model.solution
    exp = pd.DataFrame({
        # note this is scaled by I
        'i': np.array([1.074197, 0.9582637]),
        't': np.array([0.253772, 0.173215]),
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
    exp = subdata.loc[2020]['n'] * \
        natdata.loc[2020]['n'] / subdata.loc[2020]['n'].sum()
    assert_array_almost_equal(obs, exp)

    # this is a regression test, results tested 08-29-17
    obs = df[['gini', 'i']]
    exp = pd.DataFrame({
        'i': np.array([1.074197, 0.9582637]) * natdata.loc[2020]['i'],
        'gini': ineq.theil_to_gini(np.array([0.253772, 0.1732152]),
                                   empirical=False)
    }, index=pd.Index(['foo', 'bar'], name='name'))
    assert_frame_equal(obs, exp)


def test_Model1_diffusion_result():
    natdata, subdata = data()
    model = Model1(natdata, subdata)
    diffusion = {'income': True, 'theil': True}
    model.construct(diffusion=diffusion)
    model.solve()
    df = model.result()

    # this is a regression test, results tested 09-18-17
    obs = df[['gini', 'i']]
    exp = pd.DataFrame({
        'i': np.array([8.38271666486, 6.22222187602]),
        'gini': np.array([0.477747557316, 0.285296967258]),
    }, index=pd.Index(['foo', 'bar'], name='name'))
    assert_frame_equal(obs, exp)


def test_Model2_result():
    natdata, subdata = data()
    model = Model2(natdata, subdata)
    model.construct()
    model.solve()
    df = model.result()

    # this is a regression test, results tested 09-06-17
    obs = df[['gini', 'i']]
    exp = pd.DataFrame({
        'i': [7.80113, 6.549360],
        'gini': [0.348809, 0.347297]
    }, index=pd.Index(['foo', 'bar'], name='name'))
    assert_frame_equal(obs, exp)


def test_Model2_diffusion_result():
    natdata, subdata = data()
    model = Model2(natdata, subdata)
    diffusion = {'theil': True}
    model.construct(diffusion=diffusion, theil_within=0.1)
    model.solve()
    df = model.result()

    # this is a regression test, results tested 09-06-17
    obs = df[['gini', 'i']]
    exp = pd.DataFrame({
        'i': [6.39826598759, 7.33847538198],
        'gini': [0.47774761513, 0.285297024826]
    }, index=pd.Index(['foo', 'bar'], name='name'))
    assert_frame_equal(obs, exp)


def test_Model3_result():
    natdata, subdata = data()
    model = Model3(natdata, subdata)
    model.construct()
    model.solve()
    df = model.result()

    # this is a regression test, results tested 09-06-17
    obs = df[['gini', 'i']]
    exp = pd.DataFrame({
        'i': [7.12418694598, 6.93014484288],
        'gini': [0.390495617969, 0.323382701378],
    }, index=pd.Index(['foo', 'bar'], name='name'))
    assert_frame_equal(obs, exp)


def test_Model3_args_result():
    natdata, subdata = data()
    model = Model3(natdata, subdata)
    diffusion = {'theil': True}
    model.construct(diffusion=diffusion, theil_weight=1.5, threshold=0.8)
    model.solve()
    df = model.result()

    # this is a regression test, results tested 09-06-17
    obs = df[['gini', 'i']]
    exp = pd.DataFrame({
        'i': [6.34853261384, 7.36645040472],
        'gini': [0.477747607525, 0.285297068982],
    }, index=pd.Index(['foo', 'bar'], name='name'))
    assert_frame_equal(obs, exp)


def test_runner_horizon():
    # test on actual data
    natdata, subdata = data()
    runner = Runner(natdata, subdata, Model)
    assert [(2010, 2020)] == list(runner.horizon())

    # test of extended fake data
    df = pd.DataFrame({'n': range(3)}, index=[2010, 2020, 2030])
    runner = Runner(df, subdata, Model)
    assert [(2010, 2020), (2020, 2030)] == list(runner.horizon())


def test_runner_data():
    # test on actual data
    natdata, subdata = data()
    runner = Runner(natdata, subdata, Model)
    obs_n, obs_s = runner._data(2010, 2020)
    assert_frame_equal(obs_n, natdata)
    assert_frame_equal(obs_s[subdata.columns], subdata.sort_index())

    # test of extended fake data
    df1 = pd.DataFrame({'n': range(3)},
                       index=pd.Index([2010, 2020, 2030], name='year'))
    idx = pd.MultiIndex.from_product([[2010, 2020, 2030], ['foo', 'bar']],
                                     names=['year', 'name'])
    df2 = pd.DataFrame({'n': range(3, 9)}, index=idx)
    runner = Runner(df1, df2, Model)
    # 2010-2020
    obs_1, obs_2 = runner._data(2010, 2020)
    exp = df1.loc[2010:2020]
    assert_frame_equal(obs_1, exp)
    exp = df2.loc[2010:2020].sort_index()
    exp['n_orig'] = exp['n']
    assert_frame_equal(obs_2, exp)
    # 2020-2030
    obs_1, obs_2 = runner._data(2020, 2030)
    exp = df1.loc[2020:2030]
    assert_frame_equal(obs_1, exp)
    exp = df2.loc[2020:2030].sort_index()
    exp['n_orig'] = exp['n']
    assert_frame_equal(obs_2, exp)


def test_runner_update():
    natdata, subdata = data()
    runner = Runner(natdata, subdata, Model)

    # before update, the result should be same as subdata
    obs = runner.result()
    exp = subdata
    exp['n_orig'] = exp['n']
    assert_frame_equal(obs, exp)

    # update works on non-sorted index
    df = pd.DataFrame({
        'n': [15., 10.],
        'i': [9, 7],
        'gini': [0.47, 0.29],
    }, index=pd.Index(['foo', 'bar'], name='name'))
    runner._update(2020, df)
    obs = runner.result()
    exp.loc[(2020,), 'gini'] = df['gini'].values
    exp.loc[(2020,), 'i'] = df['i'].values
    exp.loc[(2020,), 'n'] = df['n'].values
    assert_frame_equal(obs, exp)
