"""A module for querying datasources (e.g., the World Bank Indicators). They
can optionally be stored locally to reduce internet queries.
"""

import contextlib
import json
import logging
import os
import requests
import warnings

import pandas as pd


from salamanca.utils import backend

WB_INDICATORS = {
    'SP.POP.TOTL': 'total_population',
    'PA.NUS.PPPC.RF': 'ppp_to_mer',  # conversion factor [PPP / MER]
    'FP.CPI.TOTL': 'cpi',
    'PA.NUS.FCRF': 'exchange_rate',
    'NY.GDP.DEFL.ZS': 'gdp_deflator',
    'SI.POV.DDAY': 'below_1_90_dollars_per_day_ppp',
    'NE.CON.PETC.ZS': 'household_fraction_gdp',
}

INDICATORS_WB = {d: k for k, d in WB_INDICATORS.items()}

WB_URL = 'http://api.worldbank.org/v2/country/{iso}/indicator/{indicator}'

EU_COUNTRIES = [
    'AUT', 'BEL', 'CYP',
    'DEU', 'ESP', 'EST',
    'FIN', 'FRA', 'GRC',
    'IRL', 'ITA', 'LTU',
    'LUX', 'LVA', 'MLT',
    'NLD', 'PRT', 'SVK',
    'SVN',
]


@contextlib.contextmanager
def query_rest_api(url, params=None, tries=5):
    """Query a REST API online

    Parameters
    ----------
    url : str
       url to query
    tries : int, optional
       number of times to try query before raising an IOError
    """
    params = {
        'format': 'json',
        'per_page': 1000,
        **(params if params is not None else {})
    }
    logging.debug('Querying: {}, tries left: {}'.format(url, tries))
    n = 0
    while n < tries:
        try:
            q = requests.get(url, params=params)
            result = q.json()
            if isinstance(result, dict):
                meta = result
            elif isinstance(result, list):
                meta = result[0]
            else:
                raise RuntimeError("Unexpected reply payload: {}".format(result))
            if 'message' in meta:
                raise RuntimeError(meta['message'])
            yield result
            break
        except IOError:
            n += 1
    else:
        raise RuntimeError('Query failed: {}'.format(q.url))


class WorldBank(object):
    """A simple object for querying the World Bank's REST API"""

    def __init__(self):
        self.query_args = ['date', 'MRV', 'Gapfill', 'frequency']

    def _do_query(self, wb, params=None, tries=5):
        params = params.copy()
        url = WB_URL.format(indicator=wb, iso=params.pop('iso', 'all'))

        pages = 1
        params['page'] = 0

        result = []
        while params['page'] < pages:
            params['page'] += 1
            with query_rest_api(url, params=params) as _result:
                pages = _result[0]['pages']
                result += _result[1]
            logging.debug('Page {} of {} Complete'.format(params['page'], pages))
        return result

    def query(self, indicator, tries=5, use_cache=True, overwrite=False, **kwargs):
        """
        kwargs include
        iso
        'date',
        'MRV',
        'Gapfill',
        'frequency'
        """
        i = indicator
        if i in WB_INDICATORS:
            # supported wb indicators
            wb = i
            ind = WB_INDICATORS[i]
        elif i in INDICATORS_WB:
            # supported indicator
            ind = i
            wb = INDICATORS_WB[i]
        else:
            # not supported indicator
            ind = i
            wb = i

        # use cache if no other API kwargs present
        if use_cache and kwargs:
            warnings.warn('Can not cache queries with additional arguments')
            use_cache = False

        # read from disc if it already exists
        if use_cache:
            db = backend()
            source = 'wb'
            exists = db.exists(source, ind)
            if exists:
                return db.read(source, ind)

        # otherwise get raw data
        result = self._do_query(wb, params=kwargs, tries=tries)

        # construct as data frame
        df = pd.DataFrame(result)
        df['country'] = df['country'].apply(lambda x: x['id'])
        df.drop(['decimal', 'indicator', 'countryiso3code',
                 'unit', 'obs_status'],
                axis=1, inplace=True)
        try:
            # convert years if possible
            df['date'] = df['date'].astype(int)
        except:
            pass

        # fix up country names to gaurantee ISO3-standard
        # in a recent update, some tables were found to be id'd to iso2,
        # which is fixed here
        # TODO: why are there NaNs? why would any be empty?
        df = df.dropna(subset=['country'])
        df = df[df['country'] != '']
        if len(df['country'].iloc[0]) == 2:
            meta = self.iso_metadata()
            mapping = {r['iso2Code']: r['id'] for idx, r in meta.iterrows()}
            df['country'] = df['country'].map(mapping)


        # write to disc if we're caching
        if use_cache and (not exists or overwrite):
            db.write(source, ind, df)

        return df

    def iso_metadata(self, overwrite=False, map_cols=None):
        db = backend()
        source = 'wb'
        ind = 'iso_mapping'
        if overwrite or not db.exists(source, ind):
            url = 'http://api.worldbank.org/v2/country'
            with query_rest_api(url) as x:
                df = pd.DataFrame(x[1])
                idcols = ['adminregion', 'incomeLevel',
                          'lendingType', 'region']
                for col in idcols:
                    df[col] = df[col].apply(lambda x: x['id'])
            db.write(source, ind, df)

        df = db.read(source, ind)
        if map_cols:
            df = df[map_cols].set_index(map_cols[0])[map_cols[1]]
        return df

    def to_wide(self, df):
        return df.pivot(index='country',
                        columns='date',
                        values='value').reset_index()

    def to_long(self, df):
        return (df
                .melt(id_vars='country', value_vars=df.columns[1:])
                .sort_values(['country', 'date'], ascending=[True, False])
                .reset_index(drop=True))

    def _merge_eu(self, df):
        df = self.to_wide(df).set_index('country')
        df.loc[EU_COUNTRIES] = df.loc[EU_COUNTRIES].fillna(df.loc['EMU'])
        df = self.to_long(df.reset_index())
        return df

    def cpi(self, **kwargs):
        df = self.query('cpi', **kwargs)
        return df

    def exchange_rate(self, **kwargs):
        df = self.query('exchange_rate', **kwargs)

        # update newer currency unions
        df = self._merge_eu(df)
        return df

    def gdp_deflator(self, **kwargs):
        df = self.query('gdp_deflator', **kwargs)
        return df

    def ppp_to_mer(self, **kwargs):
        df = self.query('ppp_to_mer', **kwargs)
        return df

    def household_fraction_gdp(self, **kwargs):
        df = self.query('household_fraction_gdp', **kwargs)
        return df
