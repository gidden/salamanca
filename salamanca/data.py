import contextlib
import json
import logging
import os
import urllib
import warnings

import pandas as pd

CACHE_DIR = os.path.expanduser(
    os.path.join('~', '.local', 'salamanca', 'data'))

WB_INDICATORS = {
    'SP.POP.TOTL': 'total_population',
    'PA.NUS.PPP': 'ppp_to_mer',
    'FP.CPI.TOTL': 'cpi',
    'PA.NUS.FCRF': 'exchange_rate',
    'NY.GDP.DEFL.ZS': 'gdp_deflator',
}

INDICATORS_WB = {d: k for k, d in WB_INDICATORS.items()}

WB_URL = 'http://api.worldbank.org/en/countries/{iso}/indicators'


def backend():
    # implement configuration reading here
    return CSVBackend()


class Backend(object):
    """Abstract base class for on-disc data backends"""

    def __init__(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    def write(self, source, indicator, data):
        raise NotImplementedError()

    def read(self, source, indicator):
        raise NotImplementedError()

    def exists(self, source, indicator):
        raise NotImplementedError()


class CSVBackend(Backend):
    """Backend class for CSV files"""

    def __init__(self):
        super(CSVBackend, self).__init__()

    def fname(self, source, indicator):
        return '{}_{}.csv'.format(source, indicator)

    def full_path(self, source, indicator):
        return os.path.join(CACHE_DIR, self.fname(source, indicator))

    def write(self, source, indicator, data):
        data.to_csv(self.full_path(source, indicator), index=False)

    def read(self, source, indicator):
        return pd.read_csv(self.full_path(source, indicator))

    def exists(self, source, indicator):
        return os.path.exists(self.full_path(source, indicator))


@contextlib.contextmanager
def query_rest_api(url, tries=5, asjson=True):
    """Query a REST API online

    Parameters
    ----------
    url : str
       url to query
    tries : int, optional
       number of times to try query before raising an IOError
    asjson : bool, optional
       read query as a json object
    """
    logging.debug('Querying: {}, tries left: {}'.format(url, tries))
    n = 0
    while n < tries:
        try:
            query = urllib.urlopen(url)
            result = query.read()
            if asjson:
                result = json.loads(result)
            yield result
            query.close()
            break
        except IOError:
            n += 1
    if n == tries:
        raise IOError('Query failed: {}'.format(url))


class WorldBank(object):
    """A simple object for querying the World Bank's REST API"""

    def __init__(self):
        self.query_args = ['date', 'MRV', 'Gapfill', 'frequency']

    def _query_url(self, wb_ind, **kwargs):
        iso = kwargs.pop('iso', 'all')
        url = '{}/{}?format=json&per_page=1000'.format(
            WB_URL.format(iso=iso), wb_ind)
        urlargs = ''
        for arg in self.query_args:
            if arg in kwargs:
                arg = '{}={}'.format(arg, kwargs[arg])
                urlargs = '&'.join([urlargs, arg])
        return url + urlargs

    def _do_query(self, url, tries=5):
        baseurl = url if '?' in url else url + '?'
        pages = 1
        page = 0
        result = []
        while page < pages:
            page += 1
            url = '{}&page={}'.format(baseurl, page)
            failed = False
            with query_rest_api(url) as _result:
                if 'message' in _result[0]:
                    failed = True
                    msg = _result[0]['message'][0]
                else:
                    pages = _result[0]['pages']
                    result += _result[1]
            if failed:
                raise IOError(msg)
            logging.debug('Page {} of {} Complete'.format(page, pages))
        return result

    def query(self, indicator, tries=5, iso3=True, use_cache=True, overwrite=False, **kwargs):
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
        url = self._query_url(wb, **kwargs)
        result = self._do_query(url, tries=tries)

        # construct as data frame
        df = pd.DataFrame(result)
        df['indicator'] = ind
        df['country'] = df['country'].apply(lambda x: x['id'])
        df['value'] = df['value'].astype(float)
        df['decimal'] = df['decimal'].astype(float)
        try:
            df['date'] = df['date'].astype(int)
        except:
            pass

        # write to disc if we're caching
        if use_cache and (not exists or overwrite):
            db.write(source, ind, df)

        return df


def download_wb_data(log=False):
    # turn this into a CLI
    if log:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    wb = WorldBank()
    for ind in INDICATORS_WB:
        wb.query(ind)
