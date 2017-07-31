import contextlib
import json
import logging
import os
import urllib

import pandas as pd


CACHE_DIR = os.path.expanduser(
    os.path.join('~', '.local', 'salamanca', 'data'))

WB_INDICATORS = {
    'SP.POP.TOTL': 'total_population',
}

INDICATORS_WB = {d: k for k, d in WB_INDICATORS.items()}

WB_URL = 'http://api.worldbank.org/en/countries/{iso}/indicators'


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
        url = '{}/{}?format=json'.format(WB_URL.format(iso=iso), wb_ind)
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

    def indicator(i, tries=5, iso3=True, cache=True, **kwargs):
        """
        kwargs include
        iso
        'date', 
        'MRV', 
        'Gapfill', 
        'frequency'
        """
        i = indicator
        wb = i if i in WB_INDICATORS else WB_INDICATORS[i]
        ind = i if i != wb else INDICATORS_WB[i]

        # pop kwargs for all possible queries in indicator api
        # if they match cached file, try to grab from cache
        # otherwise pass directly
        # if subset requested here, then just pass through
        url = self._query_url(wb, **kwargs)
        result = self._do_query(url, tries=tries)
        df = pd.DataFrame(result)
        df['indicator'] = ind
        df['country'] = df['country'].apply(lambda x: x['id'])
        if iso3:
            # implement this
            pass
