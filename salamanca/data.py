
import json
import logging
import os
import urllib

import pandas as pd

WB_URL = 'http://api.worldbank.org/countries/all/indicators'

CACHE_DIR = os.path.expanduser(
    os.path.join('~', '.local', 'salamanca', 'data'))

WB_INDICATORS = {
    'SP.POP.TOTL': 'total_population',
}

INDICATORS_WB = {d: k for k, d in WB_INDICATORS.items()}


def query_wb(url, tries=5):
    baseurl = url
    pages = 1
    page = 0
    result = []
    while page < pages:
        page += 1
        url = '{}&page={}'.format(baseurl, page)
        query = urllib.urlopen(url)
        _result = json.loads(query.read())
        query.close()
        pages = _result[0]['pages']
        result += _result[1]
    return result


def wb_indicator(i, tries=5, iso3=True, cache=True):
    i = indicator
    wb = i if i in WB_INDICATORS else WB_INDICATORS[i]
    ind = i if i != wb else INDICATORS_WB[i]

    # pop kwargs for all possible queries in indicator api
    # if they match cached file, try to grab from cache
    # otherwise pass directly
    # if subset requested here, then just pass through

    url = '{}/{}?format=json'.format(WB_URL, wb)
    result = query_wb(url, tries=tries)
    df = pd.DataFrame(result)
    df['indicator'] = ind
    df['country'] = df['country'].apply(lambda x: x['id'])
    if iso3:
        # implement this
        pass
