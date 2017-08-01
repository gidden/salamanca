import numpy as np


from salamanca import data


class Translator(object):
    """An object to translate between national currencies in different years

    By default, inflation is calculated using GDP Deflators (though CPI is
    available as well).

    Currencies are assumed to be in MER. If in PPP, 'units' must be set
    appropriately.

    Calculation is formulated based on:
    http://www.iiasa.ac.at/web/home/research/Flagship-Projects/Global-Energy-Assessment/GEA_Annex_II.pdf
    """

    def __init__(self):
        reshape = lambda df: \
            df.pivot(index='country', columns='date', values='value')
        wb = data.WorldBank()
        self._cpi = reshape(wb.query('cpi'))
        self._gdp_deflator = reshape(wb.query('gdp_deflator'))
        self._xr = reshape(wb.query('exchange_rate'))
        self._ppp_to_mer = reshape(wb.query('ppp_to_mer'))

    def inflation(self, iso, fromyr, toyr, method=None):
        """Calculate inflation for a country

        Parameters
        ----------
        iso: str
            a 3-letter ISO code
        fromyr: int
            the starting year
        toyr: int
            the ending year
        method: str
            one of: cpi, deflator (default: deflator)
        """
        method = method or 'deflator'
        df = self._cpi if method == 'cpi' else self._gdp_deflator
        x = df.loc[iso][toyr] / df.loc[iso][fromyr]
        if np.isnan(x):
            warnings.warn(
                'Bad currency translation for {}, '
                'falling back to US inflation rates'.format(iso))
            x = self.inflation('US', fromyr, toyr, method=method)
        return x

    def exchange(self, x, iso=None, yr=None, units='MER',
                 fromiso=None, fromyr=None,
                 toiso=None, toyr=None,
                 inusd=False, inflation_method=None):
        """Exchange currency from one country/year to another.

        Parameters
        ----------
        x: float
           quantity of currency
        iso: str, optional
            a 3-letter ISO code
        yr: int, optional
            the year
        units: str, optional
            current currency units
            one of: PPP or MER
        fromiso: str, optional
            a 3-letter ISO code for the origin country
        fromyr: int, optional
            the year initial year
        toiso: str, optional
            a 3-letter ISO code for the destination country
        fromyr: int, optional
            the year final year
        inusd: bool, optional
            return currency in USD of the final year
        inflation_method: str, optional
            the argument to provide to `inflation()`
        """
        fromiso = fromiso or iso
        fromyr = fromyr or yr

        toiso = toiso or 'US'
        toyr = toyr or yr

        # is using special inusd option, do full calculation and return
        if inusd:
            if iso is None:
                raise ValueError(
                    'iso parameter must have a value if usd=True')
            savex = x
            x = self.exchange(x, fromiso='US', toiso=iso,
                              yr=fromyr, units=units)
            x *= self.inflation(iso, fromyr, toyr, method=inflation_method)
            x = self.exchange(x, fromiso=iso, toiso='US',
                              yr=toyr, units=units)
            return x

        # otherwise, assume currencies are in isos of origin
        if fromyr == toyr:
            # xr is the market exchange rate
            xr_1 = self._xr.loc[fromiso][fromyr]
            xr_2 = self._xr.loc[toiso][toyr]
            if units == 'PPP':
                # multiply by ratio of ppp conversion factor to market exchange
                # rate, see http://data.worldbank.org/indicator/PA.NUS.PPPC.RF
                xr_1 *= self._ppp_to_mer.loc[fromiso][fromyr]
                xr_2 *= self._ppp_to_mer.loc[toiso][toyr]
            x *= xr_2 / xr_1
        elif iso is not None:
            # only account for inflation in same country, no exchanging
            x *= self.inflation(iso, fromyr, toyr, method=inflation_method)
        else:
            # account for exchange rate and inflation
            x = self.exchange(x, fromiso=fromiso, toiso=toiso,
                              yr=fromyr, units=units)
            x *= self.inflation(toiso, fromyr, toyr, method=inflation_method)

        return x
