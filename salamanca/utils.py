import os

import pandas as pd

from copy import deepcopy


CACHE_DIR = os.path.expanduser(
    os.path.join('~', '.local', 'salamanca', 'data'))


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
        data.to_csv(self.full_path(source, indicator),
                    index=False, encoding='utf-8')

    def read(self, source, indicator):
        return pd.read_csv(self.full_path(source, indicator))

    def exists(self, source, indicator):
        return os.path.exists(self.full_path(source, indicator))


class AttrObject(object):
    """Simple base class to have dictionary-like attributes attached to an 
    object
    """

    def __init__(self, **kwargs):
        self.update(copy=False, **kwargs)

    def update(self, copy=True, override=True, **kwargs):
        """Update attributes.

        Parameters
        ----------
        copy : bool, optional, default: True
            operate on a copy of this object
        override : bool, optional, default: True
            overrides attributes if they already exist on the object
        """
        x = deepcopy(self) if copy else self
        for k, v in kwargs.items():
            if override or getattr(x, k, None) is None:
                setattr(x, k, v)
        return x
