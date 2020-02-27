import pytest

import numpy as np


def logging_on():
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def assert_almost_equal(x, y, eps=1e-3):
    print(abs(x - y) / x)
    assert abs(x - y) / x < eps


def assert_array_almost_equal(x, y, eps=1e-6):
    assert np.all(abs(x - y) < eps)

