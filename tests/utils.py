import numpy as np


def logging_on():
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def assert_almost_equal(x, y, eps=1e-6):
    assert abs(x - y) < eps


def assert_array_almost_equal(x, y, eps=1e-6):
    assert np.all(abs(x - y) < eps)
