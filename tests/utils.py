import pytest

import numpy as np


def logging_on():
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def assert_almost_equal(x, y, eps=1e-6):
    assert abs(x - y) < eps


def assert_array_almost_equal(x, y, eps=1e-6):
    assert np.all(abs(x - y) < eps)


# decorator for test requiring internet
remote = pytest.mark.skipif(
    not pytest.config.getoption("--remote"),
    reason="need --remote option to run"
)

# decorator for slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--slow"),
    reason="need --slow option to run"
)
