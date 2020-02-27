import pytest


def pytest_addoption(parser):
    parser.addoption("--remote", action="store_true",
                     help="run tests requiring internet")
    parser.addoption("--slow", action="store_true",
                     help="run tests that are slow")



def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "remote: mark test as requiring internet")

def pytest_collection_modifyitems(config, items):
    markers = ['slow', 'remote']
    # don't mark those *not* in CLI
    markers = [m for m in markers if not config.getoption('--{}'.format(m))]
    
    for m in markers:
        for item in items:
            skip = pytest.mark.skip(reason="need --{} option to run".format(m))
            if m in item.keywords:
                item.add_marker(skip)
