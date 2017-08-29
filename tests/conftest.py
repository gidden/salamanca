import pytest


OPTIONS = {
    # decorator for test requiring internet
    '--remote': {
        'help': 'run tests requiring internet',
        'reason': 'need --remote option to run',
    },
    # decorator for slow tests
    '--slow': {
        'help': 'run tests that are slow',
        'reason': 'need --slow option to run',
    }
}

#
# generic functions required for adding CLIs
# see https://docs.pytest.org/en/latest/example/simple.html
#


def pytest_addoption(parser):
    for opt in OPTIONS:
        parser.addoption(opt, action="store_true",
                         default=False, help=OPTIONS[opt]['help'])


def pytest_collection_modifyitems(config, items):
    for opt in OPTIONS:
        if config.getoption(opt):
            return
    skips = {
        opt: pytest.mark.skip(reason=OPTIONS[opt]['reason']) for opt in OPTIONS
    }
    for item in items:
        for opt in OPTIONS:
            if opt.lstrip('--') in item.keywords.keys():
                item.add_marker(skips[opt])
