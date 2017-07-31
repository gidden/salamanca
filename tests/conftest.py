import pytest


def pytest_addoption(parser):
    parser.addoption("--remote", action="store_true",
                     help="run tests requiring internet")
    parser.addoption("--slow", action="store_true",
                     help="run tests that are slow")
