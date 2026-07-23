import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--runslow', action='store_true', default=False,
        help='run slow tests (the long-running example scripts and notebooks)',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'slow: long-running example tests, skipped unless --runslow is given',
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        return
    skip_slow = pytest.mark.skip(reason='need --runslow option to run')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)
