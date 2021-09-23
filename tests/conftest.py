"""Fixtures."""
from pathlib import Path
from typing import Any

import pytest


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--rungpu", action="store_true", default=False, help="run tests on gpu")


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "gpu: mark test as needing to run on gpu")


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
    if config.getoption("--rungpu"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_gpu = pytest.mark.skip(reason="need --rungpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def root() -> Path:
    return Path("~/Data").expanduser()
