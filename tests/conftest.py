"""Pytest wiring.

Two suites cost minutes rather than seconds and are skipped unless asked for.
`mutation` recompiles the kernel once per mutant; `slow` is deep perft, which
holds a few hundred megabytes and runs for about half a minute per case.

    pytest tests/                          # neither
    pytest tests/ --mutation --slow        # both
    pytest tests/ --slow -m slow           # deep perft alone
"""

import pytest

OPT_IN = {
    "mutation": ("--mutation", "suite that mutates the kernel to test the tests"),
    "slow": ("--slow", "deep perft, minutes of CPU and a few hundred MB"),
}


def pytest_addoption(parser):
    for marker, (flag, help_text) in OPT_IN.items():
        parser.addoption(flag, action="store_true", default=False, help=f"run the {help_text}")


def pytest_configure(config):
    for marker, (flag, help_text) in OPT_IN.items():
        config.addinivalue_line("markers", f"{marker}: {help_text}; needs {flag}")


def pytest_collection_modifyitems(config, items):
    for marker, (flag, _) in OPT_IN.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"{marker} is opt-in: pass {flag}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
