import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--level",
        action="store",
        default="full",
        choices=["quick", "full"],
        help="Set the testing level: 'quick' or 'full'.",
    )
    parser.addoption(
        "--headless",
        action="store_true",
        default=False,
        help="Don't display any graphics (runs faster)",
    )


# These parameter names match up with the parameter names for
# test functions (that is functions with the word test in the name
# in files with the word test in the name) detected by pytest,
# and we test such functions with all values in the below sets
# For example, a function with the parameter name is_yellow
# will be tested once with False and once with True.
# Notice that the key type for this dictionary is a tuple
# which allows aliasing such that multiple parameter names share the same
# test value sets.
parameter_values = {
    ("target_id",): {"quick": [0], "full": range(0, 2)},
    ("my_team_is_yellow",): {  # Probably worth running both colours even in quick mode
        "quick": [False, True],
        "full": [False, True],
    },
    ("my_team_is_right",): {
        "quick": [True],
        "full": [False, True],
    },
}


def pytest_generate_tests(metafunc):
    for param_set, cases in parameter_values.items():
        for param in param_set:
            if param in metafunc.fixturenames:
                metafunc.parametrize(param, cases[metafunc.config.getoption("level")])


@pytest.fixture
def headless(pytestconfig):
    return pytestconfig.getoption("--headless")
