import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--level",
        action="store",
        default="full",
        choices=["quick", "full"],
        help="Set the testing level: 'quick' or 'full'."
    )
    parser.addoption(
        "--headless",
        action="store_true",
        default=False,
        help="Don't display any graphics (runs faster)"
    )

parameter_values = {
    "shooter_id": {
        "quick": [5],
        "full": range(6)
    },
    "is_yellow": {                  # Probably worth running both colours even in quick mode
        "quick": [False, True],
        "full": [False, True]
    },
    "robot_to_place": {
        "quick": [1],
        "full": range(6)
    },
    "target_robot": {
        "quick": [1],
        "full": [1]
    },
    "defender_id": {
        "quick": [1],
        "full": [1,2]
    }
}

def pytest_generate_tests(metafunc):
    for (param, cases) in parameter_values.items():
        if param in metafunc.fixturenames:
            metafunc.parametrize(param, cases[metafunc.config.getoption("level")])

@pytest.fixture
def headless(pytestconfig):
    return pytestconfig.getoption("--headless")
