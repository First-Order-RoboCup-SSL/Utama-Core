"""main.py resolves strategy names through the catalog; a bad name must fail with the valid ones listed."""

import pytest

import main
from utama_core.engine.abstract_strategy import AbstractStrategy
from utama_core.strategy import catalog


def test_known_strategy_name_builds_an_abstract_strategy():
    assert isinstance(main._strategy("split_shape"), AbstractStrategy)


def test_unknown_strategy_name_exits_listing_catalog_names():
    with pytest.raises(SystemExit) as excinfo:
        main._strategy("no_such_strategy")
    message = str(excinfo.value)
    assert "no_such_strategy" in message
    assert "split_shape" in message
    assert all(name in message for name in catalog.names())
