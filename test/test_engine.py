from datetime import datetime
import random
import pytest
import numpy as np
from backtester.engine import Engine, StrategyException
import backtester.datastreams as datastreams


def test_strategy_length():
    """Tests whether the strategy is producing a valid alpha vector."""
    data = datastreams.csv_timeseries(filename="1yr_snp.csv")
    model = Engine(1000, data, strat1, context=1)
    with pytest.raises(StrategyException):
        model.run()


def test_strategy_success():
    """Tests whether the strategy is producing a valid alpha vector."""
    data = datastreams.csv_timeseries(filename="1yr_snp.csv")
    model = Engine(1000, data, strat2, context=1)
    try:
        model.run()
    except StrategyException:
        assert False


def strat1(data, context):
    """
    Purpose: one
    Must return an iterable of weightings or "alphas", base on the data.
    Currently uses the adjusted close but the method would be up to the user.
    """
    # closing = data["Adj Close"].T
    # znormed = (closing - closing.mean()) / closing.std()
    # return znormed[0].values
    return 1
    # return alphas


def strat2(data, context):
    """
    Purpose: one
    Must return an iterable of weightings or "alphas", base on the data.
    Currently uses the adjusted close but the method would be up to the user.
    """
    closing = data["Adj Close"].T
    return np.array([random.uniform(-1, 1) for _ in range(len(closing))])
