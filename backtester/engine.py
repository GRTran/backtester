"""Backtesting engine. 

This module will have step by step actions that are performed when running the backtesting algorithm.

There will be a 
"""

from typing import Iterable
from datetime import datetime
from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np


class Engine:
    """The main runner for the backtesting platform.

    The frequency of applying the strategy to the data is by default the interval between
    two timeseries data points. An order place will only be executed one the next days open.


    """

    def __init__(
        self,
        initial_amount: float,
        timeseries: pd.DataFrame,
        user_strategy,
        start: datetime | None = None,
        end: datetime | None = None,
        context=None,
    ):
        self.data = timeseries
        self.cash = np.zeros(len(timeseries))
        self.cash[0] = initial_amount
        self.strategy = user_strategy
        self.context = context
        self.pos_id = 0
        self.history = pd.DataFrame(
            columns=["symbol", "open_price", "shares", "close_price", "PnL"]
        )
        # Vector tickers, with orders and trades in order of shown tickers
        self.tickers = [ticker for ticker in self.data["Adj Close"].columns]
        # Order vector contains number of shares to be traded
        self.orders = np.zeros(len(self.tickers))
        # self.trades = np.zeros(len(self.tickers))
        self.trades = Trade(len(self.tickers))

        # Data capture
        self.out_cash = np.zeros(len(timeseries))

    def run(self):
        """Execute the backtester.

        The algorithm cannot be vectorised in time but can be across assets

        At each timestep we are doing actions for i-1-th closing and for i-th day opening.
        In practice, these actions would occur at different times but for backtesting it is
        sufficient for them to occur together.

        The positions are closed at the end of the trading day. Orders are also placed at the
        same time. Then positions are opened on the morning of the next trading day
        """
        for i in range(len(self.data)):
            logging.debug(f"Progress: {i/len(self.data)*100.:.2f}%")
            # Evening of i-th day
            # Close-out previous positions and adjust cash
            self._close_positions(i)
            # Store output data
            # self._store_outputs(i)
            # Use closing price of the i-th day
            self._place_orders(i)
            # Morning of i+1-th day
            self._place_trades(i)

        # Model is now complete, run a post-processer

    def _close_positions(self, i: int) -> None:
        """Goes through all positions and closes them out.

        The positions are closed at the end of the trading day, using the i-th
        datapoint.


        Args:
            i (int): The current time period index.
        """
        # Close position and add to history
        for i_ticker in range(len(self.tickers)):
            if self.trades.open_trade(i_ticker):
                close = self.data["Adj Close"].loc[i, self.tickers[i_ticker]]
                pnl = (close - self.trades.price[i_ticker]) * self.trades.shares[
                    i_ticker
                ]
                # Adjust cash by value of trade with and PnL generated
                self.cash[i] += abs(self.trades.value(i_ticker)) + pnl
                # self.history.loc[self.trades.id[i], ["close_price", "PnL"]] = [close, pnl]
        # Closing out
        self.trades.clear()

    def _place_orders(self, i: int) -> None:
        """Use the user supplied strategy to create alpha signals which are used to
        place trading orders.

        Orders are placed at the end of the day at the adjusted closing of the stock.
        The alphas returned by the user function are mapped onto a -1 to +1 scale and
        the number of shares to be purchased of each stock in the universe is
        calculated so that the available cash is distributed according to these alphas.
        Both long and short positions are equally weighted.

        Args:
            i (int): The current time period index.

        Raises:
            StrategyException: An error in how the strategy has been implemented is
             detected.
        """
        # Can't place orders on first data entry point because we do not have previous day's close
        alphas = self.strategy(self.data.loc[:i, :], self.context)

        # Perform some checks on the user response to make sure it satisfies requirements
        if not isinstance(alphas, Iterable) or len(alphas) != len(self.tickers):
            raise StrategyException(
                f"the user-function {self.strategy} must return alpha vector of same length ticker count"
            )

        # Fill any NaN values with zeros
        alphas[np.isnan(alphas)] = 0.0
        # Transform the alphas onto a -1 to +1 range. Integral of position should be total cash available.
        alphas = 2 * (alphas - np.min(alphas)) / (np.max(alphas) - np.min(alphas)) - 1

        # Calculate the number of shares to be bought using the current opening price
        # Total alpha
        total = np.sum(np.abs(alphas))
        # Adjust available cash if there must be a minimum in cash account
        available_cash = self.cash[i]

        # The total value of each position in the universe
        value = (alphas / total) * available_cash
        # Number of shares is then the value / closing price of the day.
        nshares = value / self.data.loc[i, "Adj Close"]
        # Create the orders
        self.orders = nshares
        # self.orders += [Order(tick, share) for tick, share in zip(self.tickers, nshares)]

    def _place_trades(self, i: int) -> None:
        """Goes through all orders and attempts to place the trade using the open price
        for the next morning, i.e. i+1-th day.

        An eligibility check is performed to ensure that there is sufficient capital to
        place a trade. If there are numerous overnight changes in stock price, some
        trades may not be placed.

        TODO: Add a feature for fixed to approximate remaining capital so that the user
        can fulfill all trades if there must be fixed remaining cash in the trading
        account, but there is some leeway.

        Args:
            i (int): The current time period index.
        """
        # Move cash over to next morning if there is a next morning
        if len(self.cash) == i + 1:
            return
        else:
            self.cash[i + 1] = self.cash[i]

        # Convert orders at previous day's closing to trades using todays open price
        for i_ticker, order in enumerate(self.orders):
            open_price = self.data["Open"].loc[i + 1, self.tickers[i_ticker]]
            # if order.eligible(open_price, self.cash):
            if eligible(order, open_price, self.cash[i + 1]):
                # Place the trade and reduce available cash in the account
                self.trades.id[i_ticker] = self.pos_id
                self.trades.price[i_ticker] = open_price
                self.trades.shares[i_ticker] = order
                self.cash[i + 1] -= abs(self.trades.value(i_ticker))
                # Add open trade to trade history
                # self.history.loc[self.pos_id] = [
                #     t.ticker,
                #     t.price,
                #     t.shares,
                #     None,
                #     None,
                # ]
                # increment trade counter
                self.pos_id += 1

    def _store_outputs(self, i: int) -> None:
        """Store required outputs used for post-processing.

        Args:
            i (int): The current time period index.
        """
        # The i-th time p
        self.out_cash[i] = self.cash


class Trade:

    def __init__(self, universe_size: int) -> None:
        self.id = np.zeros(universe_size)
        self.price = np.zeros(universe_size)
        self.shares = np.zeros(universe_size)

    def value(self, i: int):
        return self.price[i] * self.shares[i]

    def open_trade(self, i: int) -> bool:
        """Checks whether the shares are non-zero to see if a trade is open.

        Args:
            i (int): The ticker index.

        Returns:
            bool: True if trade is open and False otherwise.
        """
        return abs(self.shares[i]) > 1e-6

    def clear(self):
        """Clear all trade entries."""
        self.id[:] = 0.0
        self.price[:] = 0.0
        self.shares[:] = 0.0


def eligible(shares: float, price: float, cash: float) -> bool:
    """Checks whether the order can be placed.

    When trying to long a position, we need enough cash. When trying to short,
    we also need to make sure that we have enough cash for it.

    TODO: This doesn't check open positions, and so all open positions must be
     closed out after each period (fine when ignoring any transaction fees).

    Args:
      name (str): The reference name of the universe file to be loaded.

    Returns:
      pd.DataFrame: The universe of securities.
    """
    return abs(shares * price) <= cash


# @dataclass
# class Order:
#     ticker: str
#     shares: float

#     def eligible(self, price: float, cash: float) -> bool:
#         """Checks whether the order can be placed.

#         When trying to long a position, we need enough cash. When trying to short,
#         we also need to make sure that we have enough cash for it.

#         TODO: This doesn't check open positions, and so all open positions must be
#          closed out after each period (fine when ignoring any transaction fees).

#         Args:
#           name (str): The reference name of the universe file to be loaded.

#         Returns:
#           pd.DataFrame: The universe of securities.
#         """
#         return abs(self.shares * price) <= cash


class StrategyException(Exception):
    pass
