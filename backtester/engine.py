"""Backtesting engine. 

This module will have step by step actions that are performed when running the backtesting algorithm.

There will be a 
"""

from typing import Iterable
from datetime import datetime
from dataclasses import dataclass
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
        self.cash = initial_amount
        self.strategy = user_strategy
        self.context = context
        self.pos_id = 0
        self.history = pd.DataFrame(
            columns=["symbol", "open_price", "shares", "close_price", "PnL"]
        )
        # Vector tickers, with orders and trades in order of shown tickers
        self.tickers = [ticker for ticker in self.data["Adj Close"].columns]
        self.trades = []
        self.orders = []

    def run(self):
        """Execute the backtester.

        The algorithm cannot be vectorised in time but can be across assets

        At each timestep we are doing actions for i-1-th closing and for i-th day opening.
        In practice, these actions would occur at different times but for backtesting it is
        sufficient for them to occur together.
        """
        for i in range(len(self.data)):

            # Close-out previous positions and adjust cash
            self._close_positions(i)
            # Use closing price to create orders of day i-1
            self._place_orders(i)
            # Use opening price of day i to open trades
            self._place_trades(i)

        # Model is now complete, run a post-processer

    def _close_positions(self, i) -> None:
        # Close position and add to history
        for trade in self.trades:
            close = self.data["Adj Close"].loc[i - 1, trade.ticker]
            pnl = (close - trade.price) * trade.shares
            # Adjust cash by value of trade with and PnL generated
            self.cash += abs(trade.value) + pnl
            self.history.loc[trade.id, ["close_price", "PnL"]] = [close, pnl]
        # Closing out
        self.trades = []

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
        if i == 0:
            return

        # Can't place orders on first data entry point because we do not have previous day's close
        alphas = self.strategy(self.data.loc[: i - 1, :], self.context)

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
        available_cash = self.cash

        # The total value of each position in the universe
        value = (alphas / total) * available_cash
        # Number of shares is then the value / closing price of the previous day.
        nshares = value / self.data.loc[i - 1, "Adj Close"]
        # Create the orders
        self.orders = [Order(tick, share) for tick, share in zip(self.tickers, nshares)]

    def _place_trades(self, i: int) -> None:
        """Goes through all orders and attempts to place the trade using the open price
        for the trading day.

        An eligibility check is performed to ensure that there is sufficient capital to
        place a trade. If there are numerous overnight changes in stock price, some
        trades may not be placed.

        TODO: Add a feature for fixed to approximate remaining capital so that the user
        can fulfill all trades if there must be fixed remaining cash in the trading
        account, but there is some leeway.

        Args:
            i (int): The current time period index.
        """
        # Convert orders at previous day's closing to trades using todays open price
        for order in self.orders:
            open_price = self.data["Open"].loc[i, order.ticker]
            if order.eligible(open_price, self.cash):
                # Place the trade and reduce available cash in the account
                t = Trade(self.pos_id, order.ticker, open_price, order.shares)
                self.trades += [t]
                self.cash -= abs(t.value)
                # Add open trade to trade history
                self.history.loc[self.pos_id] = [
                    t.ticker,
                    t.price,
                    t.shares,
                    None,
                    None,
                ]
                # increment trade counter
                self.pos_id += 1


@dataclass
class Trade:
    id: int
    ticker: str
    price: float
    shares: float

    @property
    def value(self):
        return self.price * self.shares


@dataclass
class Order:
    ticker: str
    shares: float

    def eligible(self, price: float, cash: float) -> bool:
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
        return abs(self.shares * price) <= cash


class StrategyException(Exception):
    pass
