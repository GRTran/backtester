"""
Functional approach to creating the data stream?
  - Create a function that reads a data stream of open/close/high/low/volume and other data into a dataframe (could be slow?)
  - Creates a wrapper on the yfinance module to read in the data we want
  - Function with a decorator that applies it to the trading algorithm, think this is a good method
  
  - Use the following data structures:
    - pandas
    - numpy
    - numba
    - hashmap
    - decorators
    - generators
    - parallelisation (scaling multi-strats)
"""

from collections.abc import Iterable
from enum import Enum
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf


class Universe(str, Enum):
    """The eligible universes and their csv file locations"""

    SANDP500 = "backtester/sandp500.csv"
    # NASDAQ = "None"
    # DOW = None


def yf_timeseries(
    ticks: str | Iterable[str],
    start: datetime | None = None,
    end: datetime | None = None,
    interval: str = "1d",
    to_file: bool = False,
    filename: str | None = None,
) -> pd.DataFrame:
    """Load the timeseries information that will be used for backtesting into a
     ``pd.DataFrame``.

    The data includes:
      - Dates
      - Open
      - Close
      - High
      - Low
      - Volume
      - Dividends
      - Stock-splits

    The data shown is in the frequency specified by the user, but is defaulted to daily
     ticker information.

    TODO: The function also loads in the information for the ticker.


    Args:
      ticks (str | Iterable[str]): The ticker(s) as strings.

    Returns:
      pd.DataFrame: The actual data
    """
    # Get the ticker timeseries data
    try:
        histories = _get_history(ticks, start, end, interval)
    except ValueError as err:
        return err

    # Save the time-series to file if user requested
    if to_file:
        try:
            histories.to_csv(filename)
        except IOError as err:
            return err

    # Get the history in required interval
    return histories


def csv_timeseries(filename: str) -> pd.DataFrame:
    """Load the timeseries information that will be used for backtesting into a
     ``pd.DataFrame`` from a csv file.

    Args:
      ticks (str | Iterable[str]): The ticker(s) as strings.

    Returns:
      pd.DataFrame: The actual data
    """
    try:
        data = pd.read_csv(filename, header=[0, 1], index_col=0)
        data.reset_index(inplace=True)
        return data
    except IOError as err:
        return err


def _get_history(
    ticks: str | list[str],
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Get the ticker histories depending on user spec"""
    if start is None and end is None:
        return yf.download(
            " ".join(ticks), period="max", interval=interval, threads=True
        )
    elif start is not None and end is None:
        return yf.download(
            " ".join(ticks),
            start=start,
            end=datetime.now(),
            interval=interval,
            threads=True,
        )
    elif start is None and end is not None:
        return yf.download(
            " ".join(ticks),
            start="1900-01,01",
            end=end,
            interval=interval,
            threads=True,
        )

    return yf.download(
        " ".join(ticks), start=start, end=end, interval=interval, threads=True
    )


def load_universe(name: str) -> pd.DataFrame:
    """Loads in the universe in the a ``pd.DataFrame``.

    Args:
        name (str): The reference name of the universe file to be loaded.

    Returns:
        pd.DataFrame: The universe of securities.
    """
    return pd.read_csv(name.value)


# ---------------------------------------------------------------------------- #
#                     Executed when testing the datastream                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # df = load_timeseries("msft", "2023-01-01", "2024-02-01")

    # 1) Select a universe of equities to analyse
    u = load_universe(Universe.SANDP500)

    # 2) Using the names, load the time series
    df = load_timeseries(u["symbols"])
    pass
