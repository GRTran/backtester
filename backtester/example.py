import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from engine import Engine
import datastreams


univ = datastreams.load_universe(datastreams.Universe.SANDP500)
# data = datastreams.yf_timeseries(
#     univ["symbol"],
#     start=datetime(2021, 1, 1),
#     end=datetime(2022, 1, 1),
#     to_file=True,
#     filename="1yr_snp.csv",
# )
data = datastreams.csv_timeseries(filename="1yr_snp.csv")


def strat2(data, context):
    """
    Purpose: one
    Must return an iterable of weightings or "alphas", base on the data.
    Currently uses the adjusted close but the method would be up to the user.
    """
    closing = data["Adj Close"].T
    return np.array([random.uniform(-1, 1) for _ in range(len(closing))])


model = Engine(1000, data, strat2, context=1)
model.run()
plt.plot(model.data.index, model.cash)
plt.savefig("result.png")
