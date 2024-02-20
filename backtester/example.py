from datetime import datetime
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


def mystrat(data, context):
    """
    Purpose: one
    Must return an iterable of weightings or "alphas", base on the data.
    Currently uses the adjusted close but the method would be up to the user.
    """
    print("hi")
    print(context)
    closing = data["Adj Close"].T
    znormed = (closing - closing.mean()) / closing.std()
    return znormed[0].values
    # return alphas


model = Engine(1000, data, mystrat, context=1)
model.run()
