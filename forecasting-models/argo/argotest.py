# importing R objects and libraries to Python
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import pandas as pd
# convert python data to r data
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
numpy2ri.activate()
pandas2ri.activate()


# ts = robjects.r('ts')
# xts = robjects.r('xts')
# xts = importr('xts')
# forecast = importr('forecast')
argo = importr('argo')
# xts = importr('xts')
xts = importr("xts", robject_translations={".subset.xts": "_subset_xts2",
                                           "to.period": "to_period2"})

filepath = "../../data/time_series/gsr_syria.csv"
ts_syria = pd.read_csv(filepath, header=0, parse_dates=True, index_col=0,
                       squeeze=True)

packageNames = ['argo', 'forcast']
print(rpackages.isinstalled('argo'))
print(rpackages.isinstalled('forecast'))

ts_train = ts_syria[:"2016-12-31"]
ts_test = ts_syria[:"2017-01-01"]

# print(syriats.head())
print(ts_syria.shape)
print(ts_train.shape)
print(ts_test.shape)

from model.argo.converion import convert_df_to_xts


# rdata_train = ts(ts_train.values)
# rdata_train = xts.xts(ts_train.values, ts_train.index.values)
# print(rdata)
# print(type(rdata))
# print(len(rdata))


# apply auto arima
# model = forecast.auto_arima(rdata_train)
# model_res = forecast.forecast(model, h=16, level=95.0)
# print(type(model_res))
# print(model_res.names)

# model = argo.argo(ts_train.values)
#
# print(type(model))
