from pandas import DataFrame, read_csv
# Read some DataFrame with TimeSeries data (I take the AAPL stocks)
X = read_csv('http://www.quandl.com/api/v1/datasets/GOOG/NASDAQ_AAPL.csv?&trim_start=1981-03-11&trim_end=2013-10-28&sort_order=desc', parse_dates=True)
 
# function to load pandas DataFrame into R
# written by Wes McKinney
def pandas_to_r(df, name):
    from rpy2.robjects import r, globalenv
    r_df = rpy.convert_to_r_dataframe(df)
    globalenv[name] = r_df
 
# Send it to R
pandas_to_r(X, "df")
 
# convert to date
%%R
library(xts)
df$Date <- as.Date(as.character(df$Date))
df$Close <- as.numeric(df$Close)
df.xts <- xts(df$Close, order.by=df$Date)
