
import plotly
import plotly.plotly as pltly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot

from datetime import datetime
import pandas_datareader as web


df = web.DataReader("aapl", 'yahoo',
                    datetime(2015, 1, 1),
                    datetime(2016, 7, 1))

print(df.head())
data = [go.Scatter(x=df.index, y=df.High)]

aplot = iplot(data)
plotly.offline.plot(aplot, filename='test_plotly.html')

