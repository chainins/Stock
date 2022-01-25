import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stockMatrix as sm
import stockDB as sd
import os
CURRENT_DIR = os. getcwd() 
pd.options.display.max_columns= 999
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
da = sd.StockDB()
symbol = 'AMZN'
stockData = da.getStockPrice(symbol)
data = stockData
data = data.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(data)),columns=['Date', 'Close'])
new_data['Date'] = data['Date']
new_data['Close'] = data['Close']
devide = round(len(new_data)*2/3)
train = new_data[:devide]
valid = new_data[devide:]
x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']
modelname='Prophet(FB)' 
from prophet import Prophet
new_data = pd.DataFrame(index=range(0,len(stockData)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
new_data['Date'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
new_data.index = new_data['Date']
new_data.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
devide = round(len(new_data)*2/3)
train = new_data[:devide]
valid = new_data[devide:]
model = Prophet(daily_seasonality=True)
model.fit(train)
close_prices = model.make_future_dataframe(periods=len(valid))
forecast = model.predict(close_prices)
forecast_valid = forecast['yhat'][devide:]
rms=np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)),2)))
print(f'{modelname} rms {rms:.2f}')
valid = valid[valid['ds'].notnull()].copy()
valid['Predictions'] = forecast_valid.values
plt.figure(figsize=(10,8),dpi=100)
plt.title(f'Model {modelname} : {symbol}')
plt.plot(train['y'])
plt.plot(valid[['y', 'Predictions']])
plt.savefig('Prophet.png',bbox_inches='tight')
plt.show()