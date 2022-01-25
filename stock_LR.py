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
new_data['Date'] = (new_data['Date'] - new_data['Date'][0]).dt.days
devide = round(len(new_data)*2/3)
train = new_data[:devide]
valid = new_data[devide:]
x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
modelname='LR'
modelname='Linear Regression'
preds = model.predict(x_valid)
rms = np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print(f'{modelname} rms {rms:.2f}')
valid = valid.copy()
valid['Predictions'] = preds.tolist()
plt.figure(figsize=(10,8),dpi=100)
plt.title(f'Model {modelname} : {symbol}')
plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['Close'])