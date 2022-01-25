import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stockMatrix as sm
import stockDB as sd
import os
import matplotlib.font_manager
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
num_valid = valid.shape[0]
preds = []
a = train['Close'][-num_valid:]
for i in range(0,num_valid):
    c = a[-num_valid:].mean()
    a = a.append(pd. Series([c]))
preds = list(a[-num_valid:])
modelname='MA'
rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
print(f'{modelname} rms {rms:.2f}')
valid = valid.copy()
valid['Predictions'] = preds
plt.figure(figsize=(10,8),dpi=100)
plt.title(f'Model {modelname} : {symbol}')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])