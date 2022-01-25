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
modelname='LSTM' 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM 
data = stockData.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(stockData)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
dataset = new_data.values
devide = round(len(new_data)*2/3)
train = dataset[0:devide,:]
valid = dataset[devide:,:]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print(f'{modelname} rms {rms:.2f}')
train = new_data[:devide]
valid = new_data[devide:]
valid = valid.copy()
valid['Predictions'] = closing_price
plt.figure(figsize=(10,8),dpi=100)
plt.title(f'Model {modelname} : {symbol}')
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])

plt.savefig('LSTM.png',bbox_inches='tight')
plt.show()