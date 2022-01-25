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
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
modelname='KNN'
x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pd.DataFrame(x_valid_scaled)
params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)
model.fit(x_train,y_train)
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print(f'{modelname} rms {rms:.2f}')
valid = valid.copy()
valid['Predictions'] = preds
plt.figure(figsize=(10,8),dpi=100)
plt.title(f'Model {modelname} : {symbol}')
plt.plot(valid[['Close', 'Predictions']])
plt.plot(train['Close'])