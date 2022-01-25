import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stockMatrix as sm
import stockDB as sd
import pywt
import os
CURRENT_DIR = os. getcwd() 
pd.options.display.max_columns= 999
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
import sys
da = sd.StockDB()
symbol = 'AMZN'
stockData = da.getStockPrice(symbol)
days = 400
stockData = stockData[-days:]
print('Wavelet Family：',pywt.families())
print('Func of Daubechies(db):',pywt.wavelist(family='db',kind='all'))
filters_length = 8
dwt = sm.DaubechiesWavelet(stockData, filters_length) 
print(dwt.head(10))
plt.figure(figsize=(10,7),dpi=100)  
plt.plot(stockData.index[:days], stockData['Close'][:days],color='blue')
plt.title('Close Price')
plt.plot(dwt.index[:days],dwt['DB'+ str(round(filters_length/2))][:days],color='red')
plt.title('(Part)After DB'+ str(round(filters_length/2)))
rms = np.sqrt(np.mean(np.power((np.array( dwt['DB'+ str(round(filters_length/2))])-np.array(stockData['Close'])),2)))
print('Daubechies rms {rms:.2f}')
import statsmodels.api as smapi
modelname = 'ARIMA(p,d,q)'
Arima = smapi.tsa.ARIMA(np.array(dwt['DB'+ str(round(filters_length/2))]), order=(2,1,0))
results  = Arima.fit()
print(results.summary())
print('original data:\t\t\t',list(dwt['DB'+ str(round(filters_length/2))][-3:]))
predict_stock = results.predict(3)[-3:]
print('in-sample predictions:\t\t',predict_stock)
forecast_stock = results.forecast(3)  
print('out-of-sample forecast:\t\t',forecast_stock)
plt.savefig('DaubechiesWavelet.png',bbox_inches='tight')
plt.show()
rms = np.sqrt(np.mean(np.power((np.array( dwt['DB'+ str(round(filters_length/2))])-np.array(stockData['Close'])),2)))
print(f'{modelname} rms {rms:.2f}')
sys.exit()