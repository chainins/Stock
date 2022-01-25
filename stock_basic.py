import matplotlib.pyplot as plt
import pandas as pd
import stockMatrix as sm
import stockDB as sd
import os
CURRENT_DIR = os. getcwd() 
pd.options.display.max_columns= 999
# plt.rcParams['font.family'] = 'DeJavu Sans'
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # Calibri
da = sd.StockDB()
symbol = 'AMZN'
stockData = da.getStockPrice(symbol)
sm.drawSMA(sm.getCommonSMA(stockData[-100:]))
sm.drawMACD(sm.getMACD(stockData[-100:]))
sm.drawBollinger(stockData[-100:])
sm.drawKDJ(stockData[-200:],'day')
yd = sm.getYearData(stockData)
sm.drawKDJ(yd,'Year')
md = sm.getMonthData(stockData[-600:])
sm.drawKDJ(md,'Month')
wd = sm.getWeekData(stockData[-600:])
sm.drawKDJ(wd,'Week')
sm.drawKDJ(stockData[-300:],'Day')
sm.drawCrocodile(sm.getCrocodile(sm.getCommonSMA(stockData[-100:])))
sm.drawFft(stockData[:500],16)
data = sm.getGoalData(stockData,80,0.1) 
print(data.head(10))