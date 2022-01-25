import numpy as np
import numpy.fft as nf
import pylab as pl
from numpy import fft
from datetime import datetime
import pandas as pd
import stockDB as sd
import matplotlib
import matplotlib.pyplot as plt
import pywt
import statsmodels.api as smapi
import statsmodels.tsa.stattools as smts
def DaubechiesWavelet(stockDf,filters_length = 8):   
    if len(stockDf)<filters_length*15:
        print('The level is too high for the length of data, please choose a smaller one.')
        return        
    stockData = pd.DataFrame(stockDf, columns=['Adj Close'])
    stockData.rename(columns={'Adj Close':'Close'},inplace=True)
    result = pywt.wavedec(np.array(stockData['Close']),wavelet='db'+ str(round(filters_length/2)),mode='symmetric',level=round(filters_length/2))    
    result_df = pd.DataFrame([result[0],result[1],result[2],result[3],result[4]],index=['main trend','subwave1','subwave2','subwave3','subwave4']).T
    result_df.head()   
    plt.figure(figsize=(10,7),dpi=100)
    subnum= int((np.ceil(np.sqrt(round(filters_length/2)))))
    for j in list(range(1,round(filters_length/2)+1)): 
        plt.subplot(subnum, subnum,j)
        plt.plot(range(len(result[j])),result[j])
        plt.title('Subwave '+str(j))
    plt.show()
    for m in list(range(round(filters_length/4)+1,round(filters_length/2)+1)): 
        signalbak = np.array(result[m])
        signal = abs(signalbak)
        signal.sort()
        signal = signal**2
        list_risk_j = []
        N = len(signal)
        for j in range(N):
            if j == 0:
                risk_j = 1 + signal[N-1]
            else:
                risk_j = (N-2*j + (N-j)*(signal[N-j]) + sum(signal[:j]))/N
            list_risk_j.append(risk_j)
        k = np.array(list_risk_j).argmin()
        threshold = np.sqrt(signal[k])       
        result[m] = pywt.threshold(signalbak, threshold, 'soft')
    dwt = pywt.waverec(result, 'db'+ str(round(filters_length/2)))[:len(stockData)] 
    stockData['DB'+ str(round(filters_length/2))] = dwt
    plt.figure(figsize=(10,7),dpi=100)  
    plt.subplot(2,1,1)
    plt.plot(stockData.index, stockData['Close'],color='blue')
    plt.title('Close Price')
    plt.subplot(2,1,2)
    plt.plot(stockData.index,stockData['DB'+ str(round(filters_length/2))],color='red')
    plt.title('After DB'+ str(round(filters_length/2)))
    fig = plt.figure(figsize=(10,10),dpi=100)
    ax1 = fig.add_subplot(111)
    smapi.graphics.tsa.plot_acf(dwt,lags=40,ax=ax1) 
    fig2 = plt.figure(figsize=(10,10),dpi=100)
    ax2 = fig2.add_subplot(111)
    smapi.graphics.tsa.plot_pacf(dwt,lags=40,ax=ax2) 
    plt.show()
    stockData_DIFF = stockData['DB'+ str(round(filters_length/2))].diff(4) 
    dftest = smts.adfuller(stockData_DIFF[4:],autolag='AIC')   
    dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        identifier = f'Critical Value {key}'
        dfoutput[identifier] = value
    print('\nAugmented Dickey-Fuller Test:\n')
    print(dfoutput)
    stockDf = stockDf[stockDf['Adj Close'].notnull()]
    stockDf['DB'+ str(round(filters_length/2))] = dwt
    return stockDf
def FourierFilter(stockDf):
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.rcParams['axes.unicode_minus']=False 
    stockDf = stockDf[stockDf['Date'].notnull()]
    stockDf.reset_index(inplace=True)
    stockDf = stockDf.drop(['index'], axis=1)
    t = np.linspace(1,stockDf.shape[0],stockDf.shape[0])
    y = np.array(stockDf['Adj Close'])
    N = len(t)
    fft_y= nf.fft(y) 
    df_fft = pd.DataFrame(data=fft_y)
    df_fft.columns=['fft_value']
    df_fft['amplitude'] = df_fft['fft_value'].abs()
    df_fft['scaled_amplitude']=  df_fft['amplitude']/N*2
    df_fft.loc[0,'scaled_amplitude'] = df_fft['scaled_amplitude'][0]/2
    df_fft['angle'] = np.angle(fft_y)
    df_fft['freq'] = np.arange(N)/(2*np.pi)
    return df_fft
def drawFft(stockDf, tailNum = 8):
    t = np.linspace(1,stockDf.shape[0],stockDf.shape[0])
    y = np.array(stockDf['Adj Close'])
    N = len(t)
    df_fft = FourierFilter(stockDf)
    df_fft_harlf = df_fft.loc[np.arange(int(N/2)),:]
    df_fft_harlf[df_fft_harlf['scaled_amplitude']>0.5]['angle']/np.pi*180
    f = nf.fftfreq(t.size)
    large = df_fft
    large['sample_freq'] = f
    large = large.loc[np.arange(int(N/2)),:]
    large = large.sort_values(by='scaled_amplitude', ascending=False).head(tailNum)
    large = large[:tailNum]
    plt.figure(figsize=(10, 7), dpi=100)
    plt.plot(t,y,'b', linewidth=2.5 ,label='Original')
    zeroFreq = df_fft_harlf[(df_fft_harlf['freq']==0)]
    A0 = zeroFreq.loc[0,'scaled_amplitude']
    consY = t.copy()
    consY.fill(A0)
    plt.plot(t,consY,'g',label = 'constant')
    simY = t.copy()
    simY.fill(0)
    if len(stockDf) < 30:        
        xticks_index = stockDf.index 
        xticks_date = stockDf['Date'].dt.date
    else:
        num = len(stockDf) // 30
        xticks_index = stockDf.index [stockDf.index%num==0]
        xticks_date = stockDf['Date'][stockDf.index%num==0].dt.date
    plt.xticks(xticks_index,xticks_date)
    plt.setp(plt.gca().get_xticklabels(), rotation=90)
    i = 0
    for index,row in large.iterrows():
        amplitude = row['scaled_amplitude'].real
        phase = row['angle'].real
        i = i + 1
        y = amplitude*np.cos(2 * np.pi * row['sample_freq'] * t + phase)
        simY = simY + y
        plt.plot(t,y.real,label = 'Subwave'+ str(i))
    plt.plot(t,simY.real,'r', linewidth=2.5,label = 'Fourier Filter(fft)')
    plt.title( 'Decompose by Fourtier Transfer(fft)')
    plt.ylabel("Close Price")
    plt.legend()
    plt.savefig('Fourtier(fft).png',bbox_inches='tight')
    plt.show()    
    return df_fft
def getAmount(df):
    df = df.copy()
    df['Amount'] = df['Volume']*df['Adj Close']
    return df
def getWeekData(df):
    df = df.copy()
    df['Amount'] = df['Volume']*df['Adj Close']
    df['year'] = df['Day'].dt.isocalendar().year
    df['week'] = df['Day'].dt.isocalendar().week
    df['adjHigh'] = df['High']/df['Close'] * df['Adj Close']
    df['adjLow'] = df['Low']/df['Close'] * df['Adj Close']
    by_weekDf = [ df.year, df.week]
    weekDf = pd.DataFrame()
    weekDf = df.groupby(by_weekDf)['adjLow'].agg(['min'])
    weekDf.rename(columns={'min':'Low'},inplace=True)
    weekDf['High'] = df.groupby(by_weekDf)['adjHigh'].agg(['max'])
    weekDf['Close'] = df.groupby(by_weekDf)['Adj Close'].agg(['last'])
    weekDf['WeekStart'] = df.groupby(by_weekDf)['Date'].agg(['min'])
    weekDf['Date'] = weekDf['WeekStart'] 
    weekDf['WeekEnd'] = df.groupby(by_weekDf)['Date'].agg(['max'])
    weekDf['NumDay'] = df.groupby(by_weekDf)['Date'].agg(['count'])
    weekDf['AmountUnadjust'] = df.groupby(by_weekDf)['Amount'].agg(['sum'])
    weekDf['Amount'] = weekDf['AmountUnadjust'] * 5 / weekDf['NumDay']
    weekDf.reset_index(inplace=True)
    return weekDf
def getMonthData(df):
    df = df.copy()
    df['Amount'] = df['Volume']*df['Adj Close']
    df['year'] = df['Day'].dt.isocalendar().year
    df['month'] = df['Day'].dt.month
    df['adjHigh'] = df['High']/df['Close'] * df['Adj Close']
    df['adjLow'] = df['Low']/df['Close'] * df['Adj Close']
    by_monthDf = [ df.year, df.month]
    monthDf = pd.DataFrame()
    monthDf = df.groupby(by_monthDf)['adjLow'].agg(['min'])
    monthDf.rename(columns={'min':'Low'},inplace=True)
    monthDf['High'] = df.groupby(by_monthDf)['adjHigh'].agg(['max'])
    monthDf['MonthStart'] = df.groupby(by_monthDf)['Date'].agg(['min'])
    monthDf['Date'] = monthDf['MonthStart'] 
    monthDf['MonthEnd'] = df.groupby(by_monthDf)['Date'].agg(['max'])
    monthDf['Close'] = df.groupby(by_monthDf)['Adj Close'].agg(['last'])
    monthDf['NumDay'] = df.groupby(by_monthDf)['Date'].agg(['count'])
    monthDf['AmountUnadjust'] = df.groupby(by_monthDf)['Amount'].agg(['sum'])
    monthDf['Amount'] = monthDf['AmountUnadjust'] * 5 / monthDf['NumDay']
    monthDf.reset_index(inplace=True)
    return monthDf
def getYearData(df):
    df = df[df['Day'].notnull()].copy()
    df['Amount'] = df['Volume']*df['Adj Close']
    df['year'] = df['Day'].dt.isocalendar().year
    df['adjHigh'] = df['High']/df['Close'] * df['Adj Close']
    df['adjLow'] = df['Low']/df['Close'] * df['Adj Close']
    by_yearDf = [ df.year]
    yearDf = pd.DataFrame()
    yearDf = df.groupby(by_yearDf)['adjLow'].agg(['min'])
    yearDf.rename(columns={'min':'Low'},inplace=True)
    yearDf['High'] = df.groupby(by_yearDf)['adjHigh'].agg(['max'])
    yearDf['YearStart'] = df.groupby(by_yearDf)['Date'].agg(['min'])
    yearDf['Date'] = yearDf['YearStart'] 
    yearDf['YearEnd'] = df.groupby(by_yearDf)['Date'].agg(['max'])
    yearDf['Close'] = df.groupby(by_yearDf)['Adj Close'].agg(['last'])
    yearDf['NumDay'] = df.groupby(by_yearDf)['Date'].agg(['count'])
    yearDf['AmountUnadjust'] = df.groupby(by_yearDf)['Amount'].agg(['sum'])
    yearDf['Amount'] = yearDf['AmountUnadjust'] * 5 / yearDf['NumDay']
    yearDf.reset_index(inplace=True)
    return yearDf
def getKDJ(df,period=9):
    df.reset_index(inplace=True)
    df = df[df['Low'].notnull() & df['High'].notnull() & df['Close'].notnull()].copy()
    df['MinLow'] = df['Low'].rolling(period, min_periods=period).min()
    df['MinLow'].fillna(value = df['Low'].expanding().min(), inplace = True)
    df['MaxHigh'] = df['High'].rolling(period, min_periods=period).max()
    df['MaxHigh'].fillna(value = df['High'].expanding().max(), inplace = True)
    df['RSV'] = (df['Close'] - df['MinLow']) / (df['MaxHigh'] - df['MinLow']) * 100
    for i in range(len(df)):
        if i==0:     
            df.loc[i,'K']=50
            df.loc[i,'D']=50
        if i>0:
            df.loc[i,'K']=df.loc[i-1,'K']*2/3 + 1/3*df.loc[i,'RSV']
            df.loc[i,'D']=df.loc[i-1,'D']*2/3 + 1/3*df.loc[i,'K']
        df.loc[i,'J']=3*df.loc[i,'K']-2*df.loc[i,'D']
    return df
def drawKDJ(df,note='',period=9):
    stockDataFrame = getKDJ(df,period)
    plt.figure(figsize=(10,2),dpi=100)
    stockDataFrame['K'].plot(color="c",label='K')
    stockDataFrame['D'].plot(color="b",label='D')
    stockDataFrame['J'].plot(color="red",label='J')
    plt.legend(loc='best')#Draw legend
    if len(stockDataFrame) < 30:        
        xticks_index = stockDataFrame.index 
        xticks_date = stockDataFrame['Date'].dt.date
    else:
        num = len(stockDataFrame) // 30
        xticks_index = stockDataFrame.index [stockDataFrame.index%num==0]
        xticks_date = stockDataFrame['Date'][stockDataFrame.index%num==0].dt.date
    plt.xticks(xticks_index,xticks_date)
    plt.setp(plt.gca().get_xticklabels(), rotation=90)
    plt.grid(linestyle='-.')
    plt.title(f"KDJ Index of Stock Price ({note})")
    plt.rcParams['font.sans-serif']=['Apercu']
    plt.savefig('KDJ_{note}.png',bbox_inches='tight')
    plt.show()
def getEMA(df, term):
    df = df.copy()
    df.reset_index(inplace=True)
    df['Close'] = df['Adj Close']
    df.loc[0,'EMA'] = df.loc[0,'Close']
    for i in range(len(df)-1):
        df.loc[i+1,'EMA']=(term-1)/(term+1)*df.loc[i,'EMA']+2/(term+1) * df.loc[i+1,'Close']
    EMAList=list(df['EMA'])
    return EMAList
def getMACD(df, shortTerm=12, longTerm=26, DIFTerm=9):
    df = df.copy()
    df.reset_index(inplace=True)
    shortEMA = getEMA(df, shortTerm)
    longEMA = getEMA(df, longTerm)
    df['DIF'] = pd.Series(shortEMA) - pd.Series(longEMA)
    df.loc[0,'DEA'] = df.loc[0,'DIF']  
    for i in range(len(df)-1):
        df.loc[i+1,'DEA'] = (DIFTerm-1)/(DIFTerm+1)*df.loc[i,'DEA'] + 2/(DIFTerm+1)*df.loc[i+1,'DIF']
    df['MACD'] = 2*(df['DIF'] - df['DEA'])
    return df[['Date','DIF','DEA','MACD']]
def drawMACD(stockDataFrame):
    stockDataFrame = stockDataFrame[stockDataFrame['Date'].notnull()].copy()
    stockDataFrame.reset_index(inplace=True)
    stockDataFrame = stockDataFrame.drop(['index'], axis=1)
    fig = plt.figure(figsize=(10,2),dpi=100)
    ax1 = fig.add_subplot(111)
    if len(stockDataFrame) < 30:        
        xticks_index = stockDataFrame.index 
        xticks_date = stockDataFrame['Date'].dt.date
    else:
        num = len(stockDataFrame) // 30
        xticks_index = stockDataFrame.index [stockDataFrame.index%num==0]
        xticks_date = stockDataFrame['Date'][stockDataFrame.index%num==0].dt.date
    plt.xticks(xticks_index,xticks_date)
    plt.setp(plt.gca().get_xticklabels(), rotation=90)
    ax1.grid(linestyle='-.')
    plt.title("Figure MACD/DEA/DIF")
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif']=['Apercu']
    ax1.plot(stockDataFrame.index, stockDataFrame['DEA'],color='b',label = 'DEA')
    ax1.plot(stockDataFrame.index, stockDataFrame['DIF'],color='g',label = 'DIF')
    ax1.legend(prop = {'family':'Helvatica','size':8},loc='lower left')
    ax2 = ax1.twinx()
    ax2.bar(stockDataFrame.index, stockDataFrame['MACD'],width=0.8, color=['red' if i <0 else  'green' for i in  stockDataFrame['MACD']],label = 'MACD')
    ax2.legend(prop = {'family':'Helvatica','size':8})
    plt.savefig('MACD.png',bbox_inches='tight')
    plt.show()
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
def drawBollinger(stockDf, period = 20):
    stockDf = stockDf[stockDf['Date'].notnull()].copy()
    stockDf.reset_index(inplace=True)
    stockDf = stockDf.drop(['index'], axis=1)
    stockDf['Close'] = stockDf['Adj Close']
    fig = plt.figure(figsize=(10,2),dpi=100)
    ax = fig.add_subplot(111)
    stockDf = stockDf[["Date","Open","High","Low","Close"]]
    stockDf['Date'] = pd.to_datetime(stockDf['Date'])
    stockDf['Date'] = stockDf['Date'].map(mpdates.date2num)
    stockDf['DateBak'] = stockDf['Date']
    stockDf['Date'] = stockDf.index
    candlestick_ohlc(ax = ax, quotes= stockDf.values, width=0.75,  colorup='green', colordown='red')
    stockDf['Date'] = stockDf['DateBak'].map(mpdates.num2date)
    stockDf = stockDf.drop(['DateBak'], axis=1)
    if len(stockDf) < 30:        
        xticks_index = stockDf.index 
        xticks_date = stockDf['Date'].dt.date
    else:
        num = len(stockDf) // 30
        xticks_index = stockDf.index [stockDf.index%num==0]
        xticks_date = stockDf['Date'][stockDf.index%num==0].dt.date
    plt.xticks(xticks_index,xticks_date)
    plt.setp(plt.gca().get_xticklabels(), rotation=90)   
    stockDf['mid'] = stockDf['Close'].rolling(window=period).mean()
    stockDf['std'] = stockDf['Close'].rolling(window=period).std() 
    stockDf['up'] = stockDf['mid'] + 2*stockDf['std']
    stockDf['down'] = stockDf['mid'] - 2*stockDf['std']
    ax.plot(stockDf.index, stockDf['up'],color='g',label = 'Bollinger Up Line')
    ax.plot(stockDf.index, stockDf['down'],color='b',label = 'Bollinger Down Line')
    ax.plot(stockDf.index, stockDf['mid'],color='r',label = 'Bollinger Mediam Line')
    ax.set_ylabel("Close Price")
    ax.grid () 
    ax.legend (framealpha=0.7) 
    plt.rcParams['font.sans-serif']=['Apercu']
    plt.title("20 days Bollinger Bands")
    plt.savefig('Bollinger.png',bbox_inches='tight')
    plt.show()
def getSMA(df, period):
    MAname = 'MA'+  str(period)
    SMAname = 'S' + MAname
    df['Close'] = df['Adj Close']
    df[MAname] = df['Close'].rolling(window=period).mean()
    for i in range(len(df)):
        if i<period:
            df[SMAname] = df.loc[i,MAname]
        else: 
           df.loc[i,SMAname]=df.loc[i,MAname] + (df.loc[i,'Close'] - df.loc[i-1,SMAname])/period
    return df
def getCommonSMA(df):
    df = df.copy()
    df.reset_index(inplace=True)
    df = getSMA(df, 5)
    df = getSMA(df, 8)
    df = getSMA(df, 13)
    return df
def drawSMA(stockDataFrame):    
    stockDataFrame = stockDataFrame.copy()
    stockDataFrame.reset_index(inplace=True)
    stockDataFrame = stockDataFrame.drop(['index'], axis=1)
    stockDataFrame['Close'] = stockDataFrame['Adj Close']
    fig = plt.figure(figsize=(10,2),dpi=100)
    ax = fig.add_subplot(111)
    if len(stockDataFrame) < 30:        
        xticks_index = stockDataFrame.index 
        xticks_date = stockDataFrame['Date'].dt.date
    else:
        num = len(stockDataFrame) // 30
        xticks_index = stockDataFrame.index [stockDataFrame.index%num==0]
        xticks_date = stockDataFrame['Date'][stockDataFrame.index%num==0].dt.date
    plt.xticks(xticks_index,xticks_date)
    plt.setp(plt.gca().get_xticklabels(), rotation=90)
    ax.set_ylabel("Close Price")
    ax.grid(linestyle='-.')
    plt.title("Figure SMA5/SMA8/SMA13")
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif']=['Apercu']
    ax.plot(stockDataFrame.index, stockDataFrame['Close'],linewidth=2.5,color='r',label = 'Close')
    ax.plot(stockDataFrame.index, stockDataFrame['SMA5'],color='b',label = 'SMA5')
    ax.plot(stockDataFrame.index, stockDataFrame['SMA8'],color='g',label = 'SMA8')
    ax.plot(stockDataFrame.index, stockDataFrame['SMA13'],color='c',label = 'SMA13')
    ax.legend(prop = {'family':'Helvatica','size':8},loc='lower left')
    plt.savefig('SMA.png',bbox_inches='tight')
    plt.show()
def getCrocodile(df):
    df['up'] = df['SMA5'].shift(3)
    df['mid'] = df['SMA8'].shift(5)
    df['down'] = df['SMA13'].shift(8)
    return df
def drawCrocodile(stockDf):
    stockDf = stockDf[stockDf['Date'].notnull()].copy()
    stockDf.reset_index(inplace=True)
    stockDf = stockDf.drop(['index'], axis=1)
    stockDf['Close'] = stockDf['Adj Close']
    fig = plt.figure(figsize=(10,2),dpi=100)
    ax = fig.add_subplot(111)
    if len(stockDf) < 30:        
        xticks_index = stockDf.index 
        xticks_date = stockDf['Date'].dt.date
    else:
        num = len(stockDf) // 30
        xticks_index = stockDf.index [stockDf.index%num==0]
        xticks_date = stockDf['Date'][stockDf.index%num==0].dt.date
    stockDf = stockDf[["Date","Open","High","Low","Close","up","mid","down"]]
    stockDf['Date'] = pd.to_datetime(stockDf['Date'])
    stockDf['Date'] = stockDf['Date'].map(mpdates.date2num)
    stockDf['DateBak'] = stockDf['Date']
    stockDf['Date'] = stockDf.index
    candlestick_ohlc(ax = ax, quotes = stockDf.values, width=0.75, colorup='green',alpha=0.5, colordown='red')
    stockDf['up'].plot(color="green",label='Uplip Line')
    stockDf['down'].plot(color="red",label='Tooth Line')
    stockDf['mid'].plot(color="blue",label='Chin Line')
    ax.set_ylabel("Close Price")
    ax.grid() 
    ax.legend(prop = {'family':'Helvatica','size':8},loc='upper right')     
    plt.xticks(xticks_index,xticks_date)
    plt.setp(plt.gca().get_xticklabels(), rotation=90)
    plt.rcParams['font.sans-serif']=['Apercu']
    plt.title("Figure Crocodile Curve")
    plt.savefig('Crocodile.png',bbox_inches='tight')
    plt.show()
def getGoalData(stockData,days,rate):
    stockData['Close'] = stockData['Adj Close']
    stockData['max'] = stockData['Close'].rolling(days).max()
    stockData['min'] = stockData['Close'].rolling(days).min()
    stockData['max_win'] = stockData['Close']*(1 + rate) < stockData['max']
    stockData['min_win'] = stockData['Close']*(1 - rate) > stockData['min']
    return stockData    