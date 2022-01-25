import sqlite3 as sq
import pandas as pd
# from sqlalchemy.types import Integer,DateTime
import sys
from datetime import date #,datetime
import datetime
# import datetime
import numpy as np
import yfinance as yf 
from pandas_datareader import  data as pdr 
yf.pdr_override() 
# import urllib.request
# import re
# import asyncio
# from pyppeteer import launch
import nest_asyncio
nest_asyncio.apply()
# import requests
import Nasdaq100
class StockDB:
    def __init__(self, dbname = 'stock.db'):
        self.conn = sq.connect(dbname,detect_types=sq.PARSE_DECLTYPES)
        self.cur = self.conn.cursor()
    def saveStockPrice(self,df,symbol):
        df.to_sql(symbol, self.conn, if_exists="replace") 
    def appendStockPrice(self,df,symbol):
        df.to_sql(symbol, self.conn, if_exists="append") 
    def delStockPrice(self,symbol):
        sql = f'''
        drop table if exists {symbol};
        '''
        try:
            self.cur.execute(sql)   
            self.conn.commit()
        except:
            print('Drop failed.')
            self.conn.rollback()
    def getStockPrice(self,symbol):
        sql = f'''
        SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}';
        '''
        self.cur.execute(sql)
        values = self.cur.fetchall()
        if len(values)==0:
            return self.loadStockPrice(symbol)
        else:
            df_stock = pd.read_sql_query("SELECT * from '" + symbol + "' ",self.conn)
            maxday = df_stock.loc[df_stock['Day']==df_stock['Day'].max()]['Day'].values[0]
            maxdate = pd.to_datetime(maxday).date() + datetime.timedelta(days=1)
            today = date.today()
            if maxdate < datetime.date(today.year, today.month, 1):
                self.appendloadStockPrice(symbol,startdate = str(maxdate ), enddate= str(today))
            return pd.read_sql_query("SELECT * from '" + symbol + "' ",self.conn)
    def getSerialStockData(self,symbol):
        sample = self.getStockPrice(symbol)
        sample.index = pd.to_datetime(sample['Day'])
        sample.index.name = 'Index'
        sample.reset_index(inplace = True)
        sample.drop(['Index'],axis=1,inplace = True)
        sample.index.name = 'Index'
        return sample[['Date','Adj Close']]
    def fillTQQQPrice(self,symbol1,symbol2):
        self.getStockPrice(symbol1)
        self.getStockPrice(symbol2)
        df_QQQ = pd.read_sql_query(f"SELECT * from '{symbol1}' ",self.conn)
        minday_QQQ = df_QQQ.loc[df_QQQ['Day']==df_QQQ['Day'].min()]['Day'].values[0]
        df_TQQQ = pd.read_sql_query(f"SELECT * from '{symbol2}' ",self.conn)
        minday_TQQQ = df_TQQQ.loc[df_TQQQ['Day']==df_TQQQ['Day'].min()]['Day'].values[0]
        if minday_QQQ < minday_TQQQ:
            data_TQQQ = df_QQQ.loc[df_QQQ['Day']<= np.datetime64(minday_TQQQ)]
            d_TQQQ = df_TQQQ.loc[df_TQQQ['Day'] >= np.datetime64(minday_TQQQ)]
            d_QQQ = df_QQQ.loc[df_QQQ['Day'] >= np.datetime64(minday_TQQQ)]
            data_TQQQ['rate'] = (data_TQQQ['Adj Close'] - data_TQQQ['Adj Close'].shift(1))/data_TQQQ['Adj Close'].shift(1)
            data_TQQQ['tri_rate'] = data_TQQQ['rate'] * 3
            init_tri_rate = data_TQQQ.loc[data_TQQQ['Day'] == np.datetime64(minday_TQQQ),'tri_rate'].values[0]
            data_TQQQ = data_TQQQ.loc[data_TQQQ['Day']< np.datetime64(minday_TQQQ)]
            iniprice_TQQQ = df_TQQQ.loc[df_TQQQ['Day']==df_TQQQ['Day'].min()]['Adj Close'].values[0]
            data_TQQQ = data_TQQQ[::-1]
            for index,row in data_TQQQ.iterrows():
                iniprice_TQQQ = iniprice_TQQQ /(1 + init_tri_rate)
                init_tri_rate = row['tri_rate']
                data_TQQQ.loc[index,'Adj Close'] = iniprice_TQQQ
            data_TQQQ = data_TQQQ[::-1]
            data_TQQQ.drop(['rate','tri_rate'],axis=1, inplace=True)
            df_uni = pd.concat([data_TQQQ, df_TQQQ])
            df_uni = df_uni.sort_values(by=['Date'])
            df_uni.reset_index(drop=True, inplace=True)
            self.delStockPrice(symbol2)
            df_uni.to_sql(symbol2, self.conn, if_exists="append", index=False)
    def importBond(self,symbol,filename,years):
        df = pd.read_csv('data/' + filename, names=["Date", symbol], skiprows=1)
        df['Day'] = pd.to_datetime(df['Date'])
        df['Adj Close'] = 100.0/(1+ df[symbol]/100) **years
        df['rate'] = (df['Adj Close'] - df['Adj Close'].shift(1))/df['Adj Close']
        df['tri_rate'] = df['rate'] * 3
        df.to_sql(symbol, self.conn, if_exists="replace", index=False)
        newsymbol = symbol + '_tri'
        df_tri = df
        iniprice = 100
        df_tri.loc[0,'Adj Close'] = iniprice
        for index,row in df_tri.iterrows():
            if not pd.isnull(row['tri_rate']):
                iniprice = iniprice * (1 + row['tri_rate'])
                df_tri.loc[index,'Adj Close'] = iniprice
            else:
                df_tri.loc[index,'Adj Close'] = iniprice
        df_tri.to_sql(newsymbol, self.conn, if_exists="replace", index=False)
        self.b = df_tri
    def loadStockPrice(self,symbol,startdate='1980-01-01'):
        print(symbol)
        enddate=str(date.today())
        df = pdr.get_data_yahoo(symbol, start=startdate, end=enddate, progress = False)
        if df.empty:
            return None
        df['Day'] = df.index
        self.saveStockPrice(df, symbol)
        return df
    def appendloadStockPrice(self,symbol,startdate='1980-01-01',enddate='2021-12-31'):
        if yf.Ticker(symbol).isin == '-': 
            return None
        try:
            df = pdr.get_data_yahoo(symbol, start=startdate, end=enddate, progress = False)
            df['Day'] = df.index
            self.appendStockPrice(df, symbol)
        except:
            pass
    def displayDB(self):
        sql = '''
        SELECT 
            name
        FROM 
            sqlite_master 
        WHERE 
            type ='table' AND 
            name NOT LIKE 'sqlite_%' ;
        '''
        self.cur.execute(sql)
        values = self.cur.fetchall()
        for value in values:
            print('table: ',value)
        num = 0
        if values:
            print("\n")
            for value in values:
                num = num +1
                print('\n' + "#"*60 + "\ntable " + str(num) + " name: ",value[0] + '\n' + "#"*60 + '\n')
                df = pd.read_sql_query("SELECT * from "+ value[0] + " limit 5", self.conn)
                print(df)
                self.cur.execute("SELECT * from "+ value[0] + " limit 5")
                names = [description[0] for description in self.cur.description]
                print('columns: ',names)
                nu = pd.read_sql_query("SELECT count(*) from "+ value[0] , self.conn)
                print('Count: ', nu)
    def backupDB(self):
        pass
    def resotreDB(self):
        pass
    def insert(self, sql):
        try:
            cursor= self.cur.execute(sql)
            self.conn.commit()
        except:
            print('Insert failed.')
            self.conn.rollback()
        finally:
            pass
        return cursor.lastrowid 
    def SQL(self, sql):
        try:
            self.cur.execute(sql)
        except:
            print(sql)
            sys.exit()
        self.conn.commit()
    def select(self, sql):
        try:
            self.cur.execute(sql)
            self.conn.commit()
        except:
            print('Select Failed.')
            return 
        results = self.cur.fetchall()
        return results
    def initializeDB(self):
        pass
    def loadN100Data(self):
        comList,finalLinks,cpnamelist = Nasdaq100.getSymbols()
        for symbol in finalLinks:
            df = self.getStockPrice(symbol)
            if not ( df is None):
                print(symbol,' is downloaded.\n')
            else:
                print(symbol,' is None.\n')
        pass
    def checkN100Data(self):
        comList,finalLinks,cpnamelist = Nasdaq100.getSymbols()
        for symbol in finalLinks:
            sql = f'''
            SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}';
            '''
            self.cur.execute(sql)
            values = self.cur.fetchall()
            if len(values)==0:
                print(f'{symbol} doesn\'t exist. \n')
                self.delStockPrice(symbol)
                df = self.getStockPrice(symbol)
                if not ( df is None):
                    print(symbol,' is downloaded jsut now.\n')
                else:
                    print(symbol,' can\'t be downloaded.\n')
        print(f'{len(finalLinks)} symbols data founded.')
        return
    def __del__(self):
        self.conn.close()
# print(df.Day.iloc[0].days)