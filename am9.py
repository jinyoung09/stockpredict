#오전 9시 전에 진행
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime as dt
import FinanceDataReader as fdr
from pandas._libs.tslibs.timestamps import Timestamp
import pytz

Ticker_list = ['010950','103140','001450','005300','036460','001230','010620']
#S-oil,풍산,현대해상,롯데칠성,한국가스공사,동국제강,현대미포조선
Model_list = ['RandomForestRegressor_Soil.joblib','RandomForestRegressor_Poongsan.joblib','RandomForestRegressor_hyundae_marine.joblib', 'RandomForestRegressor_Lotte7.joblib', 'RandomForestRegressor_Koreagas.joblib','RandomForestRegressor_Dongkuk.joblib','RandomForestRegressor_hyundae_Mipo.joblib']
#Model_list = ['RandomForestRegressor_Soil.joblib','RandomForestRegressor_Poongsan.joblib','RandomForestRegressor_hyundae_marine.joblib', 'RandomForestRegressor_Lotte7.joblib', 'RandomForestRegressor_Koreagas.joblib','RandomForestRegressor_Dongkuk.joblib','RandomForestRegressor_hyundae_Mipo.joblib'  ]
tz = pytz.timezone('Asia/Seoul')  # 한국 시간대
today_date = dt.datetime.now(tz).strftime('%Y-%m-%d')
yesterday_date = dt.datetime.now(tz) - dt.timedelta(days=1)
yesterday_date = yesterday_date.strftime('%Y-%m-%d')

def DataUpdate_F(df_data, columns):
    ticker = ['USDKRW=X','CL=F','GC=F','HG=F','NQ=F','YM=F','^VIX','^TNX','NG=F']
    today_date = dt.datetime.now(tz).strftime('%Y-%m-%d')
    new_data = [today_date,'am9']

    for tic in ticker:
        data = yf.Ticker(tic).history(period='max')
        new_data.append(data.tail(1)['Close'].iloc[0])

    if df_data.empty:
        df_data = pd.DataFrame(columns=columns)
    df_data = df_data.append(pd.Series(new_data, index=columns), ignore_index=True)
    return df_data

#전일데이터 불러오기
def Prevdata(df_data):
    yesterday = df_data['Date'].iloc[-1]
    prev_data = []
    column_list = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
    for index, row in df_data.iterrows():
        if row['Date'] == yesterday and row['Time'] == 'pm4':
            for column in column_list:
                prev_data.append(row[column])
    print(yesterday,prev_data)
    return prev_data

#새로운데이터 불러오기
def Newdata(df_data):
    today = dt.datetime.now(tz).strftime('%Y-%m-%d')
    new_data = []
    column_list = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
    for index, row in df_data.iterrows():
        if row['Date'] == today and row['Time'] == 'am9':
            for column in column_list:
                new_data.append(row[column])
    print(today,new_data)
    return new_data

#예측하기
def Predict(prev_X, new_X):
    df_predict = pd.read_csv('predict.csv', dtype={'code': str})
    today_date = dt.datetime.now(tz).strftime('%Y-%m-%d')
    new_data = []
    for i in range(7):
        temp = {'date': today_date, 'Time': 'am9', 'code': Ticker_list[i]}
        stock_info = yf.Ticker(temp['code'] + '.KS').info
        temp['name'] = stock_info['longName']
        rf = joblib.load(Model_list[i])
        # 예측 수행
        y_pred_new = rf.predict(new_X)
        temp['predict_v'] = round(float(y_pred_new), 2)
        y_pred_prev = rf.predict(prev_X)
        temp['prev_v'] =round(float(y_pred_prev), 2)
        if y_pred_prev > y_pred_new:
            temp['predict'] = '하락'
            temp['predict_rate'] = round(float((y_pred_prev - y_pred_new) / y_pred_prev * 100), 2)
        elif y_pred_prev < y_pred_new:
            temp['predict'] = '상승'
            temp['predict_rate'] = round(float((y_pred_new - y_pred_prev) / y_pred_prev * 100), 2)
        else:
            temp['predict'] = '보합'
        new_data.append(temp)
    df_predict = pd.concat([df_predict, pd.DataFrame(new_data)], ignore_index=True)
    return df_predict
