#오후 4시이후에 진행
import pandas as pd
import yfinance as yf
import joblib
import datetime as dt
import FinanceDataReader as fdr
from pandas._libs.tslibs.timestamps import Timestamp
import pytz
import streamlit as st

Ticker_list = ['010950','103140','001450','005300','036460','001230','010620']
#S-oil,풍산,현대해상,롯데칠성,한국가스공사,동국제강,현대미포조선
#Model_list = ['RandomForestRegressor_Soil.joblib','/content/drive/MyDrive/project/RandomForestRegressor_Poongsan.joblib','/content/drive/MyDrive/project/RandomForestRegressor_hyundae_marine.joblib', '/content/drive/MyDrive/project/RandomForestRegressor_Lotte7.joblib', '/content/drive/MyDrive/project/RandomForestRegressor_Koreagas.joblib','/content/drive/MyDrive/project/RandomForestRegressor_Dongkuk.joblib','/content/drive/MyDrive/project/RandomForestRegressor_hyundae_Mipo.joblib'  ]
Model_list = ['RandomForestRegressor_Soil.joblib','RandomForestRegressor_Poongsan.joblib','RandomForestRegressor_hyundae_marine.joblib', 'RandomForestRegressor_Lotte7.joblib', 'RandomForestRegressor_Koreagas.joblib','RandomForestRegressor_Dongkuk.joblib','RandomForestRegressor_hyundae_Mipo.joblib']
df_predict = pd.read_csv('predict.csv', dtype={'code': str})
df_data = pd.read_csv('data.csv')
tz = pytz.timezone('Asia/Seoul')  # 한국 시간대

#오후에 Data 업데이트 하기기
def DataUpdate_S(df_data, columns):
    ticker = ['USDKRW=X','CL=F','GC=F','HG=F','NQ=F','YM=F','^VIX','^TNX','NG=F']
    today_date = dt.datetime.now(tz).strftime('%Y-%m-%d')
    new_data = [today_date,'pm4']

    for tic in ticker:
        data = yf.Ticker(tic).history(period='max')
        new_data.append(data.tail(1)['Close'].iloc[0])

    if df_data.empty:
        df_data = pd.DataFrame(columns=columns)
    #df_data = df_data.append(pd.Series(new_data, index=columns), ignore_index=True)
    new_row = pd.DataFrame([new_data], columns=columns)
    df_data = pd.concat([df_data, new_row], ignore_index=True)
    return df_data

#Actual data 기록하기
def ActualDataInput(df_predict):
    today_date = dt.datetime.today().strftime('%Y-%m-%d')
    
    for i in range(7):
        code = Ticker_list[i]
        
        temp = fdr.DataReader(code)
        yesterday_close = temp.iloc[-2]['Close']
        today_price = temp.iloc[-1]['Close']
        change_rate = round(((today_price - yesterday_close) / yesterday_close * 100),2)
        df_predict.loc[(df_predict['code'] == code) & (df_predict['date'] == today_date), 'actual_rate'] = change_rate

        if change_rate > 0:
            df_predict.loc[(df_predict['code'] == code) & (df_predict['date'] == today_date), 'actual'] = '상승'
        elif change_rate < 0:
            df_predict.loc[(df_predict['code'] == code) & (df_predict['date'] == today_date), 'actual'] = '하락'
        else:
            df_predict.loc[(df_predict['code'] == code) & (df_predict['date'] == today_date), 'actual'] = '보합'

    return df_predict
