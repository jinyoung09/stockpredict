import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime as dt
import FinanceDataReader as fdr
from pandas._libs.tslibs.timestamps import Timestamp
import pytz
import am9
import pm4
# import model_update

df_data = pd.read_csv('data.csv')

#pm4 원자재 data 업데이트하기
columns = ['Date','Time','Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
df_data = pm4.DataUpdate_S(df_data, columns)
df_data.to_csv('data.csv', index=False)
#pm4 Actual data 기록하기
df_predict = pd.read_csv('predict.csv', dtype={'code': str})
df_predict = pm4.ActualDataInput(df_predict)
df_predict.to_csv('predict.csv', index=False)