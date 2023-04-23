import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime as dt
import FinanceDataReader as fdr
from pandas._libs.tslibs.timestamps import Timestamp
import pytz
import altair as alt
import am9
# import model_update

df_data = pd.read_csv('data.csv')
x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
prev_data = am9.Prevdata(df_data)
prev_X = pd.DataFrame([prev_data], columns=x_data)
columns = ['Date','Time','Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
df_data = am9.DataUpdate_F(df_data, columns)
df_data.to_csv('data.csv', index=False)
new_data = am9.Newdata(df_data)
new_X = pd.DataFrame([new_data], columns=x_data)
df_predict = am9.Predict(prev_X,new_X)
df_predict.to_csv('predict.csv', index=False)
