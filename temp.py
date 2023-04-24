#temp
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime as dt
import FinanceDataReader as fdr
from pandas._libs.tslibs.timestamps import Timestamp
import pytz
#import altair as alt
# import am9st
# import pm4

Ticker_list = ['010950','103140','001450','005300','036460','001230','010620']
#S-oil,풍산,현대해상,롯데칠성,한국가스공사,동국제강,현대미포조선
Model_list = ['RandomForestRegressor_Soil.joblib','RandomForestRegressor_Poongsan.joblib','RandomForestRegressor_hyundae_marine.joblib', 'RandomForestRegressor_Lotte7.joblib', 'RandomForestRegressor_Koreagas.joblib','RandomForestRegressor_Dongkuk.joblib','RandomForestRegressor_hyundae_Mipo.joblib'  ]
tz = pytz.timezone('Asia/Seoul')  # 한국 시간대
today_date = dt.datetime.now(tz).strftime('%Y-%m-%d')
yesterday_date = dt.datetime.now(tz) - dt.timedelta(days=1)
yesterday_date = yesterday_date.strftime('%Y-%m-%d')

# df_data = pd.read_csv('data.csv')
# st.dataframe(df_data, use_container_width=False)
# df_data = df_data.drop(df_data.index[-1])
# df_data.to_csv('data.csv', index=False)
# print(df_data)
# st.dataframe(df_data, use_container_width=False)

# df_predict = pd.read_csv('predict.csv', dtype={'code': str})
# df_predict = df_predict[df_predict['date'] != "2023-04-24"]
# df_predict.to_csv('predict.csv', index=False)
# df_predict_show = df_predict.drop(['predict_v','prev_v', 'Time'], axis=1)
# df_predict_show = df_predict_show[df_predict_show['date'] == today_date]

st.dataframe(df_predict_show, use_container_width=False)


