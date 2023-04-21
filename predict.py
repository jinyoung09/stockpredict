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
import pm4
import model_update

Ticker_list = ['010950','103140','001450','005300','036460','001230','010620']
#S-oil,풍산,현대해상,롯데칠성,한국가스공사,동국제강,현대미포조선
Model_list = ['RandomForestRegressor_Soil.joblib','RandomForestRegressor_Poongsan.joblib','RandomForestRegressor_hyundae_marine.joblib', 'RandomForestRegressor_Lotte7.joblib', 'RandomForestRegressor_Koreagas.joblib','RandomForestRegressor_Dongkuk.joblib','RandomForestRegressor_hyundae_Mipo.joblib'  ]
tz = pytz.timezone('Asia/Seoul')  # 한국 시간대
today_date = dt.datetime.now(tz).strftime('%Y-%m-%d')
yesterday_date = dt.datetime.now(tz) - dt.timedelta(days=1)
yesterday_date = yesterday_date.strftime('%Y-%m-%d')

df_data = pd.read_csv('data.csv')

st.title('Stock Predict')
#st.header('오늘의 주식 예측')
st.subheader('오늘의 주요 지수')
st.write("오늘의 주요 지수 정보들을 보여줍니다")

if st.button("오늘의 주요지수 Update 및 예측(am9)"):
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

df_data_show = df_data[df_data['Date'] == today_date]
st.table(df_data_show)
st.write("어제의 주요 지수 정보들을 보여줍니다")
df_data_show = df_data[df_data['Date'] == yesterday_date]
# st.write(df_data_show)
st.table(df_data_show)
st.subheader('오늘의 주식예측')
st.write("주식의 상승/하락을 예측합니다")
df_predict = pd.read_csv('predict.csv', dtype={'code': str})
df_predict_show = df_predict.drop(['predict_v','prev_v', 'Time'], axis=1)
df_predict_show = df_predict_show[df_predict_show['date'] == today_date]

st.dataframe(df_predict_show, use_container_width=False)

if st.button("오늘의 주요지수 Update 및 주식 예측결과(pm4)"):
     #pm4 원자재 data 업데이트하기
    columns = ['Date','Time','Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
    x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']

    df_data = pm4.DataUpdate_S(df_data, columns)
    df_data.to_csv('data.csv', index=False)

    #pm4 Actual data 기록하기
    df_predict = pd.read_csv('predict.csv', dtype={'code': str})
    df_predict = pm4.ActualDataInput(df_predict)
    df_predict.to_csv('predict.csv', index=False)

st.subheader('History')
if st.button("주요지수 History"):
    st.write("주요지수 History를 보여줍니다")
    #   df_data['Date'] = pd.to_datetime(df_data['Date']).apply(lambda x: x.timestamp())

    #   st.slider("조회하고싶은 날짜범위를 선택하세요", 
    #       min_value=pd.to_datetime(df_data['Date'][0]), 
    #       max_value=pd.to_datetime(df_data['Date'].max()), 
    #       value=(pd.to_datetime(df_data['Date'][0]), pd.to_datetime(df_data['Date'].max())))

    st.write(df_data)


if st.button("주식예측 History"):
      st.write("주요지수 History를 보여줍니다")
      df_predict_show = df_predict.drop(['predict_v','prev_v', 'Time'], axis=1)
      st.write(df_predict_show)

#if st.button("그래프로 주요 지수 History를 보고싶으신가요?"):
    #column_options = df_data.columns.tolist()
    
column_options = [col for col in df_data.columns.tolist() if col not in ['Date', 'Time']]
selected_column = st.selectbox('History를 보고싶은 Data를 선택하세요', options=column_options)
st.write('You selected:', selected_column)
chart_data = df_data[[selected_column, 'Date']]
chart = alt.Chart(chart_data).mark_line().encode(
    x='Date',
    y=alt.Y(selected_column, scale=alt.Scale(zero=False))
).interactive()
st.altair_chart(chart,use_container_width=True)
