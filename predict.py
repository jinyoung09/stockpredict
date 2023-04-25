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
import math
# import model_update

Ticker_list = ['010950','103140','001450','005300','036460','001230','010620']
#S-oil,풍산,현대해상,롯데칠성,한국가스공사,동국제강,현대미포조선
Model_list = ['RandomForestRegressor_Soil.joblib','RandomForestRegressor_Poongsan.joblib','RandomForestRegressor_hyundae_marine.joblib', 'RandomForestRegressor_Lotte7.joblib', 'RandomForestRegressor_Koreagas.joblib','RandomForestRegressor_Dongkuk.joblib','RandomForestRegressor_hyundae_Mipo.joblib'  ]
tz = pytz.timezone('Asia/Seoul')  # 한국 시간대
today_date = dt.datetime.now(tz).strftime('%Y-%m-%d')
yesterday_date = (dt.datetime.now(tz) - dt.timedelta(days=1)).strftime('%Y-%m-%d')


df_data = pd.read_csv('data.csv')

st.title('Stock Predict\U0001F4C8')
#st.header('해당 예측결과는 ')
st.subheader('오늘의 주요 지수')
#코스피지수, 코스닥지수, S&P500 지수 보여주기기
# KOSPI 지수 가져오기
kospi= fdr.DataReader('KS11')
kospi_v = kospi['Close'].iloc[-1]
yesterday_kospi = kospi['Close'].iloc[-2]
kospi_change = (kospi_v - yesterday_kospi) / yesterday_kospi * 100
#kospi_change = kospi['Close'].pct_change().loc[yesterday_date] * 100

#코스닥 지수 가져오기
kosdaq= fdr.DataReader('KQ11')
#kosdaq_v = kosdaq['Close'].tail(1)
kosdaq_v = kosdaq['Close'].iloc[-1]
yesterday_kosdaq = kosdaq['Close'].iloc[-2]
kosdaq_change = (kosdaq_v - yesterday_kosdaq) / yesterday_kosdaq * 100
# kosdaq_change = kosdaq['Close'].pct_change().loc[yesterday_date]  * 100
#S&P500 지수 가져오기
sp= fdr.DataReader('US500')
sp_v = sp['Close'].iloc[-1]
yesterday_sp = sp['Close'].iloc[-2]
sp_change = (sp_v - yesterday_sp) / yesterday_sp * 100
#화면표시
kospi, kosdaq, sp = st.columns(3)
#kospi.metric("KOSPI", round(kospi_v.values[0],2), round(kospi_change.values[0],2))
kospi.metric("KOSPI", round(kospi_v,2), round(kospi_change,2))
kosdaq.metric("KOSDAQ",  round(kosdaq_v,2), round(kosdaq_change,2))
sp.metric("S&P500", round(sp_v,2), round(sp_change,2))

st.write("오늘의 주요 정보(\U0001F4B1\U0001F697\U0001F947\U0001F949\U0001F4C9\U0001F4B2)들을 보여줍니다")

# if st.button("오늘의 정보 Update 및 예측(am9)(관리자Only)"):
#     x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
#     prev_data = am9.Prevdata(df_data)
#     prev_X = pd.DataFrame([prev_data], columns=x_data)
#     columns = ['Date','Time','Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
#     df_data = am9.DataUpdate_F(df_data, columns)
#     df_data.to_csv('data.csv', index=False)

#     new_data = am9.Newdata(df_data)
#     new_X = pd.DataFrame([new_data], columns=x_data)

#     df_predict = am9.Predict(prev_X,new_X)
#     df_predict.to_csv('predict.csv', index=False)

df_data_show = df_data[df_data['Date'] == today_date]
st.table(df_data_show)
st.write("어제의 주요 지수 정보들을 보여줍니다")
df_data_show = df_data[df_data['Date'] == yesterday_date]
# st.write(df_data_show)
st.table(df_data_show)
st.subheader('오늘의 주식예측')
st.write("주식의 상승/하락을 예측합니다")
st.markdown("**:red[\U0001F645\U0001F645\U0001F645예측결과는 참고자료이고, 주식투자에 대한 책임은 본인에게 있습니다 :)]**")
df_predict = pd.read_csv('predict.csv', dtype={'code': str})
df_predict_show = df_predict.drop(['predict_v','prev_v', 'Time'], axis=1)
df_predict_show = df_predict_show[df_predict_show['date'] == today_date]

st.dataframe(df_predict_show, use_container_width=False)

# if st.button("오늘의 주요지수 Update 및 주식 예측결과(pm4)"):
#      #pm4 원자재 data 업데이트하기
#     columns = ['Date','Time','Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
#     x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']

#     df_data = pm4.DataUpdate_S(df_data, columns)
#     df_data.to_csv('data.csv', index=False)

#     #pm4 Actual data 기록하기
#     df_predict = pd.read_csv('predict.csv', dtype={'code': str})
#     df_predict = pm4.ActualDataInput(df_predict)
#     df_predict.to_csv('predict.csv', index=False)

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

st.subheader('예측결과')
st.write("어제",yesterday_date,"의 예측결과 입니다")
# st.write(yesterday_date)
df_predict_yesterday = pd.read_csv('predict.csv', dtype={'code': str})
df_predict_yesterday = df_predict_yesterday[df_predict['date'] == yesterday_date]
df_predict_show = df_predict_yesterday.drop(['predict_v','prev_v', 'Time'], axis=1)
#st.dataframe(df_predict_show, use_container_width=False)
def highlight(row):
    if row['actual'] == row['predict']:
        return ['background-color: yellow'] * len(row)
    else:
        return [''] * len(row)

# 스타일 적용
styled_df = df_predict_show.style.apply(highlight, axis=1)
st.subheader('예측결과')
st.write("어제",yesterday_date,"의 예측결과 입니다")
st.dataframe(styled_df, use_container_width=False)

#예측결과 구하기
# 'predict' 컬럼이 '상승'인 행 필터링
filter_rising = df_predict_yesterday['predict'] == '상승'
df_rising = df_predict_yesterday[filter_rising]

# 'actual predict' 컬럼의 평균 계산
mean_actual_predict_rising = df_rising['actual_rate'].mean()
if math.isnan(mean_actual_predict_rising):
    mean_actual_predict_rising = 0.0

# kospi= fdr.DataReader('KS11')
# kospi_change_prevday = kospi['Close'].pct_change().iloc[-2] * 100

# st.write('KOSPI는', round(float(kospi_change_prevday),2))
# KOSPI 지수 가져오기
# kospi = fdr.DataReader('KS11')

# # 전일 대비 등락률 계산하기
# kospi_change_prevday = ((kospi['Close'].iloc[-2] - kospi['Open'].iloc[-2]) / kospi['Open'].iloc[-2]) * 100
# print("Close, Open", kospi['Close'].iloc[-2], kospi['Open'].iloc[-2],kospi['Low'].iloc[-2])

# kospi = yf.Ticker('^KS11')
# # 어제 날짜 구하기
# today_date = dt.datetime.now(tz).strftime('%Y-%m-%d')
# yesterday_date = (dt.datetime.now(tz) - dt.timedelta(days=1)).strftime('%Y-%m-%d')
kospi= fdr.DataReader('KS11')
# kospi_v = kospi['Close'].iloc[-1]
yesterday_kospi_open = kospi['Open'].iloc[-2]
yesterday_kospi_close = kospi['Close'].iloc[-2]
pct_change_y = (yesterday_kospi_close - yesterday_kospi_open) / yesterday_kospi_open * 100

# history() 메서드를 사용하여 데이터 가져오기
# try:
#data_y = kospi.history(yesterday_date)
# except IndexError:
#     data = kospi.history(period="max-1d").tail(2)

# Open가 대비해서 Close의 등락률 계산
#pct_change_y = (data_y['Close'][0] - data_y['Open'][0]) / data_y['Open'][0] * 100
st.write('KOSPI는', round(float(pct_change_y), 2))
print(yesterday_kospi_open,yesterday_kospi_close)
#print(data_y.index[0], data_y['Close'][0],data_y['Open'][0],yesterday_date,today_date)
st.write('당신이 이 예측결과대로 투자했다면 평균', round(mean_actual_predict_rising,2))