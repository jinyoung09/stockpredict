#모델 업데이트0422
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestRegressor
import datetime as dt
import FinanceDataReader as fdr


st.title('모델 성능 향상을 위한 모델업데이트')
df_stock = pd.read_csv('update_data.csv',dtype={'code': str})
#df_stock = pd.read_csv('stock_0406.csv',dtype={'code': str})
st.subheader("마지막 모델업데이트 Date는 아래와 같습니다")
last_date = df_stock['Date'].iloc[-1]
st.write(last_date)

last_date = dt.datetime.strptime(last_date, '%Y-%m-%d')
next_day = last_date + dt.timedelta(days=1)
start_date = next_day.strftime('%Y-%m-%d')
today = dt.datetime.now().date() 
date = st.date_input(
    "업데이트를 원하는 마지막 날짜를 선택해주세요",
     dt.datetime(today.year, today.month, today.day)
)
end_date = date.strftime('%Y-%m-%d')


if st.button("모델을 업데이트"):
    st.write("모델을 업데이트 하는 중입니다")
    def get_company_name(ticker):
        stock_name = []
        for tic in ticker:
            ticker_info = yf.Ticker(tic + '.KS').info
            stock_name.append(ticker_info['longName'])
        return stock_name
    Ticker_list = ['010950','103140','001450','005300','036460','001230','010620']
    #S-oil,풍산,현대해상,롯데칠성,한국가스공사,동국제강,현대미포조선
    Model_list = ['RandomForestRegressor_Soil.joblib','RandomForestRegressor_Poongsan.joblib','RandomForestRegressor_hyundae_marine.joblib', 'RandomForestRegressor_Lotte7.joblib', 'RandomForestRegressor_Koreagas.joblib','RandomForestRegressor_Dongkuk.joblib','RandomForestRegressor_hyundae_Mipo.joblib']
    stock_name = get_company_name(Ticker_list)
    startdate = '2023-04-07'
    enddate = end_date
    # 데이터프레임 초기화
    df_stock_temp= pd.DataFrame(columns=['Date', 'code', 'Name', 'Close'])
    for ticker in Ticker_list:
        # 종목명 가져오기
        company_name =  yf.Ticker(ticker + '.KS').info['longName']
        # 주식 데이터 가져오기
        try:
            #stock_data = yf.download(ticker + '.KS', start=startdate, end=enddate)
            stock_data = fdr.DataReader(ticker,startdate,enddate)
        except Exception as e:
            print(f"Error occurred while fetching data for {ticker}: {e}")
            continue
        
        # 종가 정보 추출
        close_data = stock_data['Close']
        # 데이터프레임에 정보 추가
        data_list = [{'Date': date, 'code': ticker, 'Name': company_name, 'Close': close} for date, close in close_data.items()]
        df_stock_temp = pd.concat([df_stock_temp, pd.DataFrame(data_list)], ignore_index=True)
    df_data= pd.DataFrame(columns=['date','Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas'])
    data_ticker = ['USDKRW=X','CL=F','GC=F','HG=F','NQ=F','YM=F','^VIX','^TNX','NG=F']
    date_range = pd.date_range(start=startdate, end=enddate)
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        row = {'date': date_str}
        for ticker in data_ticker:
            try:
                ticker_data = yf.download(ticker, start=startdate, end=enddate)
                close_value = ticker_data.loc[date_str, 'Close']
            except:
                close_value = None
            # 각 종목별로 해당하는 컬럼에 값을 추가
            if ticker == 'USDKRW=X':
                row['Close_exchange'] = close_value
            elif ticker == 'CL=F':
                row['Close_oil'] = close_value
            elif ticker == 'GC=F':
                row['Close_gold'] = close_value
            elif ticker == 'HG=F':
                row['Close_copper'] = close_value
            elif ticker == 'NQ=F':
                row['Close_nasdaq'] = close_value
            elif ticker == 'YM=F':
                row['Close_dow'] = close_value
            elif ticker == '^VIX':
                row['Close_vix'] = close_value
            elif ticker == '^TNX':
                row['Close_treasury'] = close_value
            elif ticker == 'NG=F':
                row['Close_gas'] = close_value
        # 데이터프레임에 행 추가
        df_data = pd.concat([df_data, pd.DataFrame(row, index=[0])], ignore_index=True)
    df_data = df_data.rename(columns={'date': 'Date'})
    df_data['Date'] = pd.to_datetime(df_data['Date'])
    # Date 컬럼을 기준으로 merge
    merged_df = pd.merge(df_data, df_stock_temp, on='Date', how='inner')
    merged_df = merged_df[['Date', 'code', 'Name', 'Close', 'Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix', 'Close_treasury', 'Close_gas']]
    merged_df.dropna(inplace=True)
    merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
    df_merged = pd.concat([df_stock, merged_df], ignore_index=True)
    # temp_df = merged_df.head(20)
    # print(temp_df)
    df_merged.to_csv('update_data.csv', index=False)
    x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
    # 모델을 초기화합니다.
    model = RandomForestRegressor(random_state=0, max_depth=10, min_samples_leaf=1, min_samples_split=2,n_estimators=100)
    new_data = pd.read_csv('update_data.csv',dtype={'code': str})
    # print(new_data.head(10))
    # print(new_data.tail(10))
    # 새로운 데이터를 불러옵니다.
    for i in range(7):
        model = joblib.load(Model_list[i])
        df_temp= new_data[new_data['code'] == Ticker_list[i]]
        X = df_temp[x_data]
        y = df_temp['Close']
        model.fit(X, y)
        joblib.dump(model, Model_list[i])
    st.write("모델 업데이트가 완료 되었습니다 !!!!")