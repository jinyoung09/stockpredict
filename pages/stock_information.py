import streamlit as st
import FinanceDataReader as fdr
import datetime as dt
import requests

st.title('종목 차트 검색\U0001F607')
Ticker_list = ['010950','103140','001450','005300','036460','001230','010620']
stock_list = ['S-oil','풍산','현대해상','롯데칠성','한국가스공사','동국제강','현대미포조선']
with st.sidebar:
    date = st.date_input(
        "조회 시작일을 선택해 주세요",
        dt.datetime(2022, 1, 1)
    )

    # code = st.text_input(
    #     '종목코드', 
    #     value='',
    #     placeholder='종목코드를 입력해 주세요'
    # )
    selected_column = st.selectbox('History를 보고싶은 Data를 선택하세요', options=stock_list)
if selected_column == 'S-oil':
    code = '010950'
elif selected_column == '풍산':
    code = '103140'
elif selected_column == '현대해상':
    code = '001450'
elif selected_column == '롯데칠성':
    code = '005300'
elif selected_column == '한국가스공사':
    code = '036460'
elif selected_column == '동국제강':
    code = '001230'
elif selected_column == '현대미포조선':
    code = '010620'

if code and date:
    df = fdr.DataReader(code, date)
    data = df.sort_index(ascending=True).loc[:, 'Close']

    tab1, tab2 = st.tabs(['차트', '데이터'])

    with tab1:    
        st.line_chart(data)

    with tab2:
        st.dataframe(df.sort_index(ascending=False))
