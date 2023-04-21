import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
import numpy as np
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestRegressor

df_stock = pd.read_csv('stock_0406.csv',dtype={'code': str})
df_010950= df_stock[df_stock['code'] == '010950']
start_date = '2022-04-01'
end_date = df_010950['Date'].max()
df_010950 = df_010950.loc[(df_010950['Date'] >= start_date) & (df_010950['Date'] <= end_date)]
# 독립 변수와 종속 변수 지정
x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
X = df_010950[x_data]
y = df_010950['Close']
# train set과 test set으로 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestRegressor(random_state=0, max_depth=10, min_samples_leaf=1, min_samples_split=2,n_estimators=100)
rf.fit(X_train, y_train)

# 모델 예측
y_pred = rf.predict(X_test)

# 모델 성능 평가
# R-squared 계산
score = rf.score(X_test, y_test)

# MSE 계산
mse = mean_squared_error(y_test, y_pred)

# MAE 계산
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE score: {mse:.2f}')
print(f'MAE score: {mae:.2f}')
print(f'R-squared score: {score:.2f}')

# 교차 검증으로 모델 성능 평가
neg_mse_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# 교차 검증 결과 출력
print('Cross-validation Negative MSE scores:', neg_mse_scores)
print('Average score:', np.mean(avg_rmse))

joblib.dump(rf, 'RandomForestRegressor_Soil.joblib')


df_103140= df_stock[df_stock['code'] == '103140']
start_date = '2021-01-01'
end_date = df_103140['Date'].max()
df_103140 = df_103140.loc[(df_103140['Date'] >= start_date) & (df_103140['Date'] <= end_date)]
# 독립 변수와 종속 변수 지정
x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
X = df_103140[x_data]
y = df_103140['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(random_state=0, max_depth=10, min_samples_leaf=1, min_samples_split=2,n_estimators=100)

# 모델 학습
rf.fit(X_train, y_train)

# 모델 예측
y_pred = rf.predict(X_test)

# 모델 성능 평가
# R-squared 계산
score = rf.score(X_test, y_test)

# MSE 계산
mse = mean_squared_error(y_test, y_pred)

# MAE 계산
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE score: {mse:.2f}')
print(f'MAE score: {mae:.2f}')
print(f'R-squared score: {score:.2f}')

# 교차 검증으로 모델 성능 평가
neg_mse_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# 교차 검증 결과 출력
print('Cross-validation Negative MSE scores:', neg_mse_scores)
print('Average score:', np.mean(avg_rmse))

joblib.dump(rf, 'RandomForestRegressor_Poongsan.joblib')

#현대해상 모델 만들기

df_001450= df_stock[df_stock['code'] == '001450']
start_date = '2021-01-01'
end_date = df_001450['Date'].max()
df_001450 = df_001450.loc[(df_001450['Date'] >= start_date) & (df_001450['Date'] <= end_date)]
# 독립 변수와 종속 변수 지정
x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
X = df_001450[x_data]
y = df_001450['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(random_state=0, max_depth=10, min_samples_leaf=1, min_samples_split=2,n_estimators=100)

# 모델 학습
rf.fit(X_train, y_train)

# 모델 예측
y_pred = rf.predict(X_test)

# 모델 성능 평가
# R-squared 계산
score = rf.score(X_test, y_test)

# MSE 계산
mse = mean_squared_error(y_test, y_pred)

# MAE 계산
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE score: {mse:.2f}')
print(f'MAE score: {mae:.2f}')
print(f'R-squared score: {score:.2f}')

# 교차 검증으로 모델 성능 평가
neg_mse_scores = cross_val_score(rf, X, y, cv=4, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# 교차 검증 결과 출력
print('Cross-validation Negative MSE scores:', neg_mse_scores)
print('Average score:', np.mean(avg_rmse))

joblib.dump(rf, 'RandomForestRegressor_hyundae_marine.joblib')

#롯데칠성
df_005300= df_stock[df_stock['code'] == '005300']
start_date = '2021-01-01'
end_date = df_005300['Date'].max()
df_005300 = df_005300.loc[(df_005300['Date'] >= start_date) & (df_005300['Date'] <= end_date)]
# 독립 변수와 종속 변수 지정
x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
X = df_005300[x_data]
y = df_005300['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestRegressor(random_state=0, max_depth=10, min_samples_leaf=1, min_samples_split=2,n_estimators=100)

# 모델 학습
rf.fit(X_train, y_train)

# 모델 예측
y_pred = rf.predict(X_test)

# 모델 성능 평가
# R-squared 계산
score = rf.score(X_test, y_test)

# MSE 계산
mse = mean_squared_error(y_test, y_pred)

# MAE 계산
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE score: {mse:.2f}')
print(f'MAE score: {mae:.2f}')
print(f'R-squared score: {score:.2f}')

# 교차 검증으로 모델 성능 평가
neg_mse_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# 교차 검증 결과 출력
print('Cross-validation Negative MSE scores:', neg_mse_scores)
print('Average score:', np.mean(avg_rmse))

joblib.dump(rf, 'RandomForestRegressor_Lotte7.joblib')

#한국도시공사
df_036460= df_stock[df_stock['code'] == '036460']
start_date = '2022-04-01'
end_date = df_036460['Date'].max()
df_036460 = df_036460.loc[(df_036460['Date'] >= start_date) & (df_036460['Date'] <= end_date)]
# 독립 변수와 종속 변수 지정
x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
X = df_036460[x_data]
y = df_036460['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestRegressor(random_state=0, max_depth=10, min_samples_leaf=1, min_samples_split=2,n_estimators=100)

# 모델 학습
rf.fit(X_train, y_train)

# 모델 예측
y_pred = rf.predict(X_test)

# 모델 성능 평가
# R-squared 계산
score = rf.score(X_test, y_test)

# MSE 계산
mse = mean_squared_error(y_test, y_pred)

# MAE 계산
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE score: {mse:.2f}')
print(f'MAE score: {mae:.2f}')
print(f'R-squared score: {score:.2f}')

# 교차 검증으로 모델 성능 평가
neg_mse_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# 교차 검증 결과 출력
print('Cross-validation Negative MSE scores:', neg_mse_scores)
print('Average score:', np.mean(avg_rmse))

joblib.dump(rf, 'RandomForestRegressor_Koreagas.joblib')

#현대미포조선
df_010620= df_stock[df_stock['code'] == '010620']
start_date = '2022-04-01'
end_date = df_010620['Date'].max()
df_010620 = df_010620.loc[(df_010620['Date'] >= start_date) & (df_010620['Date'] <= end_date)]
# 독립 변수와 종속 변수 지정
x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
X = df_010620[x_data]
y = df_010620['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestRegressor(random_state=0, max_depth=10, min_samples_leaf=1, min_samples_split=2,n_estimators=100)

# 모델 학습
rf.fit(X_train, y_train)

# 모델 예측
y_pred = rf.predict(X_test)

# 모델 성능 평가
# R-squared 계산
score = rf.score(X_test, y_test)

# MSE 계산
mse = mean_squared_error(y_test, y_pred)

# MAE 계산
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE score: {mse:.2f}')
print(f'MAE score: {mae:.2f}')
print(f'R-squared score: {score:.2f}')

# 교차 검증으로 모델 성능 평가
neg_mse_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# 교차 검증 결과 출력
print('Cross-validation Negative MSE scores:', neg_mse_scores)
print('Average score:', np.mean(avg_rmse))

joblib.dump(rf, 'RandomForestRegressor_hyundae_Mipo.joblib')

#동국제강
df_001230= df_stock[df_stock['code'] == '001230']
start_date = '2021-01-01'
end_date = df_001230['Date'].max()
df_001230 = df_001230.loc[(df_001230['Date'] >= start_date) & (df_001230['Date'] <= end_date)]
# 독립 변수와 종속 변수 지정
x_data = ['Close_exchange', 'Close_oil', 'Close_gold', 'Close_copper', 'Close_nasdaq', 'Close_dow', 'Close_vix','Close_treasury','Close_gas']
X = df_001230[x_data]
y = df_001230['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestRegressor(random_state=0, max_depth=10, min_samples_leaf=1, min_samples_split=2,n_estimators=100)

# 모델 학습
rf.fit(X_train, y_train)

# 모델 예측
y_pred = rf.predict(X_test)

# 모델 성능 평가
# R-squared 계산
score = rf.score(X_test, y_test)

# MSE 계산
mse = mean_squared_error(y_test, y_pred)

# MAE 계산
mae = mean_absolute_error(y_test, y_pred)

print(f'MSE score: {mse:.2f}')
print(f'MAE score: {mae:.2f}')
print(f'R-squared score: {score:.2f}')

# 교차 검증으로 모델 성능 평가
neg_mse_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

# 교차 검증 결과 출력
print('Cross-validation Negative MSE scores:', neg_mse_scores)
print('Average score:', np.mean(avg_rmse))

joblib.dump(rf, 'RandomForestRegressor_Dongkuk.joblib')