import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드
df = pd.read_excel("/Users/kimminseok/Downloads/연간_통계_자치구별-11-15.xls", header=None)

# 각 행에 대해 ARIMA 모델을 학습하고 예측
for row in range(df.shape[0]):
    # 각 행에서 5년치 데이터를 추출하고 이를 사용해 모델을 학습
    traffic_data = df.iloc[row, :60].values  # 5년치 데이터

    # 모델 생성 및 학습
    model = ARIMA(traffic_data, order=(5, 1, 0))
    model_fit = model.fit()

    # 예측: 다음 1년치 교통량
    n_periods = 12
    forecast = model_fit.forecast(steps=n_periods)

    # 예측 결과의 차분을 각 월별로 다르게 조정
    diff = np.diff(forecast)

    # 1월-7월: 차분을 12배로 조정
    forecast[1:8] += diff[:7] * 12
    # 8월-12월: 차분을 24배로 조정
    forecast[8:] += diff[7:] * 100
    forecast[0] = forecast[0] * 1
    forecast = forecast * 1

    # 예측 결과와 원래 데이터를 그래프로 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(np.concatenate([traffic_data, forecast]), 'r', label='Prediction')
    plt.plot(traffic_data, 'b', label='Original')
    plt.legend()

    # x축 표시 간격을 1달 단위로 설정
    plt.xticks(np.arange(0, len(traffic_data) + len(forecast), step=12), rotation=45)

    plt.title(f"ARIMA model and prediction for row {row + 1}")
    plt.show()

    # 미세먼지량 계산
    forecast = forecast.ravel()  # 예측된 교통량을 1차원 배열로 변환
    beta0 = 0  # 상수항
    beta1 = 0.5  # 교통량에 대한 계수

    dust_amount = 4*(beta0 + beta1 * forecast)  # 미세먼지량 계산

    print(f"Forecast for row {row + 1}: ", forecast)  # 예측된 교통량
    print(f"Predicted Dust Amount for row {row + 1}: ", dust_amount)  # 예측된 미세먼지량
