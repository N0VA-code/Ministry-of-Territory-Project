from flask import Flask, jsonify
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

@app.route('/forecast', methods=['GET'])
def forecast():
    # 데이터 로드
    df = pd.read_excel("/Users/kimminseok/Downloads/연간_통계_자치구별-11-15.xls", header=None)

    all_forecasts = []
    all_dust_amounts = []

    for row in range(df.shape[0]):
        # 각 행에서 5년치 데이터를 추출하고 이를 사용해 모델을 학습
        traffic_data = df.iloc[row, :60].values  # 5년치 데이터

        # 모델 생성 및 학습
        model = ARIMA(traffic_data, order=(5, 1, 0))
        model_fit = model.fit()

        # 예측: 다음 1년치 교통량
        n_periods = 12
        forecast = model_fit.forecast(steps=n_periods)[0]
        all_forecasts.append(forecast.tolist())

        # 미세먼지량 계산
        beta0 = 0  # 상수항
        beta1 = 0.5  # 교통량에 대한 계수
        dust_amount = 4*(beta0 + beta1 * forecast)  # 미세먼지량 계산
        all_dust_amounts.append(dust_amount.tolist())

    return jsonify({'forecasts': all_forecasts, 'dust_amounts': all_dust_amounts})
