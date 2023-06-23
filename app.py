from flask import Flask, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

@app.route('/forecast/<int:row>', methods=['GET'])
def forecast(row):
    # 데이터 로드
    df = pd.read_csv("/Users/kimminseok/Downloads/연간_통계_자치구별-11-15.txt", header=None, sep='\s+')

    if row >= df.shape[0]:
        return jsonify({'error': 'row out of range'}), 400

    # 해당 행에서 5년치 데이터를 추출하고 이를 사용해 모델을 학습
    traffic_data = pd.to_numeric(df.iloc[row, :60].values, errors='coerce')  # 5년치 데이터

    # 모델 생성 및 학습
    model = ARIMA(traffic_data, order=(5, 1, 0))
    model_fit = model.fit()

    # 예측: 다음 1년치 교통량
    n_periods = 12
    forecast_values = model_fit.forecast(steps=n_periods)

    # 미세먼지량 계산
    beta0 = 0  # 상수항
    beta1 = 0.5  # 교통량에 대한 계수
    dust_amount = 4 * (beta0 + beta1 * forecast_values)  # 미세먼지량 계산

    # 각 예측값 및 미세먼지량 출력
    print("Forecast: ", forecast_values)
    print("Dust amount: ", dust_amount)

    # 둘 다 list 형태로 반환
    return jsonify({'forecast': forecast_values.tolist(), 'dust_amount': dust_amount.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
