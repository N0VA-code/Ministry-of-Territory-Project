# Ministry-of-Territory-Project
Ministry of Territory(South Korea) Traffic Data Utilizing Predicting Model Developing(Team Project)
```.txt

<서울시 호흡기 환자>
https://kosis.kr/statHtml/statHtml.do?orgId=117&tblId=DT_117049_A005&lang_mode=ko&vw_cd=MT_ZTITLE&list_id=117_11749_000_001&conn_path=I4

<서울시 미세먼지 농도>
https://cleanair.seoul.go.kr/statistics/dayAverage

<서울시 교통정보>
http://175.193.202.192:8080/refRoom/openRefRoom_7_4.do

machine learning algorithm => 교통량 예측
g(교통량 예측) => 미세먼지 예측

Dust Amount = β0 + β1 * Traffic Amount + ε
```

```..txt
machine learning predicting traffic amount.py description:
his code utilizes the ARIMA (AutoRegressive Integrated Moving Average) time series prediction model to train and predict data. Specifically, the code uses the ARIMA model to predict traffic volume data and then calculates the amount of fine dust based on this prediction.
Firstly, it imports the pandas and statsmodels libraries and loads an Excel file from a specified path into a dataframe. The code then iterates through each row of the dataframe, training the ARIMA model with the data from each row. The order parameter for the ARIMA model is given in the form (p, d, q), which represents the auto-regressive part, integrated part, and moving average part respectively. Therefore, (5, 1, 0) means AR=5, I=1, MA=0, signifying auto-regression on 5 previous values, 1 degree of differencing, and no moving average term.
After the model is trained, it is used to predict the traffic volume for the next 12 months (1 year). Post-processing work is done on the predicted values. This includes calculating the differences (diff) and adding this to the monthly predictions after different scaling. From January to July, the difference is scaled by a factor of 12, while from August to December, it is scaled by a factor of 100.
The reason for this is based on the assumption that data can increase or change gradually over time, and as time goes on, the change can occur more steeply. For example, if the traffic volume in a certain autonomous district changes according to the season, there may be a tendency for traffic volume to increase significantly between August and December. A larger scaling factor for the differences is used to reflect this.
Finally, the amount of fine dust is calculated based on the predicted traffic volume. This calculation formula looks like a simple linear regression model, multiplying the coefficient (beta1) by the traffic volume and adding a constant term (beta0). This result is multiplied by a proportional constant to obtain the final amount of fine dust.
```




