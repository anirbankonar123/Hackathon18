This repo does two functions:<br>
1.Call Center Optimization : Generates forecast for next 3 months based on historical Call Center data:<br>
a. It uses San Francisco 311 Call Metrics data by month to generate forecasts of the next 3 months (CallCenterPrediction.ipynb)<br>
b. Does initial Data Exploration on this data and applies ARIMA forecasting model <br>
c. Uses Deep Learning with LSTMs to generate forecasts of Call Volume <br>
d. Regression analysis and prediction of Call Transfer Rate, Service Level achieved %age, Average Call duration<br>
2.Used Car price prediction, based on past data:<br>
a. Uses kaggle data from two repositories, to predict Used car data in two steps.<br>
Step 1 : Based on Make, Model, State, Mileage (uses subset of makes for better accuracy)<br>
Step 2: Based on more features like Make, Style, Size, Transmission, City MPG, Engine HP.<br>
b. Does initial Data exploration and Data Analysis (CarPricePredictionAnalysis.ipynb)<br>, and selects set of features, by exploring log scale, polynomials etc.<br>
c. Does Data pre-processing and split into Train and Test data, encoding of categorical vars, normalization of features (CarPricePredictionPreProcessing.ipynb)<br>
c. Does predictive modeling using various techniques, and lists RMSE values, uses Grid search to <br>
update model parameters, k-fold CV, Deep Learning techniques, XG Boost, plot of feature importances, and selects best set of features and model (CarPricePredictionModelExpt.ipynb)<br>
