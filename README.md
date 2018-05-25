This repo does two functions:<br>
Time Series Analysis:<br>
a. It uses San Francisco 311 Call Metrics data by month to generate forecasts of the next 3 months (CallCenterPrediction.ipynb)<br>
b. Does initial Data Exploration on this data using R<br>
Used Car price prediction:<br>
a. Uses kaggle data from two repositories, to predict Used car data in two steps.<br>
Step 1 : Based on Make, Model, State, Mileage (uses subset of makes for better accuracy)<br>
Step 2: Based on more features like Make, Style, Size, Transmission, City MPG, Engine HP.<br>
b. Does initial Data exploration and Data Analysis (CarPricePredictionAnalysis.ipynb)<br>
c. Does Data pre-processing and split into Train and Test data, encoding of categorical vars, normalization of features (CarPricePredictionPreProcessing.ipynb)<br>
c. Does predictive modeling using various techniques, and lists RMSE values, feature importances, uses Grid search to <br>
update model parameters, k-fold CV, Deep Learning techniques, Error plots, and selects best set of features and model (CarPricePredictionModelExpt.ipynb)<br>
