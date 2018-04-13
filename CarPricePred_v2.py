# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:20:58 2018

@author: 103467
"""

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from keras.models import Sequential
from keras.layers import Dense   
from keras import optimizers

carSales = pd.read_csv("C:/users/hackuser1/Hackathon18/data.csv")

carSales.info()
chevrolet = carSales['Make'] == "Chevrolet"
carSales = carSales[chevrolet]
carSales = pd.DataFrame(carSales.values,columns = ["Make", "Model", "Year", "Engine Fuel Type","Engine HP","Engine Cylinders","Transmission Type","Driven_Wheels","Number of Doors","Market Category","Vehicle Size","Vehicle Style","highway MPG","city MPG","Popularity","MSRP"])
carSales.head()

carSales.describe()


carSales["Number of Doors"] = carSales["Number of Doors"].replace("?",0)
carSales["Number of Doors"] = carSales["Number of Doors"].astype('float32')
carSales["MSRP"] = carSales["MSRP"].replace("?",0)
carSales["MSRP"] = carSales["MSRP"].astype("float32")
carSales["Engine HP"] = carSales["Engine HP"].replace("?",0)
carSales["Engine HP"] = carSales["Engine HP"].astype("float32")
carSales["Year"] = carSales["Year"].astype("float32")
carSales["Age"] = 2017 - carSales["Year"]


carSales["Age"].value_counts()
carSales["Age-cat"] = np.ceil(carSales["Age"] / 5)
carSales["Age-cat"].where(carSales["Age-cat"] < 5, 5.0, inplace=True)
carSales["Age-cat"].value_counts() / len(carSales)
#
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
#
for train_index, test_index in split.split(carSales,carSales["Age-cat"]):
    strat_train_set = carSales.loc[train_index]
    strat_test_set = carSales.loc[test_index]
#
strat_train_set["Age-cat"].value_counts() / len(strat_train_set)
strat_test_set["Age-cat"].value_counts() / len(strat_test_set)

carSales_X = strat_train_set.copy()
carSales_X = strat_train_set.drop("MSRP", axis=1) # drop labels for training set
carSales_Y = strat_train_set["MSRP"].copy()

carSales_test_X = strat_test_set.copy()
carSales_test_X = strat_test_set.drop("MSRP", axis=1) # drop labels for training set
carSales_test_Y = strat_test_set["MSRP"].copy()

carSales_Y = carSales_Y.reshape(carSales_Y.shape[0],1)
carSales_test_Y = carSales_test_Y.reshape(carSales_test_Y.shape[0],1)
print(carSales_X.shape)
print(carSales_Y.shape)
print(carSales_test_X.shape)
print(carSales_test_Y.shape)

carSales_X.head()

carSales_X_num = carSales_X
carSales_X_num  = carSales_X_num.drop("Make",axis=1)
carSales_X_num  = carSales_X_num.drop("Year",axis=1)
carSales_X_num  = carSales_X_num.drop("Engine Fuel Type",axis=1)
carSales_X_num  = carSales_X_num.drop("Model",axis=1)
carSales_X_num  = carSales_X_num.drop("Transmission Type",axis=1)
carSales_X_num  = carSales_X_num.drop("Driven_Wheels",axis=1)
carSales_X_num = carSales_X_num.drop("Number of Doors",axis=1)
carSales_X_num  = carSales_X_num.drop("Market Category",axis=1)
carSales_X_num  = carSales_X_num.drop("Vehicle Style",axis=1)
carSales_X_num = carSales_X_num.drop("Vehicle Size",axis=1)
carSales_X_num = carSales_X_num.drop("highway MPG",axis=1)
carSales_X_num = carSales_X_num.drop("Popularity",axis=1)
carSales_X_num = carSales_X_num.drop("Age-cat",axis=1)


carSales_test_X_num = carSales_test_X
carSales_test_X_num  = carSales_test_X_num.drop("Make",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Year",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Engine Fuel Type",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Model",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Transmission Type",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Driven_Wheels",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("Number of Doors",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Market Category",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Vehicle Style",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("Vehicle Size",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("highway MPG",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("Popularity",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("Age-cat",axis=1)


print(carSales_X_num.shape)
print(carSales_Y.shape)
print(carSales_test_X_num.shape)
print(carSales_test_Y.shape)
carSales_X_num.head()
carSales_test_X_num.head()

carSales_X_num["Engine HP"] = carSales_X_num["Engine HP"].astype("float32")
carSales_X_num["Engine Cylinders"] = carSales_X_num["Engine Cylinders"].astype("float32")
carSales_X_num["city MPG"] = carSales_X_num["city MPG"].astype("float32")
carSales_X_num["Age"] = carSales_X_num["Age"].astype("float32")
carSales_X_num.replace('null',np.NaN,inplace=True)
carSales_X_num = pd.DataFrame(carSales_X_num)
carSales_X_num = carSales_X_num.replace('?',0)
carSales_X_num = carSales_X_num.replace('NaN',0)
carSales_X_num = carSales_X_num.replace(np.NaN,0)

carSales_test_X_num["Engine HP"] = carSales_test_X_num["Engine HP"].astype("float32")
carSales_test_X_num["Engine Cylinders"] = carSales_test_X_num["Engine Cylinders"].astype("float32")
carSales_test_X_num["city MPG"] = carSales_test_X_num["city MPG"].astype("float32")
carSales_test_X_num["Age"] = carSales_test_X_num["Age"].astype("float32")
carSales_test_X_num.replace('null',np.NaN,inplace=True)
carSales_test_X_num = pd.DataFrame(carSales_test_X_num)
carSales_test_X_num = carSales_test_X_num.replace('?',0)
carSales_test_X_num = carSales_test_X_num.replace('NaN',0)
carSales_test_X_num = carSales_test_X_num.replace(np.NaN,0)

print(carSales_X_num.shape)
print(carSales_test_X_num.shape)

m=carSales_X_num.isnull().any()
print(m[m])
m=np.isfinite(carSales_X_num.select_dtypes(include=['float64'])).any()
print(m[m])


imputer = Imputer(missing_values=0,strategy="mean")
imputer.fit(carSales_X_num)
imputer.fit(carSales_test_X_num)

scaler = StandardScaler()
X = scaler.fit_transform(carSales_X_num)
test_X = scaler.fit_transform(carSales_test_X_num)

car_trans_type = carSales_X["Transmission Type"]
encoder = LabelBinarizer()
car_trans_1hot = encoder.fit_transform(car_trans_type)
print(car_trans_1hot.shape)

X = np.concatenate((X,car_trans_1hot),axis=1)

car_driven_wheels = carSales_X["Driven_Wheels"]
encoder = LabelBinarizer()
car_drive_1hot = encoder.fit_transform(car_driven_wheels)
print(car_drive_1hot.shape)

X = np.concatenate((X,car_drive_1hot),axis=1)

car_vehicle_size = carSales_X["Vehicle Size"]
encoder = LabelBinarizer()
car_size_1hot = encoder.fit_transform(car_vehicle_size)
print(car_size_1hot.shape)

X = np.concatenate((X,car_size_1hot),axis=1)

print(X.shape)

car_trans_type_test = carSales_test_X["Transmission Type"]
encoder = LabelBinarizer()
car_trans_1hot_test = encoder.fit_transform(car_trans_type_test)
print(car_trans_1hot_test.shape)

test_X = np.concatenate((test_X,car_trans_1hot_test),axis=1)

car_driven_wheels_test = carSales_test_X["Driven_Wheels"]
encoder = LabelBinarizer()
car_drive_1hot_test = encoder.fit_transform(car_driven_wheels_test)
print(car_drive_1hot_test.shape)

test_X = np.concatenate((test_X,car_drive_1hot_test),axis=1)

car_vehicle_size_test = carSales_test_X["Vehicle Size"]
encoder = LabelBinarizer()
car_size_1hot_test = encoder.fit_transform(car_vehicle_size_test)
print(car_size_1hot_test.shape)

test_X = np.concatenate((test_X,car_size_1hot_test),axis=1)

print(test_X.shape)

Y = pd.DataFrame(carSales_Y)
m=Y.isnull().any()
print(m[m])
m=np.isfinite(Y.select_dtypes(include=['float64'])).any()
print(m[m])

test_Y = pd.DataFrame(carSales_test_Y)
m=test_Y.isnull().any()
print(m[m])
m=np.isfinite(test_Y.select_dtypes(include=['float64'])).any()
print(m[m])

print(X.shape)
print(Y.shape)

print(test_X.shape)
print(test_Y.shape)


lin_reg = LinearRegression()
lin_reg.fit(X, Y)

carSales_predictions = lin_reg.predict(test_X)
lin_mse = mean_squared_error(test_Y, carSales_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse #6851

sgd_reg = SGDRegressor(n_iter=500,penalty=None,eta0=0.1)
sgd_reg.fit(X, Y.values.ravel())

carSales_predictions = sgd_reg.predict(test_X)
sgd_mse = mean_squared_error(test_Y, carSales_predictions)
sgd_rmse = np.sqrt(sgd_mse)
sgd_rmse #6911

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X,Y)
carSales_predictions = tree_reg.predict(test_X)
tree_mse = mean_squared_error(test_Y, carSales_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse #5395

scores = cross_val_score(tree_reg,X,Y,scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores = np.sqrt(-scores)

print("scores:",tree_rmse_scores)
print("mean:",tree_rmse_scores.mean())
print("std dev:",tree_rmse_scores.std())

lin_scores = cross_val_score(lin_reg,X,Y,scoring="neg_mean_squared_error",cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

print("scores:",lin_rmse_scores)
print("mean:",lin_rmse_scores.mean())
print("std dev:",lin_rmse_scores.std())

forest_reg = RandomForestRegressor()
forest_reg.fit(X,Y)
carSales_predictions = forest_reg.predict(test_X)
forest_mse = mean_squared_error(test_Y, carSales_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse #3653 - comparable with Deep Learning

forest_scores = cross_val_score(forest_reg,X,Y,scoring="neg_mean_squared_error",cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

print("scores:",forest_rmse_scores)
print("mean:",forest_rmse_scores.mean())
print("std dev:",forest_rmse_scores.std())

from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(X, Y)
carSales_predictions = svm_reg.predict(test_X)
svm_mse = mean_squared_error(test_Y, carSales_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse#16726


from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(X, Y)
grid_search.best_params_
feature_importances = grid_search.best_estimator_.feature_importances_
num_attribs = ["Engine HP","Engine Cylinders","city MPG","Age"]
categorical_attribs = ['AUTOMATIC', 'DIRECT_DRIVE', 'MANUAL']+['all wheel drive', 'four wheel drive', 'front wheel drive',
       'rear wheel drive']+['Compact', 'Large', 'Midsize']
attributes = num_attribs+categorical_attribs
sorted(zip(feature_importances, attributes), reverse=True)

final_model = grid_search.best_estimator_
final_predictions = final_model.predict(test_X)
final_mse = mean_squared_error(test_Y, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse #3715

final_model_scores = cross_val_score(final_model,X,Y,scoring="neg_mean_squared_error",cv=10)
final_model_scores = np.sqrt(-final_model_scores)

print("scores:",final_model_scores)
print("mean:",final_model_scores.mean())
print("std dev:",final_model_scores.std())

#print(len(X))
#print(X[:1])
def plot_learning_curves(model, X, y):
    
    train_errors, val_errors = [], []
    for m in range(1, len(X)):
        model.fit(X[:m], Y[:m])
        y_train_predict = model.predict(X[:m])
        y_val_predict = model.predict(test_X)
        train_errors.append(mean_squared_error(Y[:m], y_train_predict))
        val_errors.append(mean_squared_error(test_Y, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)   

plot_learning_curves(final_model, X, Y)
#plt.axis([0, 80, 0, 3])                         # not shown in the book
#save_fig("/home/anirban/learning_curves_plot")   # not shown
plt.show()


from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, Y)
carSales_predictions = elastic_net.predict(test_X)
elastic_mse = mean_squared_error(test_Y, carSales_predictions)
elastic_rmse = np.sqrt(elastic_mse)
elastic_rmse#6922


from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, Y)
carSales_predictions = ridge_reg.predict(test_X)
ridge_mse = mean_squared_error(test_Y, carSales_predictions)
ridge_rmse = np.sqrt(ridge_mse)
ridge_rmse#6855


from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, Y)
carSales_predictions = lasso_reg.predict(test_X)
lasso_mse = mean_squared_error(test_Y, carSales_predictions)
lasso_rmse = np.sqrt(lasso_mse)
lasso_rmse#6851

from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X, Y)
carSales_predictions = gbrt.predict(test_X)
gbrt_mse = mean_squared_error(test_Y, carSales_predictions)
gbrt_rmse = np.sqrt(gbrt_mse)
gbrt_rmse#5182

gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
gbrt_slow.fit(X, Y)
carSales_predictions = gbrt_slow.predict(test_X)
gbrt_slow_mse = mean_squared_error(test_Y, carSales_predictions)
gbrt_slow_rmse = np.sqrt(gbrt_slow_mse)
gbrt_slow_rmse#3614

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X,Y)

errors = [mean_squared_error(test_Y, y_pred)
          for y_pred in gbrt.staged_predict(test_X)]
bst_n_estimators = np.argmin(errors)
print(bst_n_estimators)

gbrt_best = GradientBoostingRegressor(max_depth=2,n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X, Y)

carSales_predictions = gbrt_best.predict(test_X)
gbrt_best_mse = mean_squared_error(test_Y, carSales_predictions)
gbrt_best_rmse = np.sqrt(gbrt_best_mse)
gbrt_best_rmse#3758


Y=Y.values
test_Y = test_Y.values
print(Y.shape)
print(test_Y.shape)

model = Sequential()
 
model.add(Dense(20,input_dim=(X.shape[1]),activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.summary()

model.add(Dense(40,input_dim=(X.shape[1]),activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))
model.summary()

model.add(Dense(40,input_dim=(X.shape[1]),activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))
model.summary()

myOptimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=myOptimizer)
history = model.fit(X, Y, epochs=300,  validation_data=(test_X,test_Y), batch_size=10, verbose=2)
plt.plot(history.history['loss'], color = 'blue')
plt.plot(history.history['val_loss'], color=  'red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

carSales_predictions = model.predict(test_X)
dl_mse = mean_squared_error(test_Y, carSales_predictions)
dl_rmse = np.sqrt(dl_mse)
dl_rmse #3477

from xgboost import XGBRegressor

xgboost_reg = XGBRegressor()
xgboost_reg.fit(X,Y)

carSales_predictions = xgboost_reg.predict(test_X)
xgboost_mse = mean_squared_error(test_Y, carSales_predictions)
xgboost_rmse = np.sqrt(xgboost_mse)
xgboost_rmse#3538




    
    
