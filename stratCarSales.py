# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:46:00 2018

@author: hackuser1
"""

import pickle
import pandas as pd
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

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

ordinary=pd.read_pickle('C:/users/hackuser1/Hackathon18/ord.pkl')
deluxe=pd.read_pickle('C:/users/hackuser1/Hackathon18/del.pkl')
supdel=pd.read_pickle('C:/users/hackuser1/Hackathon18/supdel.pkl')
luxury=pd.read_pickle('C:/users/hackuser1/Hackathon18/luxury.pkl')
suplux=pd.read_pickle('C:/users/hackuser1/Hackathon18/suplux.pkl')

ordinary.head()
ordinary["Make"].value_counts()
len(ordinary)
len(deluxe)
len(supdel)
len(luxury)
len(suplux)

ordinary["Number of Doors"] = ordinary["Number of Doors"].replace("?",0)
ordinary["Number of Doors"] = ordinary["Number of Doors"].astype('float32')
ordinary["MSRP"] = ordinary["MSRP"].replace("?",0)
ordinary["MSRP"] = ordinary["MSRP"].astype("float32")
ordinary["Engine HP"] = ordinary["Engine HP"].replace("?",0)
ordinary["Engine HP"] = ordinary["Engine HP"].astype("float32")
#ordinary["Age"] = 2017 - ordinary["Year"]

car_trans_type=ordinary['Transmission Type']
encoder = LabelBinarizer()
car_trans_1hot = encoder.fit_transform(car_trans_type)
print(car_trans_1hot.shape) 

ordinary.hist(bins=50,figsize=(20,15))
plt.show()

ordinary["Age"].value_counts()
ordinary["Age-cat"] = np.ceil(ordinary["Age"] / 5)
ordinary["Age-cat"].where(ordinary["Age-cat"] < 5, 5.0, inplace=True)
ordinary["Age-cat"].value_counts() / len(ordinary)
#

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(ordinary, 0.2)
print(len(train_set), "train +", len(test_set), "test")

car_trans_type=test_set['Transmission Type']
encoder = LabelBinarizer()
car_trans_1hot = encoder.fit_transform(car_trans_type)
print(car_trans_1hot.shape) 


split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
#
for train_index, test_index in split.split(ordinary,ordinary["Age-cat"]):
    strat_train_set = ordinary.iloc[train_index]
    strat_test_set = ordinary.iloc[test_index]
    
car_trans_type=strat_train_set['Transmission Type']
encoder = LabelBinarizer()
car_trans_1hot = encoder.fit_transform(car_trans_type)
print(car_trans_1hot.shape) 

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
#carSales_X_num  = carSales_X_num.drop("Year",axis=1)
carSales_X_num  = carSales_X_num.drop("Engine Fuel Type",axis=1)
carSales_X_num  = carSales_X_num.drop("Model",axis=1)
carSales_X_num  = carSales_X_num.drop("Transmission Type",axis=1)
carSales_X_num  = carSales_X_num.drop("Driven_Wheels",axis=1)
carSales_X_num = carSales_X_num.drop("Number of Doors",axis=1)
#carSales_X_num  = carSales_X_num.drop("Market Category",axis=1)
carSales_X_num  = carSales_X_num.drop("Vehicle Style",axis=1)
carSales_X_num = carSales_X_num.drop("Vehicle Size",axis=1)
#carSales_X_num = carSales_X_num.drop("highway MPG",axis=1)
#carSales_X_num = carSales_X_num.drop("Popularity",axis=1)
carSales_X_num = carSales_X_num.drop("Age-cat",axis=1)
carSales_X_num = carSales_X_num.drop("log_MSRP",axis=1)
carSales_X_num = carSales_X_num.drop("MSRP_Mean",axis=1)
carSales_X_num = carSales_X_num.drop("MSRP_group",axis=1)

carSales_test_X_num = carSales_test_X
carSales_test_X_num  = carSales_test_X_num.drop("Make",axis=1)
#carSales_test_X_num  = carSales_test_X_num.drop("Year",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Engine Fuel Type",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Model",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Transmission Type",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Driven_Wheels",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("Number of Doors",axis=1)
#carSales_test_X_num  = carSales_test_X_num.drop("Market Category",axis=1)
carSales_test_X_num  = carSales_test_X_num.drop("Vehicle Style",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("Vehicle Size",axis=1)
#carSales_test_X_num = carSales_test_X_num.drop("highway MPG",axis=1)
#carSales_test_X_num = carSales_test_X_num.drop("Popularity",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("Age-cat",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("log_MSRP",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("MSRP_Mean",axis=1)
carSales_test_X_num = carSales_test_X_num.drop("MSRP_group",axis=1)

print(carSales_X_num.shape)
print(carSales_Y.shape)
print(carSales_test_X_num.shape)
print(carSales_test_Y.shape)
carSales_X_num.head()
carSales_test_X_num.head()

carSales_X_num["Engine HP"] = carSales_X_num["Engine HP"].astype("float32")
#carSales_X_num["Engine Cylinders"] = carSales_X_num["Engine Cylinders"].astype("float32")
carSales_X_num["city mpg"] = carSales_X_num["city mpg"].astype("float32")
carSales_X_num["Age"] = carSales_X_num["Age"].astype("float32")
carSales_X_num.replace('null',np.NaN,inplace=True)
carSales_X_num = pd.DataFrame(carSales_X_num)
carSales_X_num = carSales_X_num.replace('?',0)
carSales_X_num = carSales_X_num.replace('NaN',0)
carSales_X_num = carSales_X_num.replace(np.NaN,0)

carSales_test_X_num["Engine HP"] = carSales_test_X_num["Engine HP"].astype("float32")
#carSales_test_X_num["Engine Cylinders"] = carSales_test_X_num["Engine Cylinders"].astype("float32")
carSales_test_X_num["city mpg"] = carSales_test_X_num["city mpg"].astype("float32")
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
test_X = scaler.transform(carSales_test_X_num)

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

import pickle
train_X_ordinary='C:/users/hackuser1/Hackathon18/train_X_ord.pkl'
test_X_ordinary='C:/users/hackuser1/Hackathon18/test_X_ord.pkl'
train_Y_ordinary='C:/users/hackuser1/Hackathon18/train_Y_ord.pkl'
test_Y_ordinary='C:/users/hackuser1/Hackathon18/test_Y_ord.pkl'

with open(train_X_ordinary, "wb") as f:
    w = pickle.dump(X,f)
with open(test_X_ordinary, "wb") as f:
    w = pickle.dump(test_X,f)
with open(train_Y_ordinary, "wb") as f:
    w = pickle.dump(Y,f)
with open(test_Y_ordinary, "wb") as f:
    w = pickle.dump(test_Y,f)

lin_reg = LinearRegression()
lin_reg.fit(X, Y)

carSales_predictions = lin_reg.predict(test_X)
lin_mse = mean_squared_error(test_Y, carSales_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse #6851

forest_reg = RandomForestRegressor()
forest_reg.fit(X,Y)
carSales_predictions = forest_reg.predict(test_X)
forest_mse = mean_squared_error(test_Y, carSales_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse #3924



