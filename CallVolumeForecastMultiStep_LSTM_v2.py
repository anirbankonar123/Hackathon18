#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:22:25 2018

@author: anirban
"""
import pandas as pd
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import to_datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from keras.models import load_model
from keras import optimizers


df = pd.read_csv('/home/anirban/Downloads/311_call_metrics.csv', header=0)
print(df.head())

#convert the month column to datetime and sort on that
df['month'] = pd.to_datetime(df.month)
df=df.sort_values(by='month',ascending=True)
#print(df)
#plot the values from first two columns
dataset = df.values[:,1:2]
plt.plot(dataset)
plt.show()

n_lag = 1
n_seq = 3
n_test = 11

#use the Call volume column to create Series in pandas
series_call_volume=Series(data=df.values[:,1])
print(len(series_call_volume))

# prepare data for normalization
values = series_call_volume.values
values = values.reshape((len(values), 1))
print(values)

# normalize the values in a scale of 0 to 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print( ' Min: %f, Max: %f ' % (scaler.data_min_, scaler.data_max_))
# normalize the dataset and print
normalized = scaler.transform(values)
print(normalized)

#extract values from Time Series
values_call_volume = series_call_volume.values
	
# create Time Series from the difference of x(t) and x(t-1)
diff = list()
for i in range(1, len(values_call_volume)):
	value = values_call_volume[i] - values_call_volume[i - 1]
	diff.append(value)
	
diff_series = Series(diff)

#prepare values for Normalization
diff_values = diff_series.values
diff_values = diff_values.reshape(len(diff_values), 1)

# Normalize values in range -1, 1 using Sklearn MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_values = scaler.fit_transform(diff_values)
scaled_values = scaled_values.reshape(len(scaled_values), 1)


df = DataFrame(scaled_values)
cols = list()

#shift x values by 1 to create input sequence t-1
cols.append(df.shift(1))
    
#shift y values by 1,2,3 and create forecast sequence (t, t+1, t+2) at next 3 timesteps as y
for i in range(0, n_seq):
    cols.append(df.shift(-i))

#concat the values to create training dataset
agg = concat(cols, axis=1)
# drop rows with NaN values
agg.dropna(inplace=True)

#supervised training dataset with first column as x and next three columns as y
supervised_values = agg.values

print(scaler.inverse_transform(supervised_values[0,0:4]))
print(scaler.inverse_transform(supervised_values[124,0:4]))

#use the 11 monthly values from 2017 as test data and all prev yrs as training data
train, test = supervised_values[0:-11], supervised_values[-11:]

print(train.shape)
print(test.shape)

# reshape training into [samples, timesteps, features]
X, y = train[:, 0:1], train[:, 1:]
X = X.reshape(X.shape[0], 1, X.shape[1])
print(X.shape)
print(y.shape)
X_test, y_test = test[:, 0:1], test[:, 1:]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print(X_test.shape)
print(y_test.shape)
# design network
def fit_model_stateful(n_cells):
    # define model
    model = Sequential()
    model.add(LSTM(n_cells, batch_input_shape=(1,X.shape[1],X.shape[2]),stateful=True))
    model.add(Dense(3))
    # compile model
    myOptimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=myOptimizer)
    #model.compile(loss='mse',optimizer='adam')
    model.fit(X, y, epochs=1500, shuffle=False, verbose=0,batch_size=1)
    model.reset_states()
    # evaluate model
    loss = model.evaluate(X_test, y_test, verbose=0, batch_size=1)
    return loss

def fit_model_stateless(n_cells):
    # define model
    model = Sequential()
    model.add(LSTM(n_cells, batch_input_shape=(1,X.shape[1],X.shape[2]),stateful=False))
    model.add(Dense(3))
    # compile model
    myOptimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=myOptimizer)
    #model.compile(loss='mse',optimizer='adam')
    model.fit(X, y, epochs=1500, shuffle=False, verbose=0,batch_size=1)
    # evaluate model
    loss = model.evaluate(X_test, y_test, verbose=0, batch_size=1)
    return loss

# define scope of search
params = [1, 5, 10]
n_repeats = 5
# grid search parameter values
scores = DataFrame()
for value in params:
# repeat each experiment multiple times
    loss_values_stateful = list()
    for i in range(n_repeats):
        loss = fit_model_stateful(value)
        loss_values_stateful.append(loss)
        print( ' >%d/%d param=%f, loss=%f ' % (i+1, n_repeats, value, loss))
# store results for this parameter
    scores[str(value)] = loss_values_stateful
# summary statistics of results
print(scores.describe())
# box and whisker plot of results
scores.boxplot()
pyplot.show()


# define scope of search
params = [1, 5, 10]
n_repeats = 5
# grid search parameter values
scores = DataFrame()
for value in params:
# repeat each experiment multiple times
    loss_values_stateless = list()
    for i in range(n_repeats):
        loss = fit_model_stateless(value)
        loss_values_stateless.append(loss)
        print( ' >%d/%d param=%f, loss=%f ' % (i+1, n_repeats, value, loss))
# store results for this parameter
    scores[str(value)] = loss_values_stateless
# summary statistics of results
print(scores.describe())
# box and whisker plot of results
scores.boxplot()
pyplot.show()


#Based on error plot, choose 5 neurons in intermediate layer
model = Sequential()
model.add(LSTM(10, batch_input_shape=(1, 1, 1), stateful=False))
model.add(Dense(3))
myOptimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=myOptimizer)
history = model.fit(X, y, epochs=1400, validation_data=(X_test,y_test), batch_size=1, verbose=1, shuffle=False)
loss = model.evaluate(X_test,y_test,batch_size=1,verbose=0)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Save the model
model.save('/home/anirban/my_project/lstm_time_series_stless.h5')

#Load model
model = load_model('/home/anirban/my_project/lstm_time_series_stless.h5')

# make forecast
forecasts = list()
	
for i in range(len(test)):
    X_test_inp = X_test[i,0,0]
    X_test_inp = X_test_inp.reshape(1,1,1)
    forecast = model.predict(X_test_inp, batch_size=1)
    forecasts.append(forecast)
    
print(forecasts)

# invert difference values to have absolute value
def inverse_difference(last_ob, unscaled_diff):
	inverted = list()
	inverted.append(last_ob+unscaled_diff[0])
	for i in range(1, len(unscaled_diff)):
		inverted.append(inverted[i-1]+unscaled_diff[i])
	return inverted

inverted = list()
for i in range(len(forecasts)):
	# create array from forecast
	forecast = np.array(forecasts[i])
    #the 3 step values of call volume into the future
	forecast = forecast.reshape(1, 3)
	# invert scaling
	inv_scale = scaler.inverse_transform(forecast)
	inv_scale = inv_scale[0, :]

	index = len(series_call_volume) - (n_test+2) + i - 1
	last_ob = series_call_volume.values[index]
    
    #get actual value of call volume based on forecast difference, and actual
	inv_diff = inverse_difference(last_ob, inv_scale)
	inverted.append(inv_diff)

inverted = np.around(inverted, decimals=0)

forecasts_inv = inverted
print(forecasts_inv)

#get actual values of Call volume for the 3 time steps, by taking from index=n_lag onwards
actuals = [row[n_lag:] for row in test]
print(test)
print(actuals)

# invert scaling of actual values and get actual call volume from difference
inverted = list()
for i in range(len(actuals)):
	actual = np.array(actuals[i])
	actual = actual.reshape(1, 3)

	inv_scale = scaler.inverse_transform(actual)
	inv_scale = inv_scale[0, :]

	index = len(series_call_volume) - (n_test+2) + i - 1
	last_ob = series_call_volume.values[index]
	inv_diff = inverse_difference(last_ob, inv_scale)

	inverted.append(inv_diff)

inverted = np.around(inverted, decimals=0)

actuals_inv = inverted
print(actuals_inv)

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(actuals, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in actuals]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
 
   
# evaluate forecasts
print(len(actuals_inv))
evaluate_forecasts(actuals_inv, forecasts_inv, n_lag, n_seq)
# plot forecastss
print(actuals_inv.shape)
pyplot.plot(actuals_inv[:,0])
pyplot.plot(forecasts_inv[:,0],color='red')
pyplot.show()

