# Recurrent Neural Networks

# Part I - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:5].values

#Preprocess data for training by removing all commas
cols = list(dataset_train)[1:5]
dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0,len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(",","")
 
dataset_train = dataset_train.astype(float)
 
 
training_set = dataset_train.as_matrix() # Using multiple predictors.

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating the data structure with 60 timestamps and 1 output
X_train = []
y_train = []
for i in range(120, 1258):
    X_train.append(training_set_scaled[i-120:i, 0:5])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))

# Part II - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and Dropout Regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 4)))
regressor.add(Dropout(rate = 0.2))

# Adding the second LSTM layer and Dropout Regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Adding the third LSTM lasyer and Dropout Regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

# Adding the fourth LSTM layer and Dropout Regularisation
regressor.add(LSTM(units = 100, return_sequences = False))
regressor.add(Dropout(rate = 0.2))

# Adding the output layer
regressor.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the RNN
import keras
from keras import optimizers
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
regressor.compile(optimizer = optimizer , loss = 'mean_squared_error')

# Fitting the RNN to the Training Set
regressor.fit(X_train, y_train, epochs = 150, batch_size = 32)

# Part 3 - Making the Predictions and Visualising Results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of Jan 2017
ds_test = dataset_test.iloc[:, 1:5]
dataset_total = pd.concat((dataset_train, ds_test), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 120 :].values
#inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = []
for i in range(120, 140):
    X_test.append(inputs[i-120:i, 0:5])
X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))
predicted_stock_price = regressor.predict(X_test)
sc_predict = MinMaxScaler(feature_range=(0,1))
sc_predict.fit_transform(training_set[:,0:1])
predicted_stock_price = sc_predict.inverse_transform(predicted_stock_price)

# Visualising the Results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
