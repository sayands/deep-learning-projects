# Artificial Neural Network

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Making the ANN

# Importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing ANN
classifier = Sequential()

# Adding the input layer and the hidden layer
classifier.add(Dense(units = 6,kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.12))
classifier.add(Dense(units = 6,kernel_initializer = 'glorot_uniform', activation = 'relu'))
classifier.add(Dropout(rate = 0.12))
classifier.add(Dense(output_dim = 1,kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Pitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 25, epochs = 100)

# Predicting the Test Set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# EVALUATING,TUNING AND IMPROVING THE ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6,kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(rate = 0.12))
    classifier.add(Dense(units = 6,kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.12))
    classifier.add(Dense(output_dim = 1,kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 25, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# Tuning the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6,kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(rate = 0.12))
    classifier.add(Dense(units = 6,kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier.add(Dropout(rate = 0.12))
    classifier.add(Dense(output_dim = 1,kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = { 'batch_size' : [10, 25],
                'epochs'  : [100, 150],
                'optimizer' : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
gird_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

pred = grid_search.predict(X_test)
pred = pred > 0.5
