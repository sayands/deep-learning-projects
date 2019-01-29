# Import the neccessay packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input
from keras.models import Model

def create_mlp(dim, regress = False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim = dim, activation = 'relu'))
    model.add(Dense(4, activation = 'relu'))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation = "linear"))
    
    # return our model 
    return model 


