# Import the neccessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation 
from keras.layers.core import Flatten 
from keras.layers.core import Dropout
from keras.layers.core import Dense 
from keras import backend as K 

class SimpleNet:
    @staticmethod
    def build(width, height, depth, classes, reg):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        # CONV => RELU => POOL layers
        model.add(Conv2D(64, (11, 11), input_shape = inputShape, padding = 'same', kernel_regularizer = reg))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # CONV => RELU => POOL layers
        model.add(Conv2D(128, (5, 5), padding = 'same', kernel_regularizer = reg))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # CONV => RELU => POOL layers
        model.add(Conv2D(256, (3, 3), padding = 'same', kernel_regularizer = reg))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer = reg))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation('softmax'))
        # return the constructed network architecture
        return model 