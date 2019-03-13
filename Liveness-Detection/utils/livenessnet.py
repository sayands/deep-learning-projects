# Importing the neccessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense 
from keras import backend as K 

class LivenessNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model with the input shape to be "channels last"
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = -1

        # fist CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(16, (3, 3), padding = "same", input_shape = inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(16, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(32, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model 