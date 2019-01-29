# Import the neccessay packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input
from keras.models import Model

def create_cnn(width, height, depth, filters=(16, 32, 64), regress = False):
    # initialise the input shape and channel dimension
    inputShape = (height, width, depth)
    chanDim = -1

    # define model input
    inputs = Input(shape = inputShape)

    # loop over the no.of filters
    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs
        
        # CONV=>RELU=>BN=>POOL
        x = Conv2D(f, (3, 3), padding = 'same')(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = chanDim)(x)
        x = MaxPooling2D(pool_size = (2, 2))(x)

        # flatten the volume, then FC=>RELU=>BN=>DROPOUT
        x = Flatten()(x)
        x = Dense(16)(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis = chanDim)(x)
        x = Dropout(0.5)(x)

        # Apply another FC layer, to match the no.of nodes coming out of MLP
        x = Dense(4)(x)
        x = Activation("relu")(x)

        if regress:
            x = Dense(1, activation = "linear")(x)

        # Construct the CNN
        model = Model(inputs, x)

        # return the CNN
        return model 

