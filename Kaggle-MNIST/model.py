from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam, RMSProp
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

def init_model():
    # Defining Model
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Valid', activation = 'relu', input_shape = (28, 28, 1)))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'Valid', activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(519, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 1e-3), metrics = ["accuracy"])

    annealer = ReduceLROnPlateau(monitor = 'val_acc', patience = 1, verbose = 2, factor = 0.5, min_lr = 0.0000001)

    datagen =ImageDataGenerator(
        featurewise_center = False,
        samplewise_center = False,
        featurewise_std_normalization = False,
        samplewise_std_normalization = False,
        zca_whitening = False,
        rotation_range = 10,
        zoom_range = 0.1,
        width_shift_range = 0.1,
        horizontal_flip = False,
        vertical_flip = False
    )

    return model, annealer, datagen