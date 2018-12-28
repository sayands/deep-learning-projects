import numpy as np 
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Flatten, Reshape 
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score 

# Load the data
X = np.load('X.npy')
y = np.load('y.npy')

# Convert classes to vector 
nb_classes = 2
y = np_utils.to_categorical(y, nb_classes).astype(np.float32)

# shuffle all the data
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# prepare weighting for classes since they are unbalanced
class_totals = y.sum(axis = 0)
class_weight = class_totals.max() / class_totals

print("[INFO]X : Datatype - {} Minimum Value - {} Maximum Value - {} Shape - {}".format(X.dtype, X.min(), X.max(), X.shape))
print("[INFO]Y : Datatype - {} Minimum Value - {} Maximum Value - {} Shape - {}".format(y.dtype, y.min(), y.max(), y.shape))

# Setup the network
nb_filters = 32
nb_pool = 2
nb_conv = 3

model = Sequential()

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation = 'relu', input_shape = X.shape[1:]))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation = 'relu'))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

# Training the model
validation_split = 0.10
model.fit(X, y, batch_size = 64, class_weight = class_weight, epochs = 15, verbose = 1, validation_split = validation_split)

open('model.json', 'w').write(model.to_json())
model.save_weights('weights.h5')

# Visualising the loss and accuracy plots
plt.plot(model.model.history.history['loss']) 
plt.plot(model.model.history.history['acc'])
plt.plot(model.model.history.history['val_loss'])
plt.plot(model.model.history.history['val_acc'])
plt.show()

# Find the ROC score of the model
n_validation = int(len(X) * validation_split)
y_predicted = model.predict(X[-n_validation:])

print("[INFO] ROC AUC SCORE - {}".format(roc_auc_score(y[-n_validation:], y_predicted)))