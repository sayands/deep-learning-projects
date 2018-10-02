# Google Colab Setup Code for Mounting Drive
from google.colab import drive
# This will prompt for authorization.
drive.mount('/content/drive)

# Importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Reading the data
train_df = pd.read_csv('data/fashion-mnist_train.csv')
test_df = pd.read_csv('data/fashion-mnist_test.csv')
print(train_df.head())

# Splitting the training and test data into X(image) and Y(label) arrays
train_data = np.array(train_df, dtype = 'float32')
test_data = np.array(test_df, dtype = 'float32')

X_train = train_data[:, 1:] / 255
y_train = train_data[:, 0]

X_test = test_data[:, 1:] / 255
y_test = test_data[:, 0]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 2)

# Visualising a random image
image = X_train[50, :].reshape((28, 28))

plt.imshow(image)
plt.show()

# Reshaping the data
im_rows = 28
im_cols = 28
im_shape = (im_rows, im_cols, 1)

X_train = X_train.reshape(X_train.shape[0], *im_shape)
X_test = X_test.reshape(X_test.shape[0], *im_shape)
X_val  = X_val.reshape(X_val.shape[0], *im_shape)


# Importing the keras packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam

# Defining the Model
model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation = 'relu', input_shape = im_shape, kernel_initializer = 'he_normal', name = 'Conv2D-1'))
model.add(MaxPooling2D(pool_size = 2, name = 'Maxpool'))
model.add(Dropout(0.25, name = 'Dropout-1'))

model.add(Conv2D(64, kernel_size = 3, activation = 'relu', name = 'Conv2D-2'))
model.add(Dropout(0.25, name = 'Dropout-2'))

model.add(Conv2D(128, kernel_size = 3, activation = 'relu', name = 'Conv2D-3'))
model.add(Dropout(0.4, name = 'Dropout-3'))

model.add(Flatten(name = 'Flatten'))
model.add(Dense(128, activation = 'relu', name = 'Dense'))
model.add(Dropout(0.4, name = 'Dropout'))
model.add(Dense(10, activation = 'softmax', name = 'Output'))


print(model.summary())

# Hyperparameters
batch_size = 512
epochs = 50

# Compiling The Data
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_val, y_val))


# Finding model performance on test set
print(model.evaluate(X_test, y_test))

# Summarizing history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Acccuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'])
#plt.show()

# Saving the accuracy plot
plt.savefig('Accuracy.png')

