# Importing the neccessary packages
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import model as md 
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical
import time 

start_time = time()

# loading the data
train = pd.read_csv('train.csv')
print(train.shape)

# Preparing the dataset
X = train.drop("label", axis = 1)
y = train["label"]
print(y.value_counts().to_dict())
y = to_categorical(y, num_classes = 10)
del train

X = X / 255.0
X = X.values.reshape(-1, 28, 28, 1)

# Shuffle Split Train and Test from Original Dataset
seed = 2
train_index, valid_index = ShuffleSplit(n_splits = 1, 
                                        train_size = 0.9,
                                        test_size = None,
                                        random_state = seed).split(X).__next__()

x_train = X[train_index]
Y_train = y[train_index]
x_test = X[valid_index]
Y_test = y[valid_index]

# Parameters
epochs = 10
batch_size = 64
validation_steps = 10000

# Initialize Model, Annealer and Datagen
model, annealer, datagen = md.init_model()

# Start training
train_generator = datagen.flow(x_train, Y_train, batch_size = batch_size)
test_generator = datagen.flow(x_test, Y_test, batch_size = batch_size)

history = model.fit_generator(train_generator, 
                        steps_per_epoch = x_train.shape[0] // batch_size,
                        epochs = epochs,
                        validation_data = test_generator,
                        validation_steps = validation_steps // batch_size,
                        callbacks = [annealer])

score = model.evaluate(x_test, Y_test)
print("Test Accuracy: ", score[1])

# Saving model for future reference
model.save('Digits-1.3.0.h5')
print('Saved Model to Disk')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Acccuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc = 'lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc = 'upper right')
plt.show()


