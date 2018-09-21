# Importing the neccessary packages
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import model as md 
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical
import time 

start_time = time.time()

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

# Train Test Split
random_seed = 2
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.1, random_state=random_seed)

# Parameters
epochs = 30
batch_size = 86

# Initialize Model, Annealer and Datagen
model, annealer, datagen = md.init_model()

# Start training
datagen.fit(X_train)

# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[annealer])


score = model.evaluate(X_val, Y_val)
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

# Model Predicition
test = pd.read_csv('test.csv')
print(test.shape)
test = test / 255
test = test.values.reshape(-1, 28, 28, 1)
p = np.argmax(model.predict(test), axis = 1)

# Base Model Scores
print('Base Model Scores')
valid_loss, valid_acc = model.evaluate(X_val, Y_val, verbose = 0)
valid_p = np.argmax(model.predict(X_val), axis = 1)
target = np.argmax(Y_val, axis = 1)
cm = confusion_matrix(target, valid_p)
print(cm)

# Preparing for submission
submission = pd.DataFrame(pd.Series(range(1, p.shape[0] + 1), name = 'ImageId'))
submission['label'] = p
filename = 'keras-cnn-{0}.csv'.format(str(int(score[1] * 10000)))
submission.to_csv(filename, index = False)

elapsed_time = time.time() - start_time
print("Elapsed time: {0}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
