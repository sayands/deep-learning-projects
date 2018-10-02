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
train_df = pd.read_csv('/content/drive/My Drive/app/Fashion-MNIST/fashion-mnist_train.csv')
test_df = pd.read_csv('/content/drive/My Drive/app/Fashion-MNIST/fashion-mnist_test.csv')
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
batch_size = 512
im_shape = (im_rows, im_cols, 1)

X_train = X_train.reshape(X_train.shape[0], *im_shape)
X_test = X_test.reshape(X_test.shape[0], *im_shape)
X_val  = X_val.reshape(X_val.shape[0], *im_shape)