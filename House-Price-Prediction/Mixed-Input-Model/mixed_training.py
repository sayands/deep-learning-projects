# Import the neccessary packages and libraries
from utils import datasets
from utils import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense 
from keras.models import Model 
from keras.optimizers import Adam 
from keras.layers import concatenate
import numpy as np 
import argparse
import locale 
import os 

# Construct the argument parser and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type = str, required = True, help = 'Path to input dataset of house images')
args = vars(ap.parse_args())

# Construct the path to the input .txt file that contains information on
# each house and then load the dataset
print("[INFO] Loading House Attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

# Load the house images and scale the pixel intensities to the 
# range [0, 1]
print("[INFO] Loading House Images...")
images = datasets.load_house_images(df, args["dataset"])
images = images / 255.0

# Partition the data into training and testing splits using 75% for training
print("[INFO] Processing data...")
split = train_test_split(df, images, test_size = 0.25, random_state = 42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# Find the largest house price in the training set and use it to scale
# our house prices to the range [0, 1]
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

(trainAttrX, testAttrX) = datasets.process_house_attributes(df, trainAttrX, testAttrX)

# Create the MLP and CNN models
mlp = models.create_mlp(trainAttrX.shape[1], regress = False)
cnn = models.create_cnn(64, 64, 3, regress = False)

# Create the input to our final set of layers as the output of both MLP and CNN
combinedInput = concatenate([ mlp.output, cnn.output])

# Our Final FC layer head will have two dense layers
x = Dense(4, activation = "relu")(combinedInput)
x = Dense(1, activation = "linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs = x)

opt = Adam(lr = 1e-3, decay = 1e-3 / 200)
model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

# train the model 
print("[INFO] Training model...")
model.fit([ trainAttrX, trainImagesX], trainY, validation_data = ([ testAttrX, testImagesX], testY), 
            epochs = 200, batch_size = 8)

# Make predictions on test data
print("[INFO]Predicting House Prices...")
preds = model.predict([testAttrX, testImagesX])

# Compute the differences between predicted house prices and the actual house prices
diff = preds.flatten() - testY 
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# Compute the mean and standard deviation of absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# Show some statistics on the model 
locale.setlocale(locale.LC_ALL, "en-UTF-8")
print("[INFO] Avg. House Price: {}, Std House Price: {}".format(
    locale.currency(df["price"].mean(), grouping = True),
    locale.currency(df["price"].std(), grouping = True)))

print("[INFO] Mean: {:.2f}%, std: {:.2f}%".format(mean, std))