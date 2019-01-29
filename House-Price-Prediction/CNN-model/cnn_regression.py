# Import the neccessary packages
from keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
from utils import datasets, model 
import numpy as np 
import argparse
import locale
import os 

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type = str, required = True, 
                help = "Path to input dataset of house images")

args = vars(ap.parse_args())

# Construct the path to input txt file 
print("[INFO] Loading House Attributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

# Load the house images and then scale the pixel intensities to range [0, 1]
print("[INFO]Loading House Images...")
images = datasets.load_house_images(df, args["dataset"])
images = images / 255.0 

# Split the dataset
split = train_test_split(df, images, test_size = 0.25, random_state = 42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split 

# Find the largest house price in the training set and use
# it to scale our prices in the range [0, 1]
maxPrice = trainAttrX["price"].max()
trainY = trainAttrX["price"] / maxPrice
testY = testAttrX["price"] / maxPrice

# Create the CNN and compile the model
model = model.create_cnn(64, 64, 3, regress=True)
opt = Adam(lr = 1e-3, decay = 1e-3 / 200)
model.compile(loss = "mean_absolute_percentage_error", optimizer=opt)

# train the model 
print("[INFO] training model...")
model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY), epochs = 200, batch_size=8)

# Make predictions on testing data
print("[INFO] predicting house prices...")
preds = model.predict(testImagesX)

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100 
absPercentDiff = np.abs(percentDiff)

# Compute the mean and standard deviation of the absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# Show some statistics on our model 
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
    locale.currency(df["price"].mean(), grouping = True),
    locale.currency(df["price"].std(), grouping = True)))

print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))