# Import the neccessary packages
from keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
from utils import datasets
from utils import models 
import numpy as np 
import argparse
import locale
import os 

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type = str, required = True, help = "Path to input dataset of house images")
args = vars(ap.parse_args())

print("[INFO] Loading House Atrributes...")
inputPath = os.path.sep.join([args["dataset"], "HousesInfo.txt"])
df = datasets.load_house_attributes(inputPath)

# Construct train test split 
print("[INFO]Constructing Train/Test Split...")
(train, test) = train_test_split(df, test_size = 0.25, random_state = 42)

# Find the largest House Price in the training set and use it 
# to scale prices in the range [0, 1]
maxPrice = train["price"].max()
trainY = train["price"] / maxPrice
testY = test["price"] / maxPrice

print("[INFO]Processing Data...")
(trainX, testX) = datasets.process_house_attributes(df, train, test)

# Create our MLP 
model = models.create_mlp(trainX.shape[1], regress = True)
opt = Adam(lr = 1e-3, decay = 1e-3 / 200)
model.compile(loss = "mean_absolute_percentage_error", optimizer = opt)

# train the model
print('[INFO] training model...')
model.fit(trainX, trainY, validation_data=(testX, testY), epochs = 200, batch_size=2)

# make predictions on the test set
print("[INFO] Predicting House Prices...")
preds = model.predict(testX)

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100 
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] Avg. House Price: {}, STD House Pice: {}".format(
        locale.currency(df["price"].mean(), grouping=True), 
        locale.currency(df["price"].std(), grouping=True)))

print("[INFO] Mean : {:.2f}%, std: {:.2f}%".format(mean, std))