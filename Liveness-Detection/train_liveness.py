# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Importing the neccessary packages
from utils.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils 
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import pickle
import cv2 
import os 

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to input dataset")
ap.add_argument("-m", "--model", type = str, required = True, help = "Path to trained model")
ap.add_argument("-l", "--le", type = str, required = True, help = "Path to label encoder")
ap.add_argument("-p", "--plot", type = str, default = "plot.png", help = "Path to output loss/accuracy plot")

args = vars(ap.parse_args())

# Initialise the initial learning rate, batch size and number of epochs to train for
INIT_LR = 1e-4
BS = 8
EPOCHS = 50

# grab the list of images in our dataset directory, then initialise the
# list of data and class images
print("[INFO] Loading Images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    # extract the class label from filename, load the image and resize
    # it to be a fixed 96*96 pixels, ignoring aspect ratio
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))

    # Update the data and label lists, respectively
    data.append(image)
    labels.append(label)

# Convert the data to a numpy array, then preprocess it by scaling all
# pixel intensities to the range [0, 1]
data = np.array(data, dtype = "float") / 255.0