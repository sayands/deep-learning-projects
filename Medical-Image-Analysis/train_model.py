# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the neccessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from utils.resnet import ResNet
from utils import config
from imutils import paths
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
import numpy as np 
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type = str, default = "plot.png", help = "Path to output loss/accuracy plot")
args = vars(ap.parse_args())

# Hyperparameters
NUM_EPOCHS = 50
INIT_LR = 1e-1
BS = 32

def  poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute the new learning rate
    alpha = baseLR * (1- (epoch /float(maxEpochs))) ** power

    return alpha 

# determine the total number of image paths in training, validation and testing directories
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# initialise the training data augmentation object
trainAug = ImageDataGenerator(
    rescale = 1 / 255.0,
    rotation_range = 20,
    zoom_range = 0.05,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    shear_range = 0.05,
    horizontal_flip = True,
    fill_mode = "nearest")

# initialise the validation(and testing) data augmentation object
valAug = ImageDataGenerator(rescale = 1 / 255.0)

# initialise the training generator
trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode = "categorical",
    target_size = (64, 64),
    color_mode = "rgb",
    shuffle = True,
    batch_size = BS)

# initialise the validation generator 
valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode = "categorical",
    target_size = (64, 64),
    color_mode = "rgb",
    shuffle = False,
    batch_size = BS)

# initialise the testing generator
testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode = "categorical",
    target_size = (64, 64),
    color_mode = "rgb",
    shuffle = False,
    batch_size = BS)
