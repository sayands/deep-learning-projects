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

# initialise our ResNet model and compile it
model = ResNet.build(64, 64, 3, 2, (3, 4, 6), (64, 128, 256, 512), reg = 0.0005)
opt = SGD(lr = INIT_LR, momentum = 0.9)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

# define our set of callbacks and fit the model
callbacks = [LearningRateScheduler(poly_decay)]

H = model.fit_generator(
    trainGen,
    steps_per_epoch = totalTrain // BS,
    validation_data = valGen,
    validation_steps = totalVal // BS,
    epochs = NUM_EPOCHS,
    callbacks = callbacks)

# reset the testing generator and use the trained model to
# make predictions on the data
print("[INFO] Evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen, steps = (totalTest // BS) + 1)

# find index of label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis = 1)

# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs, target_names = testGen.class_indices.keys()))

# Plot the training loss and accuracy 
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig(args["plot"])