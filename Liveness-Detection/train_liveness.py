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

# Encode the labels as integers and one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = np_utils.to_categorical(labels, 2)

# Partition the data into training and testing splits using 75% of 
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

# Construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range = 20, zoom_range = 0.15, width_shift_range = 0.2, 
                        height_shift_range = 0.2, shear_range = 0.15, 
                        horizontal_flip = True, fill_mode = "nearest")

# Initialise the optimizer and model 
print("[INFO] Compiling Model...")
opt = Adam(lr = INIT_LR, decay = INIT_LR / EPOCHS)
model = LivenessNet.build(width = 32, height = 32, depth = 3, classes = len(le.classes_))
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

# Train the network
print("[INFO] Training the network for {} epochs...".format(EPOCHS))
H = model.fit_generator(aug.flow( trainX, trainY, batch_size = BS),
    validation_data = (testX, testY), steps_per_epoch = len(trainX) // BS,
    epochs = EPOCHS)

# Evaluate the network
print("[INFO] Evaluating the network...")
predictions = model.predict(testX, batch_size = BS)
print(classification_report(testY.argmax(axis = 1), 
      predictions.argmax(axis = 1), target_names= le.classes_))

# Save the network to disk
print("[INFO] Serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

# Save the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_acc"], label = "val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])