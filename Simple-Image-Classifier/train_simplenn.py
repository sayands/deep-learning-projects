# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt 
import numpy as np 
import argparse 
import random 
import pickle
import cv2
import os 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to input dataset of images")
ap.add_argument("-m", "--model", required = True, help = "Path to output trained model")
ap.add_argument("-l", "--label-bin", required = True, help = "Path to output label binarizer")
ap.add_argument("-p", "--plot", required = True, help = "Path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialise the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype = "float")/ 255.0
labels = np.array(labels)

# perform train and test splits
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.25, random_state = 42)

# converting the labels
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# define the model architecture
model = Sequential()
model.add(Dense(1024, input_shape = (3072, ), activation = 'sigmoid'))
model.add(Dense(512, activation = 'sigmoid'))
model.add(Dense(len(lb.classes_), activation = 'softmax'))

# initialize our learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 75

print("[INFO] training network...")
opt = SGD(lr = INIT_LR)
model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

# train the network
H = model.fit(trainX, trainY, validation_data = (testX, testY), epochs = EPOCHS, batch_size = 32)

# evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY.argmax(axis = 1), predictions.argmax(axis = 1), target_names = lb.classes_))

# plot training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use('ggplot')
plt.figure()
plt.plot(N, H.history["loss"], label = 'train_loss')
plt.plot(N, H.history['val_loss'], label = 'val_loss')
plt.plot(N, H.history['acc'], label = 'train_acc')
plt.plot(N, H.history['val_acc'], label = 'val_acc')
plt.title('Training loss and Accuracy(Simple NN)')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(args["plot"])


# save the model and label binarizer to disk
print('[INFO] serializing network and label binarizer...')
model.save(args['model'])
f = open(args['label_bin'], "wb")
f.write(pickle.dumps(lb))
f.close()