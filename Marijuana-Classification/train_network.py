# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Import the necessary packages
from simplenet import SimpleNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras.regularizers import l2 
from keras.utils import np_utils 
from imutils import build_montages
from imutils import paths 
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
import cv2 
import os 

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to input dataset")
ap.add_argument("-e", "--epochs", type = int, default = 100, help ="No.of epochs to train our network for")
ap.add_argument("-p","--plot", type = str, default = "plot.png", help = "Path to output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the list of images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args['dataset']))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from filename
    label = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))

    # update the data and label lists,respectively
    data.append(image)
    labels.append(label)

data = np.array(data, dtype = 'float') / 255.0
data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))

le = LabelEncoder()
labels = le.fit_transform(labels)

labels = np_utils.to_categorical(labels, 2)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.40, stratify = labels, random_state = 42)

# Initialise the optimiser and model
print("[INFO] Compiling model...")
opt = Adam(lr = 1e-4, decay = 1e-4 / args["epochs"])
model = SimpleNet.build(width = 64, height = 64, depth = 1, classes = len(le.classes_), reg = l2(0.0002))
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])

# Train the network
print('[INFO] training network for {}  epochs...'.format(args['epochs']))
H = model.fit(trainX, trainY, validation_data = (testX, testY), batch_size = 32, epochs = args['epochs'], verbose = 1)

# Evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size = 32)
print(classification_report(testY, argmax(axis = 1), predictions.argmax(axis = 1), target_names = le.classes_))

# Plot the training loss and accuracy
N = args['epochs']
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label = 'train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label = 'val_loss')
plt.plot(np.arange(0, N), H.history['acc'], label = 'train_acc')
plt.plot(np.arange(0, N), H.history['val_acc'], label = 'val_acc')
plt.title('Training Loss and Accuracy on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc = 'lower left')
plt.savefig(args['plot'])

# Randomly select a few test images and then initialize the output 
idxs = np.arange(0, testY.shape[0])
idxs = np.random.choice(idxs, size (25, ), replace = False)
images = []

# loop over the testing indexes
for i in idxs:
    image = np.expand_dims(testX[i], axis = 0)
    preds = model.predict(image)
    j = preds.argmax(axis = 1)[0]
    label = le.classes_

    output = (image[0] * 255).astype("uint8")
    output = np.dstack([output] * 3)
    output = cv2.resize(output, (128, 128))

    label_color = (0, 0, 255) if "non" in label else (0, 255, 0)
    cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
    images.append(output)

montage = build_montages(images, (128, 128), (5, 5))[0]

# show the output montage
cv2.imshow("Output", montage)
cv2.waitKey(0)