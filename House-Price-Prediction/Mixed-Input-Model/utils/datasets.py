# Importing the neccessary packages and libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np 
import glob
import cv2 
import os 

def load_house_attributes(inputPath):
    # Initialise the list of column names in the CSV file and load using pandas
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep = " ", header = None, names = cols)

    # determine the unique zip codes and no.of data points
    # with each zip code
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    # loop over each of unique zip codes and their corresponding count
    for (zipcode, count) in zip(zipcodes, counts):
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace = True)
    
    # return the data frame
    return df 

def process_house_attributes(df, train, test):
    # initialise the column names of the continuous data 
    continuous = ["bedrooms", "bathrooms", "area"]

    # perform min-max scaling each continuouse feature column to
    # the range [0, 1]
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])

    # one-hot encode the zip code categorical data
    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"])

    # Construct our final training and testing data
    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])

    return (trainX, testX)

def load_house_images(df, inputPath):
    # initialise our images array
    images = []

    # loop over the indexes of the houses
    for i in df.index.values:
        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        housePaths = sorted(list(glob.glob(basePath)))

        # initialise our list of input images along with the output image
        # after combining the four input images
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype = "uint8")

        # loop over the input house paths
        for housePath in housePaths:
            # load the input image, resize it to be 32*32, and then update the list of images
            image = cv2.imread(housePath)
            image = cv2.resize(image, (32, 32))
            inputImages.append(image)
        
        # title the four input images in the output image
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, 0:32] = inputImages[3]

        # add the tiled image to our set of images the network will be trained on
        images.append(outputImage)
    
    # return our set of images
    return np.array(images)