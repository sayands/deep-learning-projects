# Import the neccessary packages and libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np 
import glob
import cv2 
import os 

def load_house_attributes(inputPath):
    # initialise the list of columnn names in the CSV file and then load it using Pandas
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep = " ", header = None, names = cols)

    # determine - unique zip codes and no.of data points
    # with each zip code 
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    # loop over each of the unique zip codes and their corresponding count
    for (zipcode, count) in zip(zipcodes, counts):
        # removing imbalance
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace = True)
    
    # return the data frame
    return df

def load_house_images(df, inputPath):
    # initialise our images array
    images = []

    # loop over the indexes of the houses
    for i in df.index.values:
        # find the four images for the house and sort the file paths
        basePath = os.path.sep.join([inputPath, "{}_*".format(i+1)])
        housePaths = sorted(list(glob.glob(basePath)))

        # initialise our list of input images along with output image
        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype = "uint8")

        # loop over the input house paths
        for housePath in housePaths:
            # load the input image, resize to 32 * 32 and then update list
            image = cv2.imread(housePath)
            image = cv2.resize(image, (32, 32))
            inputImages.append(image)

        # build the montage
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, 0:32] = inputImages[3]

        # add the tiled image to our set of training images
        images.append(outputImage)

    return np.array(images)
