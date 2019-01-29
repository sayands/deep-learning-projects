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

def process_house_attributes(df, train, test):
    # initialise the column names of the continuous data
    continuous = ["bedrooms", "bathrooms", "area"]

    # perform min-max scaling
    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])

    # one hot encode the zip code categorical data
    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"])

    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical, testContinuous])

    return (trainX, testX)