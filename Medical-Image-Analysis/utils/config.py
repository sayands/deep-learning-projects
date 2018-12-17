# Import the necessary packages
import os

# initialise the path to the original input directory of image
ORIG_INPUT_DATASET = 'cell_images'

# initialise the base path to the new directory
BASE_PATH = 'malaria'

# derive the training, testing and validation directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define the amount of data to be used for training
TRAIN_SPLIT = 0.8

# amount of validation data inside training data
VAL_SPLIT = 0.1