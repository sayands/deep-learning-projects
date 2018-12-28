import keras 
import numpy as np 
import cv2 
from keras.models import model_from_json
from utils.show_array import show_array
from utils.crop_and_resize import crop_and_resize

model = model_from_json(open('model.json').read())
model.load_weights('weights.h5')

# Evaluate a single image and print a corresponding indicator
def print_indicator(data, model, class_names, bar_width = 50):
    probabilities = model.predict(np.array([data]))[0]
    print("Probalities - {}".format(probabilities))
    left_count = int(probabilities[1] * bar_width)
    right_count = bar_width - left_count
    left_side = '-' * left_count
    right_side = '-' * right_count
    
    print(class_names[0], left_side + '###' + right_side, class_names[1])

X = np.load('X.npy')
class_names = ['Neutral', 'Smiling']
img = X[90]
show_array(255 * img)
print_indicator(img, model, class_names)