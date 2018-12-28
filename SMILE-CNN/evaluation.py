import keras 
from keras.models import model_from_json

model = model_from_json(open('model.json').read())
model.load_weights('weights.h5')