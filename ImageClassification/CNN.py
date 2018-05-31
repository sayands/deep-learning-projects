# Convolutional Neural Networks

# Part I -Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape = (128, 128, 3), activation = 'relu'))

# Setp 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd - Conv + Pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd - Conv + Pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Fully Connected Layer
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(rate = 0.5))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part II - Fitting the CNN to the image
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '/my_data/dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '/my_data/dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000/32,
        epochs=25,
        validation_data = test_set,
        validation_steps=2000/32)

# Making New Predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/my_data/dataset/single_prediction/cat_or_dog_2.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)


