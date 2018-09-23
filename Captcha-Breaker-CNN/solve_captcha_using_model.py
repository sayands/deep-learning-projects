# Import Necessary Packages and Libraries
from keras.models import load_model
from helpers import resize_to_fit
from imutils import paths 
import numpy as np 
import imutils
import cv2
import pickle

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"

# load the model labels
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained network
model = load_model(MODEL_FILENAME)

# grab a random image to test
captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size = (10, ), replace = False)

# loop over the image paths
for image_file in captcha_image_files:
    # Load the image and convert it to grayscale
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add some extra padding around the image
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    # threshold the image - convert to pure black and white
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours(continous blobs of pixels) in the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w/h > 1.25:
            half_width = int(w/2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))

        else:
            letter_image_regions.append((x, y, w, h))
    
    if len(letter_image_regions) != 4:
        continue
    
    # Sorted the detected letter images based on the x-coordinate
    letter_image_regions = sorted(letter_image_regions, key = lambda x : x[0])

    # Create an output image and a list to hold our predicted letters
    output = cv2.merge([image] * 3)
    predictions = []

    # loop over the letters
    for letter_bounding_box in letter_image_regions:
        x, y, w, h = letter_bounding_box

        letter_image = image[y-2:y+h+2, x-2:x+w+2]
        letter_image = resize_to_fit(letter_image, 20, 20)

        # Turn the single image into a 4d list of images
        letter_image = np.expand_dims(letter_image, axis = 2)
        letter_image = np.expand_dims(letter_image, axis = 0)

        prediction = model.predict(letter_image)

        # Convert the one-hot encoding to make a normal letter
        letter = lb.inverse_transform(prediction)[0]
        predictions.append(letter)

        # draw the output on the image
        cv2.rectangle(output, (x-2, y-2), (x+w+4, y+h+4), (0, 255, 0), 1)
        cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    
    # Print the captcha's text
    captcha_text = "".join(predictions)
    print("CAPTCHA text is: {}".format(captcha_text))

    # Show the annotated image
    cv2.imshow("Output", output)
    cv2.waitKey(0)

