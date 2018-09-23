# Importing packages
import imutils
import cv2 

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    """
    (h, w) = image.shape[:2]

    if w>h:
        image = imutils.resize(image, width = width)

    else:
        image = imutils.resize(image, height = height)

    # determine the padding values for width and height to obtain target dimensions
    padW = int((width - image.shape[1])/ 2.0)
    padH = int((height - image.shape[0])/ 2.0)

    # pad the image, then apply one more resizing to handle any rounding issues
    image = cv2.cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image