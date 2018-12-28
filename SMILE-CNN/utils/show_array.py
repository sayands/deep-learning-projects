import numpy as np 
import cv2

def show_array(a, format='png', filename = None):
    a = np.squeeze(a)
    a = np.uint8(np.clip(a, 0, 255))
    if filename is not None:
        cv2.imwrite(filename, a)
    cv2.imshow(str(filename), a)
    cv2.waitKey(0)