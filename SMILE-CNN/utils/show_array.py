from cStringIO import cStringIO
import numpy as np 
import PIL.Image 
import IPython.display
import shutil 

def show_array(a, format='png', filename = None):
    a = np.squeeze(a)
    a = np.uint8(np.clip(a, 0, 255))
    image_data = StringIO()
    PIL.Image.fromarray(a).save(image_data, format)
    if filename is None:
        IPython.display.display(IPython.display.Image(data = image_data.get_value()))
    else:
        with open(filename, 'w') as f:
            image_data.seek(0)
            shutil.copyfileobj(image_data, f)
