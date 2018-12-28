# List all the image files and create a big list of examples as (path, class) tuples 
import numpy as np 
import time
from skimage.measure import block_reduce
from skimage.io import imread 
from utils.list_all_files import list_all_files 
from utils.make_mosaic import make_mosaic
from utils.show_array import show_array

negative_paths = list(list_all_files('./dataset/SMILEsmileD-master/SMILEs/negatives/negatives7', ['.jpg']))
print("[INFO]Loaded {} negative examples".format(len(negative_paths)))

positive_paths = list(list_all_files('./dataset/SMILEsmileD-master/SMILEs/positives/positives7', ['.jpg']))
print("[INFO]Loaded {} positive examples".format(len(positive_paths)))

examples = [(path, 0) for path in negative_paths] + [(path, 1) for path in positive_paths]

# Load the image and resize from 64*64 to 32*32 with block_reduce
start = time.time()

def examples_to_dataset(examples, block_size = 2):
    X = []
    y = []
    
    for path, label in examples:
        img = imread(path, as_grey = True)
        img = block_reduce(img, block_size=(block_size, block_size), func = np.mean)
        X.append(img)
        y.append(label)
    
    return np.asarray(X), np.asarray(y)

X, y = examples_to_dataset(examples)

end = time.time()
print("[INFO]Time taken - {}s".format(end - start))

# Turn X and Y into numpy arrays and coerce them into right shape and range
X = X.astype(np.float32) / 255.
y = y.astype(np.int32)

print("[INFO]X : Datatype - {} Minimum Value - {} Maximum Value - {} Shape - {}".format(X.dtype, X.min(), X.max(), X.shape))
print("[INFO]Y : Datatype - {} Minimum Value - {} Maximum Value - {} Shape - {}".format(y.dtype, y.min(), y.max(), y.shape))

# Show negative examples
show_array(a = 255 * make_mosaic(X[:len(negative_paths)], 8), format='png', filename='Neg.png')
# Show positive examples
show_array(a = 255 * make_mosaic(X[-len(positive_paths):], 8), format='png', filename='Pos.png')

# Save data to disk
X = np.expand_dims(X, axis = -1)
np.save('X.npy', X)
np.save('y.npy', y)
print("[INFO}Shape of X after expanding dimensions - {}".format(X.shape))

