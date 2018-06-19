import numpy as np
import pickle
from PIL import Image
import glob
import cv2
from scipy.misc import imsave
import ImageFunctions as imagefunctions

import sys

MODEL_PATH = "test_cnn.h5" # TODO
PICKLE_PATH = 'benchmark_data.p'

##### GET DATA #####

# filename = 'vid180611_raw/test01/output=0001.png'

# outputdir = 'vid180611_raw/benchmark/'

# images = []
# labels = []

# imgsize = (128,96)
# windowsize = (50,50)
# slidestep = (10,10)

# startx = 0
# stopx = imgsize[0]-windowsize[0]
# starty = 0
# stopy = imgsize[1]-windowsize[1]

# min_red_pixels = -1
# imgcount=0
# framenum=0

# im=Image.open(filename)
# img = np.array(im) #imnp = list(im.getdata())

# cropnum = 0
# for x in range(startx, stopx, slidestep[0]):
#     for y in range(starty, stopy, slidestep[1]):
#         # edge - crop needs to be full size
#         if (y+windowsize[1]-1>=imgsize[1]):
#             y = imgsize[1]-windowsize[1]
#         if (x+windowsize[0]-1>=imgsize[0]):
#             x = imgsize[0]-windowsize[0]

#         window=((x, y), (x+windowsize[0], y+windowsize[1]))
#         test_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]

#         if (imagefunctions.num_red_pixels(test_img)>min_red_pixels):
#             fname = str(framenum)+'-'+str(cropnum)+'.png'
#             #print test_img.shape
#             imsave(outputdir+fname, test_img)
#             cropnum = cropnum + 1
#             imgcount = imgcount + 1

# print 'extracted ' + str(imgcount) + ' training images'


##### PREP DATA #####


# images = []
# labels = []

# for filename in glob.glob('vid180611_raw/benchmark/*.png'):
#     im=Image.open(filename)
#     imnp = np.array(im)
#     images.append(imnp)
#     labels.append(1)

# alldata = {
#     "images" : images,
#     "labels" : labels
# }

# out_file = 'benchmark_data.p'
# with open(out_file, 'wb') as handle:
#     pickle.dump(alldata, handle, protocol=2)

# print 'saved ' + str(len(alldata['images'])) + ' in ' + out_file







##### CNN ######

import tensorflow.contrib.keras as keras
from sklearn.model_selection import train_test_split

def to_one_hot(labels):
    one_hot = np.zeros((labels.size, labels.max()+1))
    one_hot[np.arange(labels.size),labels] = 1
    return one_hot

# Load pickled data
data_all = pickle.load(open(PICKLE_PATH, 'rb'))

X_test, y_test = data_all['images'], data_all['labels']


X_test = np.array(X_test)
y_test = np.array(y_test)
y_test = to_one_hot(y_test)

model = keras.models.load_model(MODEL_PATH)


##### BENCHMARK MODEL


import time

num_trials = 20

start_time = time.time()
for i in range(num_trials):
    model.predict(X_test)

runtime = time.time() - start_time

print("Total time over " + str(num_trials) + " trials:" + str(runtime))
print("Time per trial: " + str(runtime/num_trials))

