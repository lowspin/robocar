import numpy as np
import pickle
from PIL import Image
import glob
import cv2
from scipy.misc import imsave
import ImageFunctions as imagefunctions

import sys
vid_name = sys.argv[1]

inputpath = 'vid180611_raw/' + vid_name + '/output=*.png'
outputdir = 'vid180611_50x50/' + vid_name + '/'

images = []
labels = []

imgsize = (128,96)
windowsize = (50,50)
slidestep = (10,10)

startx = 0
stopx = imgsize[0]-windowsize[0]
starty = 0
stopy = imgsize[1]-windowsize[1]

min_red_pixels = 40
imgcount=0
framenum=0
for filename in glob.glob(inputpath):
    im=Image.open(filename)
    img = np.array(im) #imnp = list(im.getdata())

    cropnum = 0
    for x in range(startx, stopx, slidestep[0]):
        for y in range(starty, stopy, slidestep[1]):
            # edge - crop needs to be full size
            if (y+windowsize[1]-1>=imgsize[1]):
                y = imgsize[1]-windowsize[1]
            if (x+windowsize[0]-1>=imgsize[0]):
                x = imgsize[0]-windowsize[0]

            window=((x, y), (x+windowsize[0], y+windowsize[1]))
            test_img = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]

            if (imagefunctions.num_red_pixels(test_img)>min_red_pixels):
                fname = vid_name + '_crop-'+ str(framenum)+'-'+str(cropnum)+'.png'
                #print test_img.shape
                imsave(outputdir+fname, test_img)
                cropnum = cropnum + 1
                imgcount = imgcount + 1

    framenum=framenum+1

print 'extracted ' + str(imgcount) + ' training images'

