import time
import cv2
import numpy as np
import imageio 

TESTMODE = 0

IMG_WIDTH = 960 #128
IMG_HEIGHT = 540 # 96

if (TESTMODE == 1): # use live collected images    
    # empty numpy array to hold image
    imgbuffer = np.empty((IMG_WIDTH*IMG_HEIGHT*3,), dtype=np.uint8)

    # get image and perform perspective transform
    with picamera.PiCamera() as camera:
        camera.resolution = (IMG_WIDTH, IMG_HEIGHT)
        camera.capture(imgbuffer,'bgr')
else: # use pre-saved images
    # open image
    imgfile = "../frame1.png"
    imgbuffer = imageio.imread(imgfile)
    # crop if necessary
    imgbuffer = imgbuffer[:min(imgbuffer.shape[0],IMG_HEIGHT),:min(imgbuffer.shape[1],IMG_WIDTH),:]

########### Perspective Transform ###########

# perspective transform function
def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
     [[37,24],
      [31,55],
      [83,24],
      [88,55]])
  
    dst = np.float32(
     [[31,24],
      [31,72],
      [95,24],
      [95,72]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

img = imgbuffer.reshape((IMG_HEIGHT,IMG_WIDTH,3))
cv2.imwrite('01original.jpg',cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img_warped = img #warp(img)
cv2.imwrite('02warped.jpg',cv2.cvtColor(img_warped, cv2.COLOR_BGR2RGB))
#-------------------------------------------#

############## Extract Pixels ###############

# thresholding - grayscale
gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
ret,imggray_bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('03darkbin.jpg',imggray_bin)

# thresholding - Saturation channel
hsv = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HSV)
ret, img_bin = cv2.threshold(hsv[:, :, 1], 102, 255, cv2.THRESH_BINARY)
cv2.imwrite('03saturationbin.jpg',img_bin)
#-------------------------------------------#

############ Extract Edges/Corners ##########

####################################################
# TO-DO: 
#     1) Extract edges using openCV and save result in '04edge.jpg')
#     2) Extract corners using openCV and save result in '05corners.jpg')
####################################################


