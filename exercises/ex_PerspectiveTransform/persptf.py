import time
import picamera
import cv2
import numpy as np

IMG_WIDTH = 128
IMG_HEIGHT = 96

# perspective transform function
def warp(img):

  img_size = (img.shape[1], img.shape[0])
  src = np.float32(
     [[43,26],
      [38,58],
      [90,26],
      [96,58]])
  
  dst = np.float32(
     [[31,23],
      [31,71],
      [95,23],
      [95,71]])
 
  M = cv2.getPerspectiveTransform(src,dst)
  warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

  return warped

imgbuffer = np.empty((128*96*3,), dtype=np.uint8)

with picamera.PiCamera() as camera:

    camera.resolution = (IMG_WIDTH, IMG_HEIGHT)

    camera.capture(imgbuffer,'bgr')
    img = imgbuffer.reshape((96,128,3))

    warped = warp(img)
    
    cv2.imwrite('original.jpg',img)
    cv2.imwrite('warped.jpg',warped)
