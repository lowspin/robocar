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
     [[37,24],
      [31,55],
      [83,24],
      [88,55]])
  
  dst = np.float32(
     [[31,24],
      [31,72],
      [95,24],
      [95,72]])
 
  M = cv2.getPerspectiveTransform(src,dst)
  warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

  return warped

imgbuffer = np.empty((IMG_WIDTH*IMG_HEIGHT*3,), dtype=np.uint8)

with picamera.PiCamera() as camera:

    camera.resolution = (IMG_WIDTH, IMG_HEIGHT)

    print 'position calibration sheet'
    camera.start_preview()
    time.sleep(10)
    camera.stop_preview()

    camera.capture(imgbuffer,'bgr')
    img = imgbuffer.reshape((IMG_HEIGHT,IMG_WIDTH,3))

    warped = warp(img)
    
    cv2.imwrite('original.jpg',img)
    cv2.imwrite('warped.jpg',warped)

    print 'images updated'
