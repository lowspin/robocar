import cv2
import numpy as np
import picamera

SCALE = 10
IMG_WIDTH = SCALE*128
IMG_HEIGHT = SCALE*96

########################################
# perspective transform function
def warp(img):
  img_size = (img.shape[1], img.shape[0])
  src = SCALE*np.float32(
     [[36,26],
      [31,58],
      [83,26],
      [89,58]])
  dst = SCALE*np.float32(
     [[31,23],
      [31,71],
      [95,23],
      [95,71]])
  M = cv2.getPerspectiveTransform(src,dst)
  warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
  return warped
#######################################

# number of corners
nx = 9
ny = 7

# empty numpy array to hold image
imgbuffer = np.empty((IMG_HEIGHT*IMG_WIDTH*3,), dtype=np.uint8)

# get image and perform perspective transform
with picamera.PiCamera() as camera:
    camera.resolution = (IMG_WIDTH, IMG_HEIGHT)
    camera.capture(imgbuffer,'bgr')
    img = imgbuffer.reshape((IMG_HEIGHT,IMG_WIDTH,3))
    cv2.imwrite('00original.jpg',img)
    img_warped = warp(img)
    cv2.imwrite('01warped.jpg',img_warped)

    # convert to grayscale
    img_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

    # Find corners
    ret, corners = cv2.findChessboardCorners(img_gray, (nx,ny), None)

    if ret == True:
        # draw corners
        cv2.drawChessboardCorners(img_warped, (nx,ny), corners, ret)
        cv2.imwrite('02corners.jpg',img_warped)

        # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)
