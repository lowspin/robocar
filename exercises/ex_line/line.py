import time
import picamera
import cv2
import numpy as np

IMG_WIDTH = 128
IMG_HEIGHT = 96

########################################
# perspective transform function
def warp(img):
  img_size = (img.shape[1], img.shape[0])
  src = np.float32(
     [[36,26],
      [31,58],
      [83,26],
      [89,58]])
  dst = np.float32(
     [[31,23],
      [31,71],
      [95,23],
      [95,71]])
  M = cv2.getPerspectiveTransform(src,dst)
  warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
  return warped

# pick points for interpolation
def pickpoints(img_bin):
  nz = np.nonzero(img_bin)
  pts_x = nz[1]
  pts_y = nz[0]
  return pts_x, pts_y

def pickpoints2(img_bin):
  pts_x = []
  pts_y = []
  # work in progress
  return pts_x, pts_y

#######################################

# empty numpy array to hold image
imgbuffer = np.empty((128*96*3,), dtype=np.uint8)

# get image and perform perspective transform
with picamera.PiCamera() as camera:
    camera.resolution = (IMG_WIDTH, IMG_HEIGHT)
    camera.capture(imgbuffer,'bgr')
    img = imgbuffer.reshape((96,128,3))
    img_warped = warp(img)
    cv2.imwrite('01original.jpg',img)
    cv2.imwrite('02warped.jpg',img_warped)

# convert to grayscale
gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

# thresholding
ret,img_bin = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imwrite('03bin.jpg',img_bin)

# pick points for interpolation
pts_x, pts_y = pickpoints(img_bin)

# fit polynomial
z = np.polyfit(pts_y, pts_x, 1)
p = np.poly1d(z)
print z
print p(IMG_HEIGHT)

# generate plot coordinates
numpts = 100
min_domain = min(pts_y)
max_domain = max(pts_y)
ploty = np.linspace(min_domain, max_domain, numpts)
plotx = p(ploty)
pts = np.stack((plotx,ploty))
pts = np.transpose(pts)
pts = pts.reshape((-1,1,2))
ptsplot = pts.astype(int)

# plot line on image
colorimg = cv2.cvtColor(img_bin,cv2.COLOR_GRAY2RGB)
cv2.polylines(colorimg,[ptsplot],False,(0,255,0))
cv2.line(colorimg,(int(IMG_WIDTH/2),IMG_HEIGHT-1),(int(IMG_WIDTH/2),int(IMG_HEIGHT/2)),(0,0,255),1)

# output
cv2.imwrite('04line.jpg',colorimg)
cv2.imshow("Frame",colorimg)
cv2.waitKey(0)
