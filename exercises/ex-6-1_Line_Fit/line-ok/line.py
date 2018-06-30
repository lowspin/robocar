import time
import cv2
import numpy as np
import imageio 

TESTMODE = 0

IMG_WIDTH = 128
IMG_HEIGHT = 96

if (TESTMODE == 1): # use live collected images    
    # empty numpy array to hold image
    imgbuffer = np.empty((IMG_WIDTH*IMG_HEIGHT*3,), dtype=np.uint8)

    # get image and perform perspective transform
    with picamera.PiCamera() as camera:
        camera.resolution = (IMG_WIDTH, IMG_HEIGHT)
        camera.capture(imgbuffer,'bgr')
else: # use pre-saved images
    # open image
    imgfile = "frame1.png"
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
img_warped = warp(img)
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

edge = cv2.Canny(gray,10,20)
cv2.imwrite('04edge.jpg',edge)

gray01 = np.float32(gray)
corners = cv2.cornerHarris(gray01,2,3,0.1)
cv2.imwrite('04corners.jpg',corners)
#-------------------------------------------#

############ Interpolation ##################
# pick points for interpolation
def pickpoints(img_bin,minx=0,miny=0,maxx=999999,maxy=999999):
    nz = np.nonzero(img_bin)
    all_x = nz[1]
    all_y = nz[0]
    pts_x = []
    pts_y = []
    for x,y in zip(all_x,all_y):
        if (x>=minx and x<=maxx and y>=miny and y<=maxy):
            pts_x.append(x)
            pts_y.append(y)
    return pts_x, pts_y

pts_x, pts_y = pickpoints(img_bin)

# fit polynomial
z = np.polyfit(pts_y, pts_x, 1)
p = np.poly1d(z)
#print z
#print p(IMG_HEIGHT)

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
cv2.imwrite('05line.jpg',colorimg)
#-------------------------------------------#

