#!/usr/bin/env python

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Configuration
IMG_WIDTH = 128
IMG_HEIGHT = 96

CAMNODE_DT = 0.1
FRAMERATE = 50

# Main Camera Node Class
class CamNode(object):

    def __init__(self):

        # initialize camera
        self.camera = PiCamera()

        # Use capture parameters consistent with camera mode 6, which doesn't crop.
        self.camera.resolution = (IMG_WIDTH, IMG_HEIGHT)
        self.camera.framerate = FRAMERATE
        self.camera.sensor_mode = 6

        # allow the camera to warmup
        time.sleep(1.0)

        # spin up ros
        rospy.init_node('cam_control_node')
        self.image_pub = rospy.Publisher('camera/image', Image, queue_size=1)
        self.bridge = CvBridge()

        self.loop()

    def loop(self):

        dt = CAMNODE_DT
        rate = rospy.Rate(1/dt)
        imgcapture = np.zeros((IMG_WIDTH*IMG_HEIGHT*3,), dtype=np.uint8)

        while not rospy.is_shutdown():

            self.camera.capture(imgcapture, 'bgr', use_video_port=True, resize=(IMG_WIDTH, IMG_HEIGHT))
            img = imgcapture.reshape(IMG_HEIGHT, IMG_WIDTH, 3)

            # publish robot's view
            try:
                imgmsg = self.bridge.cv2_to_imgmsg(img, "bgr8")
                imgmsg.header.stamp = rospy.get_rostime()
                self.image_pub.publish(imgmsg)
            except CvBridgeError as e:
                print(e)

            rate.sleep()


if __name__ == '__main__':
    try:
        CamNode()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start camcontrol node.')
