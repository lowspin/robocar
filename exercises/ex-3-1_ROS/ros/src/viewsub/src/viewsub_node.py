#!/usr/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
import time

class ViewSub():

    def __init__(self):

        # ROS nodes
        rospy.init_node('viewsub', anonymous=True)
        rospy.Subscriber("camera/image", Image, self.callback, queue_size=1)

        # start
        self.dt = 0.1
        self.img = np.empty(1)
        self.loop()

    def callback(self,rosimg):

        self.img = CvBridge().imgmsg_to_cv2(rosimg)
        self.lastimgtime = rosimg.header.stamp.secs

    def loop(self):

        rate = rospy.Rate(1/self.dt)
        time.sleep(0.1)

        while not rospy.is_shutdown():
            if(self.img.ndim == 3):
                cv2.imshow("Frame",self.img)
                cv2.waitKey(100)
        
        rate.sleep()

if __name__ == '__main__':
    ViewSub()
