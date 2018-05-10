#!/usr/bin/env python

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import imagefunctions 
import rospy
from geometry_msgs.msg import Twist, TwistStamped

IMG_WIDTH = 320
IMG_HEIGHT = 240

class CamNode(object):
	
	def __init__(self):
		# initialize the camera and grab a reference to the raw camera capture
		self.camera = PiCamera()
		self.camera.resolution = (IMG_WIDTH, IMG_HEIGHT)
		self.camera.framerate = 32
		self.rawCapture = PiRGBArray(self.camera, size=(640, 480))
		self.max_linear_vel = 0.2;
		self.max_angular_vel = 1.5707;

		# allow the camera to warmup
		time.sleep(0.1)

		# spin up ros
		rospy.init_node('cam_control_node')
		self.pub = rospy.Publisher('driver_node/cmd_vel', Twist, queue_size=1)
		#self.pubStamped = rospy.Publisher('cmd_vel_stamp', TwistStamped, queue_size=10)
	
		self.loop()

	def loop(self): 
                dt = 0.2
		rate = rospy.Rate(1/dt)
                imgcapture = np.empty((IMG_WIDTH*IMG_HEIGHT*3,), dtype=np.uint8)
		while not rospy.is_shutdown():
			# capture one frame
			self.camera.capture(imgcapture, 'bgr')
			image = imgcapture.reshape((IMG_HEIGHT,IMG_WIDTH,3))

			# process frame
                        vel = self.twist_from_frame(image, int((0.8*dt)*1000))

			# publish drive command
	                self.pub.publish(vel)
                        print str(vel.linear.x) + ", " + str(vel.angular.z)

			# clear the stream in preparation for next frame
			#self.rawCapture.truncate(0) 

        def twist_from_frame(self, image, showtime=0):
		# extract pixels
                #img_bin = imagefunctions.extractpixels(image)
                #out_img = np.dstack((img_bin,img_bin,img_bin))*255

                # show the frame on screen
                out_img = image
                cv2.imshow("Frame",out_img)
                cv2.waitKey(showtime) # required to show image

                # Twist Command
                #ss, xshift = imagefunctions.find_white_patch(img_bin)
			
                cte = 0.0
		#cte = -xshift/IMG_WIDTH
#		print xshift, cte
		vel = Twist()
		vel.linear.x = 0.5*self.max_linear_vel
		vel.angular.z = -0.5*self.max_angular_vel*cte

		# return
                return vel 
 

if __name__ == '__main__':
	try:
		CamNode()
	except rospy.ROSInterruptException:
		rospy.logerr('Could not start camcontrol node.')
