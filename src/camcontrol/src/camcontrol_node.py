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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from pid import PID

# Params - to be moved to launch file later
IMG_WIDTH = 128
IMG_HEIGHT = 96
MAX_THROTTLE_GAIN = 0.06
MAX_STEER_GAIN = 1.5707
FRAMERATE = 32
PID_Kp = 1.0
PID_Ki = 0.05
PID_Kd = 20.0

class CamNode(object):
	
	def __init__(self):


                # initialize controller
		self.steercontroller = PID(PID_Kp, PID_Ki, PID_Kd, -MAX_THROTTLE_GAIN, MAX_THROTTLE_GAIN)
		self.my_twist_command = None

		# initialize the camera and grab a reference to the raw camera capture
		self.camera = PiCamera()
		self.camera.resolution = (IMG_WIDTH, IMG_HEIGHT)
		self.camera.framerate = FRAMERATE
		self.max_linear_vel = MAX_THROTTLE_GAIN
		self.max_angular_vel = MAX_STEER_GAIN

		# allow the camera to warmup
		time.sleep(0.1)

		# spin up ros
		rospy.init_node('cam_control_node')
		self.pub = rospy.Publisher('driver_node/cmd_vel', Twist, queue_size=1)

		self.image_pub = rospy.Publisher('camera/image', Image)
                self.bridge = CvBridge()

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
                        self.twist_from_frame(image, dt)

			# publish drive command
	                self.pub.publish(self.my_twist_command)

        def twist_from_frame(self, image, dt):
		# prepare image
                img_warped = imagefunctions.warp(image)
                gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
                ret,img_bin = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
                
                # pick points for interpolation
                pts_x, pts_y = imagefunctions.pickpoints(img_bin)

                # fit polynomial
                if (len(pts_x)>0 and len(pts_y)>0):
                    z = np.polyfit(pts_y, pts_x, 1)
	            p = np.poly1d(z)

                    # publish robot's view
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
                    out_img = cv2.cvtColor(img_bin,cv2.COLOR_GRAY2RGB)
                    cv2.polylines(out_img,[ptsplot],False,(0,255,0))
                    cv2.line(out_img,(int(IMG_WIDTH/2),IMG_HEIGHT-1),(int(IMG_WIDTH/2),int(IMG_HEIGHT/2)),(0,0,255),1)
                    #cv2.imshow("Frame",out_img)
                    #cv2.waitKey(showtime) # required to show image

    		    # publish 
		    try:
		        self.image_pub.publish(self.bridge.cv2_to_imgmsg(out_img, "bgr8"))
		    except CvBridgeError as e:
		        print(e)

                else:
                    z = [0, 0]
	            p = np.poly1d(z)


                # cross track error
                dist_to_line = p(IMG_HEIGHT) - (IMG_WIDTH/2) # +ve: line is to the right of car
                slope = z[0] # np.arctan2
                ang_deviation = -slope # +ve: line deviates to right of car
                wt_dist = 0.005
                wt_ang = 2.0
                cte = wt_dist*dist_to_line + wt_ang*ang_deviation 

                # Controllers
                throttle = MAX_THROTTLE_GAIN
		steering = self.steercontroller.step( cte, dt )

                # Twist Command
		vel = Twist()
		vel.linear.x = self.max_linear_vel
		vel.angular.z = self.max_angular_vel*cte
                print 'dist=' + str(dist_to_line) + " ang=" + str(ang_deviation) + " => throttle=" + str(vel.linear.x) + ", steer=" + str(vel.angular.z)

		# assign Twist Command
		self.my_twist_command = vel

if __name__ == '__main__':
	try:
		CamNode()
	except rospy.ROSInterruptException:
		rospy.logerr('Could not start camcontrol node.')
