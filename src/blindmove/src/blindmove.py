#!/usr/bin/env python

import time
import rospy
from geometry_msgs.msg import Twist

class BlindDrive(object):
    def __init__(self):
        rospy.init_node('robocar_blindmove')
        self.pub = rospy.Publisher('driver_node/cmd_vel',Twist,queue_size=1)

        self.throttle = 0.0
        self.steer = 0.0
        self.twist = Twist()

        self.max_linear_vel = rospy.get_param('~max_linear_vel',0.2)
        self.max_angular_vel = rospy.get_param('~max_angular_vel',1.5707)

        #print 'max linear vel = ' + str(self.max_linear_vel) + ', max angular vel = ' + str(self.max_angular_vel)
        self.driveroutine()

    def driveroutine(self):
        self.drive(1.0, 0.0, 2.0)
        self.drive(-1.0, 1.0, 3.8)
        self.drive(-1.0, 0.5, 0.7) 

    def drive(self, cmdthrottle, cmdsteer, cmdduration):
        
        print 'throttle ' + str(cmdthrottle) + ', steer ' + str(cmdsteer) + ', duration ' + str(cmdduration)
        self.twist.linear.x = self.max_linear_vel * cmdthrottle
        self.twist.angular.z = self.max_angular_vel * cmdsteer
        print 'twist.linear=' + str(self.twist.linear.x) + ', twist.angular=' + str(self.twist.angular.z)

        dt = 0.02
        rate = rospy.Rate(1/dt)
        numsteps = int(cmdduration / dt);
        for tt in range(numsteps):
            self.pub.publish(self.twist)
            #time.sleep(dt)
            rate.sleep()

        #self.pub.publish(self.twist)
        #time.sleep(cmdduration)

        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.pub.publish(self.twist)


if __name__ == '__main__':
    BlindDrive()

