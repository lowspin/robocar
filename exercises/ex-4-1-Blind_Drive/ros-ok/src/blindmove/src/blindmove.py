#!/usr/bin/env python

import os
import csv
import time

import rospy
from geometry_msgs.msg import Twist

CSV_HEADER = ['throttle', 'steer', 'duration']

class DriveCmd:
    def __init__(self, throttle, steer, duration):
        self.throttle = throttle
        self.steer = steer
        self.duration = duration

class BlindDrive(object):
    def __init__(self):
        rospy.init_node('robocar_blindmove')
        self.pub = rospy.Publisher('driver_node/cmd_vel',Twist,queue_size=1)

        self.throttle = 0.0
        self.steer = 0.0
        self.twist = Twist()

        self.max_linear_vel = rospy.get_param('~max_linear_vel',0.2)
        self.max_angular_vel = rospy.get_param('~max_angular_vel',1.5707)

        self.driveroutine()

    def driveroutine(self):
        drivecommands = self.new_command_loader(rospy.get_param('~divecmdfile'))
        for cmd in drivecommands:
            self.drive(cmd.throttle, cmd.steer, cmd.duration)
        print 'All Commands Executed. Ctrl-c to end'

    def new_command_loader(self, path):
        if os.path.isfile(path):
            drivecommands = self.load_commands(path)
            rospy.loginfo('Drive Commands Loaded')
            return drivecommands
        else:
            rospy.logerr('%s is not a file', path)

    def load_commands(self, fname):
        commands = []
        with open(fname) as wfile:
            reader = csv.DictReader(wfile, CSV_HEADER)
            for cmd in reader:
                throttle = float(cmd['throttle'])
                steer = float(cmd['steer'])
                duration = float(cmd['duration'])
                print 'Load cmd = ' + str(throttle) + ', ' + str(steer) + ', ' + str(duration)
                c = DriveCmd(throttle,steer,duration)
                commands.append(c)
        return commands

    def drive(self, cmdthrottle, cmdsteer, cmdduration):        
        print 'Execute: throttle ' + str(cmdthrottle) + ', steer ' + str(cmdsteer) + ', duration ' + str(cmdduration)
        self.twist.linear.x = self.max_linear_vel * cmdthrottle
        self.twist.angular.z = self.max_angular_vel * cmdsteer
        #print 'twist.linear=' + str(self.twist.linear.x) + ', twist.angular=' + str(self.twist.angular.z)

        dt = 0.02
        rate = rospy.Rate(1/dt)
        numsteps = int(cmdduration / dt);
        for tt in range(numsteps):
            self.pub.publish(self.twist)
            rate.sleep()

        #self.pub.publish(self.twist)
        #time.sleep(cmdduration)

        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.pub.publish(self.twist)


if __name__ == '__main__':
    BlindDrive()

