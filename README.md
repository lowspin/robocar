# robocar
DIY Robocar with RaspberryPi and Pi Camera

## Hardware
* Raspberry Pi 3 Model B/B+
* Adafruit DC & Stepper Motor HAT for Raspberry Pi

## Software
* [Raspbian Stretch](https://www.raspberrypi.org/downloads/raspbian/)
* [ROS Kinetic](http://wiki.ros.org/kinetic)
* [Adafruit drivers](https://github.com/adafruit/Adafruit-Motor-HAT-Python-Library.git)

## Function: Teleops
Using an XBox game controller to control robocar. XBox Controller configuration:
* Right trigger (axes 5): forward
* Left trigger (axes 2): reverse/brake
* Left joy-stick (axes 0): steer
```
catkin_make
source devel/setup.bash
roslaunch launch/dbw.launch
```

## Function: Preset Drive Commands (Blind)
Using pre-saved drive command (throttle,steer,duration) to drive robocar.
```
catkin_make
source devel/setup.bash
roslaunch launch/blindmove.launch
```
Note: Edit `src/blindmove/launch/drivecommands.csv` with one drive command per line in the following format:
* `throttle`(-1.0:1.0), `steer`(-1.0:1.0), `duration`(sec)

## Function: Camera Control Driving
Using camera input to calculate drive commands.
```
catkin_make
source devel/setup.bash
roslaunch launch/cam.launch
```
Note: Work-in-progress (5/10/2018) 
