# robocar
DIY Robocar with RaspberryPi and Pi Camera

## Hardware
* Raspberry Pi 3 Model B/B+
* Adafruit DC & Stepper Motor HAT for Raspberry Pi

## Software
* [Raspbian Stretch](https://www.raspberrypi.org/downloads/raspbian/)
* [ROS Kinetic](http://wiki.ros.org/kinetic)
* [Adafruit drivers](https://github.com/adafruit/Adafruit-Motor-HAT-Python-Library.git)

## Teleops
XBOX Controller configuration:
* Right trigger (axes 5): forward
* Left trigger (axes 2): reverse/brake
* Left joy-stick (axes 0): steer
```
catkin_make
source devel/setup.bash
roslaunch launch/dbw.launch
```
