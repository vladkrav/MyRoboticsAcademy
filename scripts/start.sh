#!/bin/bash

source /opt/ros/melodic/setup.bash
source /home/vlad/catkin_ws/devel/setup.bash
python3 /home/vlad/TFM/RoboticsAcademy/manage.py runserver 0.0.0.0:8000 &
python3.8 manager.py $1