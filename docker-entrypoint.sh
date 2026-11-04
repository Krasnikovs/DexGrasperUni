#!/bin/bash
# Source the ROS 2 environment
source /opt/ros/humble/setup.bash

# Execute the command passed into the container
exec "$@"
