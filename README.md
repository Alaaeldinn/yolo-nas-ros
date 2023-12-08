# YOLO-NAS-ROS 

## Overview

Welcome to the YOLO-NAS ROS package! This package integrates YOLO (You Only Look Once) object detection with neural architecture search (NAS) capabilities into the Robot Operating System (ROS). YOLO-NAS is designed to provide real-time object detection with the ability to dynamically optimize the underlying neural network architecture for efficiency and performance.

## Prerequisites

Before using this ROS package, ensure that you have the following prerequisites installed:

- ROS Kinetic, Melodic, or Noetic
- OpenCV
- CUDA and cuDNN (for GPU acceleration, optional but recommended)
- YOLO-NAS dependencies (refer to the YOLO-NAS documentation)

## Installation

1. Clone this repository into your ROS workspace:

    ```bash
    cd ~/catkin_ws/src
    git clone https://github.com/Alaaeldinn/yolo-nas-ros.git
    ```

2. Build your ROS workspace:

    ```bash
    cd ~/catkin_ws
    catkin_make
    source devel/setup.bash
    ```

## Usage

Launch the YOLO-NAS node:

    roslaunch yolo_nas_ros yolo_nas.launch

