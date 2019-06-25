Extraction of human body and computation of human moving speed
==============================================================

##  Implemented by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) associated with [ZED camera](https://www.stereolabs.com/).

This project is to aquire video stream based on ZED camera processed by OpenPose framework.

1. op_zed.cpp file only processes 2D images. 
_When executing the project, the first argument is to control the opening camera, i.e., left or right._

2. sync_zed.cpp file is under developing. It processes 3D images and expected to implement 3D reconstruction.
