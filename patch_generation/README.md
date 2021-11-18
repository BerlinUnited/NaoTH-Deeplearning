# Patch Generation and Evaluation

This sub project contains code for getting patches from images. This is done by utilizing the current naoth lib written in C++. This is the same code that is running on the robot. To be able to use it a few requirements are necessary.

- you need to compile the naoth lib for your system
- the images given to patch generation or evaluation scripts must contain the camera matrix and other options. This is done automatically for images that were extracted from logfiles recorded on the nao robot. The reason for that is that the patch generation algorithm uses the position of the camera to determine the interesting regions in the image.

## Running Patch Generation
TODO