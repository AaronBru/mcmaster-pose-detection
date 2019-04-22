# mcmaster-pose-detection

A library for determing rotation and translation from marked objects, utilizing the D435 from Intel Realsense, Aruco markers,
and Kabsch's algorithm.  For use with Jetson's TX2.

## Installation
1. Flash the TX2 using L4T 28.2.1 [here](https://developer.nvidia.com/embedded/linux-tegra-r2821). This is not the most recent version.  This is to ensure a stable installation of librealsense.
2. Clone (https://github.com/jetsonhacks/buildLibrealsense2TX) and follow the instructions to install librealsense and patch the kernel.
3. Run the following commands in this root directory once cloned:
```bash
git submodule init
git submodule update
make
```
4. The demo application should now be built.  Run markerDetection to view the application.

## Documentation

The demo application in markerDetection.cpp provides a thorough example of how to interface with this library.

Use initRsCam and rsConfig structure to configure the camera and setup the hardware. Populate the structure with the configuration required.
Print aruco markers (this generator [here](http://chev.me/arucogen/) creates them) and put them on an object.  Carefully measure the position of the
corners of all the markers on an object with respect to its local coordinate system.

The main use of the library is in the detectPoseRealsense class.  Create an instance with a vector of all Aruco marker ids
that should be detected for use with the constructor.

When retrieving pose, create a map object with the Aruco ids as keys and an array of arrays.  The inner arrays have the x, y and z
components of a corner.  The outer arrays hold the positions of the 4 corners of the associated marker moving clockwise from the top left corner.
Use getPose with the proper array of arrays to get the translation and rotation of an object with respect to the camera.
