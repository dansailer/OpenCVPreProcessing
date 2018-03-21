# OpenCVPreProcessing

## Install OpenCV on Mac OSX with Java bindings
Installation instruction based on ``opencv-java-tutorials <https://github.com/opencv-java/opencv-java-tutorials/blob/master/docs/source/01-installing-opencv-for-java.rst>``

First you will need to make sure that the XCode Command Line Tools are installed by executing ``xcode-select --install`` and Ant is installed as well (e.g. via homebrew) by executing ``brew install ant``.

Because the default OpenCV formulae of Homebrew does not include the JAVA support, you need to edit the formula first (and if you already have OpenCV installed, you need to uninstall it ``brew uninstall opencv``)
Now edit the formulae ``brew edit opencv`` and change ``-DBUILD_opencv_java=OFF`` into ``-DBUILD_opencv_java=ON``.
OpenCV can now be installed with ``brew install --build-from-source opencv`` and the JAVA support should be available in
```
ls -1 /usr/local/Cellar/opencv/3*/share/OpenCV/java/
libopencv_java341.dylib
opencv-341.jar
```

## Setup Eclipse external library
To register OpenCV as external library, go to ``Preferences - Java - Build Path - User Libraries`` and create new Library with name ``OpenCV`` and add the external Jar ``/usr/local/Cellar/opencv/3.4.1_2/share/OpenCV/java/opencv-341.jar`` (depending on the OpenCV version installed).
In the created opencv-341.jar entry, select native library location and set the value to ``/usr/local/Cellar/opencv/3.4.1_2/share/OpenCV/java`` the folder containing the file ``libopencv_java341.dylib``.

Now in a new project, go to ``Build Path - Add Library - User Library`` and select OpenCV.


