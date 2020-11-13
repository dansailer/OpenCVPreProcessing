# OpenCVPreProcessing

## Install OpenCV on Mac OSX with Java bindings
Installation instruction based on ``opencv-java-tutorials <https://github.com/opencv-java/opencv-java-tutorials/blob/master/docs/source/01-installing-opencv-for-java.rst>``

First you will need to make sure that the XCode Command Line Tools are installed by executing ``xcode-select --install`` and Ant is installed as well (e.g. via homebrew) by executing ``brew install ant``.

Because the default OpenCV formulae of Homebrew does not include the JAVA support, you need to edit the formula first (and if you already have OpenCV installed, you need to uninstall it ``brew uninstall opencv``)
Now edit the formulae ``brew edit opencv`` and change ``-DBUILD_opencv_java=OFF`` into ``-DBUILD_opencv_java=ON``.
```
      -DBUILD_opencv_java=ON
      -DOPENCV_JAVA_TARGET_VERSION=11
      -DBUILD_JAVA=ON
      -DBUILD_opencv_java_bindings_generator=ON
      -DBUILD_FAT_JAVA_LIB=ON
      -DBUILD_SHARED_LIBS=OFF

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

## Run JAR
To run the built jar, the reference to the OpenCV libraries must be passed on as well.
```
java -Djava.library.path=/usr/local/Cellar/opencv/3.4.1_2/share/OpenCV/java -jar bin/OpenCVPreProcessing.jar resources/ocr02.jpg
```
or
```
export JAVA_LIBRARY_PATH="/usr/local/Cellar/opencv/3.4.1_2/share/OpenCV/java"; java -jar bin/OpenCVPreProcessing.jar resources/ocr02.jpg
```

## Alpine Docker Image
The Dockerfile builds OpenCV in a separate builder container with all build dependencies. The build uses static libraries instead of shared ones. All depending libraries are therefore built and do not need to be installed separately.
Building without tests (faster) ``docker build .``
Building with tests ``docker build --file Dockerfile_withTests .``
During the build the following 2 warnings are shown.
```
[ 33%] Building CXX object modules/core/CMakeFiles/opencv_core.dir/src/hal_internal.cpp.o
In file included from /tmp/opencv-3.4.1/modules/core/src/hal_internal.cpp:50:
In file included from /tmp/opencv-3.4.1/build/opencv_lapack.h:2:
In file included from /usr/include/cblas.h:5:
/usr/include/openblas_config.h:99:44: warning: '__STDC_VERSION__' is not defined, evaluates to 0 [-Wundef]
#if ((defined(__STDC_IEC_559_COMPLEX__) || __STDC_VERSION__ >= 199901L || \
                                           ^
1 warning generated.
```
```
[ 38%] Building CXX object modules/core/CMakeFiles/opencv_core.dir/src/parallel_impl.cpp.o
/tmp/opencv-3.4.1/modules/core/src/parallel_impl.cpp:42:5: warning: "Can't detect sched_yield() on the target platform. Specify CV_YIELD() definition via compiler flags." [-W#warnings]
#   warning "Can't detect sched_yield() on the target platform. Specify CV_YIELD() definition via compiler flags."
    ^
1 warning generated.
```

## Run in Alpine container

```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/share/OpenCV/java:/usr/local/lib:/usr/local/lib64"
java -jar OpenCVPreProcessing.jar ocr02.jpg
```
