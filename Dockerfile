FROM openjdk:8-jdk-alpine as builder
LABEL maintainer="Daniel Sailer <dsailer@gmail.com>"

ARG OPENCV_VERSION="3.4.1"

# Since this is just the builder container, multiple layers are used to better re-use layers at build time

# Download and unzip OpenCV source
RUN set -x \
    && apk add --no-cache unzip curl \
    && curl -sSL -o /tmp/opencv.zip https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip \
    && unzip -d /tmp/ /tmp/opencv.zip \
    && mkdir -p /tmp/opencv-$OPENCV_VERSION/build/

# Install build tools and dependencies
RUN set -x \
    && apk add --no-cache build-base cmake libjpeg libjpeg-turbo-dev libpng-dev jasper-dev tiff-dev libwebp-dev clang-dev linux-headers python2 python3 py-pip py-numpy apache-ant libgomp protobuf-dev openblas-dev

# Do not use video, digital camera and live stream devices:
# v4l-utils-dev libdc1394-dev libgphoto3-dev gst-plugins-base-dev ffmpeg-dev
# Do not use OpenExr from Industrial Light and Magic
# openexr-dev

WORKDIR /tmp/opencv-$OPENCV_VERSION/build/
ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++

# Run cmake with OpenCV build config
RUN set -x \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/tmp/opencv/usr/local -D BUILD_opencv_java=ON -D WITH_V4L=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF ..

## https://github.com/opencv/opencv/blob/master/CMakeLists.txt
#    && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/tmp/opencv/usr/local -D CMAKE_INSTALL_LIBDIR=lib -D BUILD_SHARED_LIBS=ON \
#             -D CMAKE_CXX_FLAGS="$CXXFLAGS" -D CMAKE_C_FLAGS="$CFLAGS" -D ENABLE_PRECOMPILED_HEADERS=OFF -D WITH_OPENMP=OFF -D WITH_OPENCL=OFF \
#             -D WITH_1394=OFF -D WITH_CUDA=OFF -D WITH_EIGEN=OFF -D WITH_FFMPEG=OFF -D WITH_GPHOTO2=OFF -D WITH_GSTREAMER=OFF \
#             -D WITH_IPP=OFF -D WITH_OPENEXR=OFF -D WITH_OPENGL=OFF -D WITH_QT=OFF -D WITH_TBB=OFF -D WITH_VTK=OFF -D WITH_MATLAB=OFF \
#             -D WITH_OPENCLAMDBLAS=OFF -D WITH_CUBLAS=OFF \
#             -D WITH_JASPER=ON -D WITH_JPEG=ON -D WITH_PNG=ON -D WITH_WEBP=ON -D WITH_TIFF=ON -D WITH_LAPACK=ON \
#             -D BUILD_opencv_java=ON \
#             -D BUILD_JASPER=OFF -D BUILD_JPEG=OFF -D BUILD_ZLIB=OFF -D BUILD_PNG=OFF -D BUILD_TIFF=OFF -D BUILD_ILMIMF=OFF \
#             -D BUILD_WEBP=OFF -D BUILD_TBB=OFF -D BUILD_PROTOBUF=OFF ..

# Make 
RUN set -x \
    && make -j8

# Make install
RUN set -x \
    && make install


FROM openjdk:8-jdk-alpine
LABEL maintainer="Daniel Sailer <dsailer@gmail.com>"
RUN set -x \
    && apk add --no-cache ca-certificates libjpeg libpng libwebp tiff openblas jasper protobuf \
    && echo "export PATH=\"\${PATH}/:/usr/local/share/OpenCV/java\"" >> /etc/profile \
    && echo "export LD_LIBRARY_PATH=\"\${LD_LIBRARY_PATH}/:/usr/local/share/OpenCV/java:/usr/local/lib:/usr/local/lib64\"" >> /etc/profile
COPY --from=builder /tmp/opencv/usr/ /usr/ 
COPY bin/OpenCVPreProcessing.jar /
COPY resources/* /

