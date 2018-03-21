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
    && apk add --no-cache build-base cmake libjpeg libjpeg-turbo-dev libpng-dev jasper-dev tiff-dev libwebp-dev clang-dev linux-headers python2 python3 py-pip py-numpy apache-ant openblas-dev ffmpeg-dev libgomp openexr-dev gst-plugins-base-dev libgphoto2-dev v4l-utils-dev libdc1394-dev doxygen

WORKDIR /tmp/opencv-$OPENCV_VERSION/build/
ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++

# Run cmake with OpenCV build config
RUN set -x \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/tmp/opencv/usr/local -D CMAKE_INSTALL_LIBDIR=lib -D BUILD_SHARED_LIBS=OFF \
             -D CMAKE_CXX_FLAGS="$CXXFLAGS" -D CMAKE_C_FLAGS="$CFLAGS" -D ENABLE_PRECOMPILED_HEADERS=OFF -D WITH_OPENMP=ON -D WITH_OPENCL=ON \
             -D WITH_1394=OFF -D WITH_CUDA=OFF -D WITH_EIGEN=OFF -D WITH_FFMPEG=OFF -D WITH_GPHOTO2=OFF -D WITH_GSTREAMER=OFF \
             -D WITH_IPP=OFF -D WITH_JASPER=OFF -D WITH_OPENEXR=OFF -D WITH_OPENGL=OFF -D WITH_QT=OFF -D WITH_TBB=ON -D WITH_VTK=OFF \
             -D BUILD_opencv_java=ON -D BUILD_JASPER=ON -D BUILD_JPEG=ON -D BUILD_ZLIB=ON -D BUILD_PNG=ON -D BUILD_TIFF=ON -D BUILD_ILMIMF=ON \
             -D WITH_OPENCLAMDBLAS=OFF -D WITH_CUBLAS=OFF -D WITH_MATLAB=OFF -D BUILD_WEBP=ON -D BUILD_TBB=ON \
             -D BUILD_FAT_JAVA_LIB=ON ..

# Make 
RUN set -x \
    && make 

# Make install
RUN set -x \
    && make install


FROM openjdk:8-jdk-alpine
LABEL maintainer="Daniel Sailer <dsailer@gmail.com>"
RUN set -x \
    && apk add --no-cache ca-certificates libjpeg libpng libwebp tiff openblas \
    && echo "export PATH=\"${PATH}:/usr/local/share/OpenCV/java\"" >> /etc/profile
COPY --from=builder /tmp/opencv/usr/ /usr/ 

