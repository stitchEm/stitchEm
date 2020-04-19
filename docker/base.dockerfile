FROM ubuntu:bionic

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
  bison \
  ccache \
  clang \
  doxygen \
  flex \
  git \
  libceres-dev \
  libeigen3-dev \
  libglm-dev \
  libfaac-dev \
  libfaad-dev \
  libglew-dev \
  libglfw3-dev \
  libmp3lame-dev \
  libopencv-dev \
  libopenexr-dev \
  libportaudio-ocaml-dev \
  librtmp-dev \
  libx264-dev \
  ninja-build \
  ocl-icd-opencl-dev \
  opencl-headers \
  portaudio19-dev \
  python-pip \
  qt5-default \
  qtmultimedia5-dev \
  qttools5-dev \
  swig \
  wget \
  xxd
RUN pip install cmake
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-key add 7fa2af80.pub
RUN apt clean
