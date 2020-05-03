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

ENV CUDA=10.2.89-1
ENV CUDA_APT=10-2
ENV CUDA_SHORT=10.2
ENV CUDA_INSTALLER=cuda-repo-ubuntu1804_${CUDA}_amd64.deb
RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/${CUDA_INSTALLER}
RUN apt install ./${CUDA_INSTALLER}
RUN apt update -qq && apt install -y \
  cuda-compiler-${CUDA_APT} \
  cuda-cudart-dev-${CUDA_APT} \
  cuda-cufft-dev-${CUDA_APT} \
  cuda-nvml-dev-${CUDA_APT} \
  ${NV_LIB}

ENV CUDA_HOME=/usr/local/cuda-${CUDA_SHORT}
ENV PATH=${PATH}:${CUDA_HOME}/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
