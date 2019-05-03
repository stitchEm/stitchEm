FROM stitchem/stitchem-base:latest

ENV CUDA=8.0.61-1
ENV CUDA_APT=8-0
ENV CUDA_SHORT=8.0
ENV CUDA_INSTALLER=cuda-repo-ubuntu1604_${CUDA}_amd64.deb
ENV NV_DRIVER=nvidia-418

RUN apt update && apt install -y gcc-5 g++-5
ENV CC=gcc-5
ENV CXX=g++-5

RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_INSTALLER}
RUN dpkg -i ${CUDA_INSTALLER}
RUN apt update -qq && apt install -y \
  cuda-core-${CUDA_APT} \
  cuda-cudart-dev-${CUDA_APT} \
  cuda-cufft-dev-${CUDA_APT} \
  cuda-nvml-dev-${CUDA_APT}
RUN mkdir /usr/lib/nvidia
RUN apt install -y -o Dpkg::Options::="--force-overwrite" \
  ${NV_DRIVER}


ENV CUDA_HOME=/usr/local/cuda-${CUDA_SHORT}
ENV PATH=${PATH}:${CUDA_HOME}/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
