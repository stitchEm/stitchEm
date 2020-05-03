FROM stitchem/stitchem-base:latest

ENV CUDA=10.2.89-1
ENV CUDA_APT=10-2
ENV CUDA_SHORT=10.2
ENV CUDA_INSTALLER=cuda-repo-ubuntu1804_${CUDA}_amd64.deb
ENV NV_LIB=libnvidia-compute-418

RUN wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/${CUDA_INSTALLER}
RUN dpkg -i ${CUDA_INSTALLER}
RUN apt update -qq && apt install -y \
  cuda-compiler-${CUDA_APT} \
  cuda-cudart-dev-${CUDA_APT} \
  cuda-cufft-dev-${CUDA_APT} \
  cuda-nvml-dev-${CUDA_APT} \
  ${NV_LIB}

ENV CUDA_HOME=/usr/local/cuda-${CUDA_SHORT}
ENV PATH=${PATH}:${CUDA_HOME}/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64
