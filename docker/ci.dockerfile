ARG version
FROM stitchem/stitchem-base-cuda${version}:latest

ADD . stitchEm
WORKDIR stitchEm
RUN echo ${CUDA_SHORT} > cuda.version
WORKDIR build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DLINUX_PKG=ON -DRTMP_NVENC=OFF -DCUDA_TARGET_ARCH="50" -G Ninja ..
CMD ninja
