ARG version
FROM stitchem/stitchem-base-cuda${version}:latest

ADD . stitchEm
WORKDIR stitchEm
RUN echo ${CUDA_SHORT} > cuda.version
WORKDIR build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DRTMP_NVENC=OFF -DCUDA_TARGET_ARCH="50" -DGPU_BACKEND_CUDA=ON -DGPU_BACKEND_OPENCL=OFF -DDISABLE_OPENCL_SPIR=ON -G Ninja ..
CMD ninja
