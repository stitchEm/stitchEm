ARG version
FROM stitchem/stitchem-base:latest

ADD . stitchEm
WORKDIR stitchEm
WORKDIR build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND_CUDA=OFF -DGPU_BACKEND_OPENCL=ON -DDISABLE_OPENCL_SPIR=ON -G Ninja ..
CMD ninja
