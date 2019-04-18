// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "gpu/exampleKernel.hpp"

#include "deviceBuffer.hpp"
#include "deviceStream.hpp"

#include "cuda/error.hpp"
#include "cuda/util.hpp"

namespace {

__global__ void vecAddDummy(float* output, const float* input, unsigned int nbElem, float mult) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int gridSize = gridDim.x * blockDim.x;
  for (int i = tid; i < nbElem; i += gridSize) {
    output[i] = mult * input[i];
  }
}
}  // namespace

namespace VideoStitch {
namespace Core {

Status callDummyKernel(GPU::Buffer<float> outputBuff, const GPU::Buffer<const float>& inputBuff,
                       unsigned int nbElements, float mult, GPU::Stream stream) {
  dim3 dimBlock(16, 1, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(nbElements, dimBlock.x), 1, 1);
  inputBuff.get();
  vecAddDummy<<<dimGrid, dimBlock, 0, stream.get()>>>(outputBuff.get().raw(), inputBuff.get().raw(), nbElements, mult);
  return CUDA_STATUS;
}

}  // namespace Core
}  // namespace VideoStitch
