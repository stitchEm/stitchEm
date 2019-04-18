// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "parallax/mergerPair.hpp"

#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "cuda/error.hpp"
#include "cuda/util.hpp"
#include "core/rect.hpp"
#include "parallax/flowConstant.hpp"

#include <string.h>

namespace VideoStitch {
namespace Core {

#define TILE_X 16
#define TILE_Y 16

__global__ void pairMappingMaskKernel(const int wrapWidth, const int input0OffsetX, const int input0OffsetY,
                                      const int input0Width, const int input0Height, const float2* input0CoordBuffer,
                                      const int input1OffsetX, const int input1OffsetY, const int input1Width,
                                      const int input1Height, const float2* input1CoordBuffer, const int outputOffsetX,
                                      const int outputOffsetY, const int outputWidth, const int outputHeight,
                                      uint32_t* outputMask) {
  // calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < outputWidth && y < outputHeight) {
    outputMask[y * outputWidth + x] = 0;

    // Put the first mask in
    const int input0X = (x + outputOffsetX - input0OffsetX + wrapWidth) % wrapWidth;
    const int input0Y = (y + outputOffsetY - input0OffsetY);
    if (input0X >= 0 && input0X < input0Width && input0Y >= 0 && input0Y < input0Height) {
      float2 coord0 = input0CoordBuffer[input0Y * input0Width + input0X];
      if (coord0.x != INVALID_FLOW_VALUE && coord0.y != INVALID_FLOW_VALUE) {
        outputMask[y * outputWidth + x] += 1 << 1;
      }
    }

    // Put the second mask in
    const int input1X = (x + outputOffsetX - input1OffsetX + wrapWidth) % wrapWidth;
    const int input1Y = (y + outputOffsetY - input1OffsetY);
    if (input1X >= 0 && input1X < input1Width && input1Y >= 0 && input1Y < input1Height) {
      float2 coord1 = input1CoordBuffer[input1Y * input1Width + input1X];
      if (coord1.x != INVALID_FLOW_VALUE && coord1.y != INVALID_FLOW_VALUE) {
        outputMask[y * outputWidth + x] += 1 << 2;
      }
    }
  }
}

Status MergerPair::setupPairMappingMask(GPU::Buffer<uint32_t> devMask, GPU::Stream gpuStream) const {
  const Rect rect0 = getBoundingPanoRect(0);
  const Rect rect1 = getBoundingPanoRect(1);
  Rect iRect = getBoundingPanosIRect();
  dim3 dimBlock(TILE_X, TILE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(iRect.getWidth(), dimBlock.x),
               (unsigned)Cuda::ceilDiv(iRect.getHeight(), dimBlock.y), 1);

  cudaStream_t stream = gpuStream.get();

  pairMappingMaskKernel<<<dimGrid, dimBlock, 0, stream>>>(
      (int)wrapWidth, (int)rect0.left(), (int)rect0.top(), (int)rect0.getWidth(), (int)rect0.getHeight(),
      panoToInputSpaceCoordMapping0.borrow_const().get(), (int)rect1.left(), (int)rect1.top(), (int)rect1.getWidth(),
      (int)rect1.getHeight(), panoToInputSpaceCoordMapping1.borrow_const().get(), (int)iRect.left(), (int)iRect.top(),
      (int)iRect.getWidth(), (int)iRect.getHeight(), devMask.get());

  return CUDA_STATUS;
}

}  // namespace Core
}  // namespace VideoStitch
