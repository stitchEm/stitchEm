// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "parallax/linearFlowWarper.hpp"

#include "./kernels/patchDifferenceFunction.cu"

#include "backend/common/vectorOps.hpp"

#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "backend/cuda/core1/kernels/samplingKernel.cu"
#include "gpu/image/sampling.hpp"
#include "gpu/image/imageOps.hpp"
#include "gpu/image/blur.hpp"
#include "gpu/stream.hpp"
#include "cuda/error.hpp"
#include "cuda/util.hpp"

#include <string.h>

namespace VideoStitch {
namespace Core {

#define WARPER_BLOCK_SIZE_X 16
#define WARPER_BLOCK_SIZE_Y 16

struct BilinearLookupFlow {
  typedef float2 Type;
  static inline __device__ Type outOfRangeValue() { return make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE); }

  static inline __device__ Type interpolate(const float2 uv, const Type topLeft, const Type topRight,
                                            const Type bottomRight, const Type bottomLeft) {
    Type total = make_float2(0, 0);
    float weight = 0;
    const int uTopLeft = floorf(uv.x);
    const int vTopLeft = floorf(uv.y);
    const float du = (uv.x - uTopLeft);
    const float dv = (uv.y - vTopLeft);

    if (topLeft.x != INVALID_FLOW_VALUE) {
      total += topLeft * (1.0f - du) * (1.0f - dv);
      weight += (1.0f - du) * (1.0f - dv);
    } else {
      return outOfRangeValue();
    }

    if (topRight.x != INVALID_FLOW_VALUE) {
      total += topRight * du * (1.0f - dv);
      weight += du * (1.0f - dv);
    } else {
      return outOfRangeValue();
    }

    if (bottomRight.x != INVALID_FLOW_VALUE) {
      total += bottomLeft * (1.0f - du) * dv;
      weight += (1.0f - du) * dv;
    } else {
      return outOfRangeValue();
    }

    if (bottomLeft.x != INVALID_FLOW_VALUE) {
      total += bottomRight * du * dv;
      weight += du * dv;
    } else {
      return outOfRangeValue();
    }

    if (weight) {
      return total / weight;
    } else {
      return outOfRangeValue();
    }
  }
};

// Warp an image from pano space to input space
// Use both the reference mapping and the
__global__ void linearFlowWarpKernel(
    const int warpOutputPanoWidth, const int warpedOffsetX, const int warpedOffsetY, const int warpedWidth,
    const int warpedHeight, uint32_t* warpedBuffer, const int inputWidth, const int inputHeight,
    const uint32_t* inputBuffer, const float2* panoToInputBuffer, const float2* panoToInterBuffer,
    const int interOffsetX, const int interOffsetY, const int interWidth, const int interHeight,
    const float2* interToInputBuffer, const int flowOffsetX, const int flowOffsetY, const int flowWidth,
    const int flowHeight, const float2* interToInterFlowBuffer, const int lookupOffsetX, const int lookupOffsetY,
    const int weightOffsetX, const int weightOffsetY, const int weightWidth, const int weightHeight,
    const unsigned char* interToInterWeightBuffer, float4* debug, uint32_t* flowWarpedBuffer) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < warpedWidth && y < warpedHeight) {
    const int index = y * warpedWidth + x;
    const float2 panoToInput = panoToInputBuffer[index];
    float2 lookupInput = panoToInput;
    const float2 panoToInter = panoToInterBuffer[index];

    // Weight map is in the pano coordinate
    // Get the correct weight
    const int2 weightLookupCoord =
        make_int2(x, y) + make_int2(warpedOffsetX, warpedOffsetY) - make_int2(weightOffsetX, weightOffsetY);
    float weight = 255;
    if (weightLookupCoord.x >= 0 && weightLookupCoord.x < weightWidth && weightLookupCoord.y >= 0 &&
        weightLookupCoord.y < weightHeight) {
      weight = interToInterWeightBuffer[weightLookupCoord.y * weightWidth + weightLookupCoord.x];
    }
    weight /= 255;
    //// Find the flow here
    float2 interFlowInput = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
    const float2 interFlowLookupcoord = panoToInter - make_float2(flowOffsetX, flowOffsetY);
    if (interFlowLookupcoord.x >= 0 && interFlowLookupcoord.x < flowWidth && interFlowLookupcoord.y >= 0 &&
        interFlowLookupcoord.y < flowHeight) {
      // This is the flow from 0 to 1
      float2 interToInterFlow = Image::bilinearLookup<BilinearLookupFlow>(
          interFlowLookupcoord, make_int2(flowWidth, flowHeight), interToInterFlowBuffer);
      // Proceed with valid flow only
      if (interToInterFlow.x != INVALID_FLOW_VALUE && interToInterFlow.y != INVALID_FLOW_VALUE) {
        // Convert from optical-flow based coordinate to intermediate coordinate
        interToInterFlow =
            interToInterFlow - make_float2(interOffsetX, interOffsetY) + make_float2(lookupOffsetX, lookupOffsetY);
        if (interToInterFlow.x >= 0 && interToInterFlow.y >= 0 && interToInterFlow.x < interWidth &&
            interToInterFlow.y < interHeight) {
          interFlowInput = Image::bilinearLookup<BilinearLookupFlow>(
              interToInterFlow, make_int2(interWidth, interHeight), interToInputBuffer);
          if (interFlowInput.x != INVALID_FLOW_VALUE && interFlowInput.y != INVALID_FLOW_VALUE) {
            lookupInput = panoToInput * weight + interFlowInput * (1 - weight);
          } else {
            lookupInput = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
          }
        }
      }
    }

    debug[index] = make_float4(weight, weight, weight, weight);
    warpedBuffer[index] = Image::bilinearLookup<Image::BilinearLookupRGBAtoRGBA>(
        lookupInput, make_int2(inputWidth, inputHeight), inputBuffer);
    if (flowWarpedBuffer) {
      flowWarpedBuffer[index] = Image::bilinearLookup<Image::BilinearLookupRGBAtoRGBA>(
          interFlowInput, make_int2(inputWidth, inputHeight), inputBuffer);
    }
  }
}

Status LinearFlowWarper::warp(GPU::Buffer<uint32_t> warpedBuffer, const GPU::Buffer<const uint32_t> inputBuffer,
                              const Rect& flowRect, const GPU::Buffer<const float2> flow, const int lookupOffsetX,
                              const int lookupOffsetY, GPU::Buffer<float4> debug,
                              GPU::Buffer<uint32_t> flowWarpedBuffer, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  // Flow is in the intermediate space, flow from image 1 to image 0 based on template from image 0
  // Input buffer - the original input images
  // Warped buffer - final image in the pano space
  // Weight buffer - in the pano space
  // Need to blend flow in the input space
  Rect panoRect1 = mergerPair->getBoundingPanoRect(1);
  Rect iRect = mergerPair->getBoundingPanosIRect();
  dim3 dimBlock(WARPER_BLOCK_SIZE_X, WARPER_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(panoRect1.getWidth(), dimBlock.x),
               (unsigned)Cuda::ceilDiv(panoRect1.getHeight(), dimBlock.y), 1);
  // First, lookup the flow in pano space from intermediate space
  Rect interRect1 = mergerPair->getBoundingInterRect(1, 0);

  const int2 inputSize1 = mergerPair->getInput1Size();

  linearFlowWarpKernel<<<dimGrid, dimBlock, 0, stream>>>(
      mergerPair->getWrapWidth(), (int)panoRect1.left(), (int)panoRect1.top(), (int)panoRect1.getWidth(),
      (int)panoRect1.getHeight(), warpedBuffer.get(), inputSize1.x, inputSize1.y, inputBuffer.get(),
      mergerPair->getPanoToInputSpaceCoordMapping(1).get(), mergerPair->getPanoToInterSpaceCoordMapping(1).get(),
      (int)interRect1.left(), (int)interRect1.top(), (int)interRect1.getWidth(), (int)interRect1.getHeight(),
      mergerPair->getInterToLookupSpaceCoordMappingBufferLevel(1, 0).get(), (int)flowRect.left(), (int)flowRect.top(),
      (int)flowRect.getWidth(), (int)flowRect.getHeight(), flow.get(), lookupOffsetX, lookupOffsetY, (int)iRect.left(),
      (int)iRect.top(), (int)iRect.getWidth(), (int)iRect.getHeight(), linearMaskWeight.borrow_const().get(),
      debug.get(), flowWarpedBuffer.get());

  return CUDA_STATUS;
}

}  // namespace Core
}  // namespace VideoStitch
