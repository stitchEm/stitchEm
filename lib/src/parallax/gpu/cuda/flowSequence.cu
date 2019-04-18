// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "parallax/flowSequence.hpp"

#include "./kernels/patchDifferenceFunction.cu"

#include "backend/common/vectorOps.hpp"
#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "cuda/error.hpp"
#include "cuda/util.hpp"
#include "gpu/image/sampling.hpp"
#include "gpu/image/imageOps.hpp"
#include "gpu/image/blur.hpp"
#include "gpu/stream.hpp"

#include <string.h>

namespace VideoStitch {
namespace Core {
#define WARPER_BLOCK_SIZE_X 16
#define WARPER_BLOCK_SIZE_Y 16
#define WARPER_BLOCK_SIZE_Z 16

__global__ void weightedAvgFlowWarpKernel(const int2 size, const int frameId, const int frameCount,
                                          const float sigmaTime, const float* frames, const float2* inputFlows,
                                          float2* outputFlow) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < size.x && y < size.y) {
    const int index = y * size.x + x;
    float2 avgFlow = make_float2(0, 0);
    float totalWeight = 0.0;
    for (int t = 0; t < frameCount; t++)
      if (frames[t] >= 0) {
        // Now calculate the distance of time
        float distTime = float(frames[t] - frames[frameId]) / frameCount;
        float weightTime = exp(-abs(distTime * distTime * sigmaTime));
        const float2 inputFlow = inputFlows[t * size.x * size.y + index];
        if (inputFlow.x != INVALID_FLOW_VALUE) {
          avgFlow += inputFlow * weightTime;
          totalWeight += weightTime;
        }
      }
    if (totalWeight > 0) {
      outputFlow[index] = avgFlow / totalWeight;
    } else {
      outputFlow[index] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
    }
  }
}

Status FlowSequence::regularizeFlowTemporally(const std::string& name, const frameid_t frame, const int2 size,
                                              const int2 offset, GPU::Buffer<float2> flow, GPU::Stream gpuStream) {
  // Cache the input flow
  FAIL_RETURN(cacheBuffer<float2>(frame, name, size, offset, flow, gpuStream));

  TypedCached<float2>* cache = dynamic_cast<TypedCached<float2>*>(getFlowCachedBuffer(name).get());
  if (!cache) {
    return {Origin::ImageFlow, ErrType::InvalidConfiguration, "FlowSequence::cache is not valid"};
  }
  const int frameIndex = getFrameIndex(frame);
  if (frameIndex < 0) {
    return {Origin::ImageFlow, ErrType::InvalidConfiguration, "FlowSequence::frameindex < 0"};
  }

  // Now compute the weighted average flow
  // Now make the flow as stable as possible from previous computation
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(WARPER_BLOCK_SIZE_X, WARPER_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, dimBlock.x), (unsigned)Cuda::ceilDiv(size.x, dimBlock.y), 1);

  weightedAvgFlowWarpKernel<<<dimGrid, dimBlock, 0, stream>>>(size, frameIndex, (int)getFrames().numElements(), 5,
                                                              getFrames().get(), cache->getBuffer().get(), flow.get());
  return CUDA_STATUS;
}
}  // namespace Core
}  // namespace VideoStitch
