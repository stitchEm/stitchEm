// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "parallax/simpleFlow.hpp"

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
#include "util/imageProcessingGPUUtils.hpp"

namespace VideoStitch {
namespace Core {

#define TILE_WIDTH 16
#define CUDABLOCKSIZE 512
#define SIMPLEFLOW_KERNEL_BLOCK_SIZE_X 16
#define SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y 8

__global__ void forwardFlowKernel(const int flowSize, const int windowSize, const float flowMagnitudeWeight,
                                  const float gradientWeight, const int2 size0, const int2 offset0,
                                  const uint32_t* input0, const float* gradient0, const int2 size1, const int2 offset1,
                                  const uint32_t* input1, const float* gradient1, const float2* inputFlowOffset,
                                  float2* flow, float* confidence) {
  // Check whether we need to calculate the flow
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size0.x || y >= size0.y) return;

  uint32_t v0 = input0[y * size0.x + x];
  if (Image::RGBA::a(v0) == 0) {  // If current alpha is 0, do nothing
    if (inputFlowOffset) {
      flow[y * size0.x + x] = inputFlowOffset[y * size0.x + x];
      if (confidence) {
        confidence[y * size0.x + x] = 1;
      }
      return;
    }
    flow[y * size0.x + x] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
    if (confidence) {
      confidence[y * size0.x + x] = 0;
    }
    return;
  }
  int2 coord1 = make_int2(x + offset0.x - offset1.x, y + offset0.y - offset1.y);
  if (!inRange(coord1, size1)) {
    flow[y * size0.x + x] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
    if (confidence) {
      confidence[y * size0.x + x] = 0;
    }
    return;
  }
  uint32_t v1 = input1[coord1.y * size1.x + coord1.x];
  if (Image::RGBA::a(v1) == 0) {
    flow[y * size0.x + x] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
    if (confidence) {
      confidence[y * size0.x + x] = 0;
    }
    return;
  }

  float2 flowOffset = make_float2(0, 0);
  if (inputFlowOffset) {
    flowOffset = inputFlowOffset[y * size0.x + x];
  }
  // Try to find the best forward flow here
  int2 sampleCoord = make_int2(x, y);
  float bestCost = MAX_INVALID_COST;
  float2 minFlow = make_float2(flowOffset.x, flowOffset.y);
  float totalCost = 0;
  float totalCount = 0;
  for (int i = -flowSize; i <= flowSize; i++)
    for (int j = -flowSize; j <= flowSize; j++) {
      int2 mapCoord =
          make_int2(flowOffset.x + x + i + offset0.x - offset1.x, flowOffset.y + y + j + offset0.y - offset1.y);
      if (inRange(mapCoord, size1)) {
        float cost = getCost(windowSize, gradientWeight, size0, input0, gradient0, sampleCoord, size1, input1,
                             gradient1, mapCoord) +
                     flowMagnitudeWeight * sqrtf(i * i + j * j) / sqrtf(2 * flowSize * flowSize);
        if (cost < MAX_INVALID_COST) {
          totalCost += cost;
          totalCount++;
        }
        if (cost < bestCost) {
          bestCost = cost;
          minFlow = make_float2(flowOffset.x + i, flowOffset.y + j);
        }
      }
    }

  flow[y * size0.x + x] = minFlow;

  if (confidence) {
    if (bestCost != MAX_INVALID_COST) {
      int2 mapCoord = make_int2(sampleCoord.x + minFlow.x + offset0.x - offset1.x,
                                sampleCoord.y + minFlow.y + offset0.y - offset1.y);
      confidence[y * size0.x + x] = (getCUR(windowSize, gradientWeight, size0, input0, gradient0, sampleCoord, size1,
                                            input1, gradient1, mapCoord));
      // confidence[y * size0.x + x] = totalCost / totalCount - bestCost;
    } else {
      confidence[y * size0.x + x] = 0;
    }
  }
}

__global__ void flowAgreementConfidenceKernel(const int flowSize, const int2 size0, const int2 offset0,
                                              const float2* flow0, const float* confidence0, const int2 size1,
                                              const int2 offset1, const float2* flow1, const float* confidence1,
                                              float* flowAgreementConfidence0) {
  // Check whether we need to calculate the agreement confidence
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size0.x || y >= size0.y) return;
  const float2 forwardFlow = flow0[y * size0.x + x];
  const float forwardConfidence = confidence0[y * size0.x + x];
  const int2 lookupCoord =
      make_int2(x + forwardFlow.x + offset0.x - offset1.x, y + forwardFlow.y + offset0.y - offset1.y);
  if (!inRange(lookupCoord, size1)) {
    flowAgreementConfidence0[y * size0.x + x] = 0;
    return;
  }
  const float2 backwardFlow = flow1[lookupCoord.y * size1.x + lookupCoord.x];
  const float backwardConfidence = confidence1[lookupCoord.y * size1.x + lookupCoord.x];

  // Check if forward and backward flow agree
  float normalizedAgreementLength =
      length(forwardFlow + backwardFlow) / (length(make_float2(2 * flowSize + 1, 2 * flowSize + 1)));
  flowAgreementConfidence0[y * size0.x + x] =
      powf(fmaxf(1 - normalizedAgreementLength, 0.0), 3)  // * forwardConfidence;
      * sqrtf(backwardConfidence * forwardConfidence);
}

__global__ void confidenceTransformKernel(const int width, const int height, const float threshold, const float gamma,
                                          const float clampedValue, const float* inputConfidence,
                                          float* outputConfidence) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  float inputValue = inputConfidence[y * width + x];
  if (inputValue < threshold) {
    outputConfidence[y * width + x] = 0;
  } else {
    outputConfidence[y * width + x] = powf(inputValue, gamma);
  }
}

__device__ float getSpacialWeight(const float sigmaSpace, const float x) { return exp(-abs(sigmaSpace) * x * x); }

__global__ void confidenceAwareFlowBlurKernel(const bool extrapolation, const int2 size, const int kernelSize,
                                              const float sigmaSpace, const float sigmaImage,
                                              const float sigmaConfidence, const uint32_t* const inputImage,
                                              const float2* const inputFlow, const float* const inputConfidence,
                                              float2* const outputFlow) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size.x || y >= size.y) return;
  if (extrapolation) {
    float2 inFlow = inputFlow[y * size.x + x];
    if (inFlow.x != INVALID_FLOW_VALUE) {
      outputFlow[y * size.x + x] = inFlow;
      return;
    }
  }
  // check if the current flow is not valid, then just do nothing
  float maxDist = kernelSize * 1.4142;
  uint32_t imageColor;
  if (sigmaImage) {
    imageColor = inputImage[y * size.x + x];
  }
  if (!extrapolation) {
    if (sigmaImage > 0) {
      if (!Image::RGBA::a(imageColor)) {
        outputFlow[y * size.x + x] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
        return;
      }
    }
  }
  float sumWeight = 0;
  float2 sumContribution = make_float2(0, 0);

  if (maxDist == 0) maxDist = 1;
  for (int dx = -kernelSize; dx <= kernelSize; dx++) {
    for (int dy = -kernelSize; dy <= kernelSize; dy++) {
      // Here i came across a neighbor, what he is look like
      int2 neighborCoord = make_int2(x + dx, y + dy);
      if (!inRange(neighborCoord, size)) {
        continue;
      }
      float neighborConfidence = 1;
      if (inputConfidence) {
        neighborConfidence = inputConfidence[neighborCoord.y * size.x + neighborCoord.x];
      }
      if (neighborConfidence == 0) {
        continue;
      }

      float2 neighborflowOffset = inputFlow[neighborCoord.y * size.x + neighborCoord.x];
      if (neighborflowOffset.x == INVALID_FLOW_VALUE) {
        continue;
      }
      float weightImage = 1.0f;
      if (sigmaImage) {
        uint32_t imageColorNeighbor = inputImage[neighborCoord.y * size.x + neighborCoord.x];
        if (Image::RGBA::a(imageColorNeighbor) > 0 && Image::RGBA::a(imageColor) > 0) {
          const float sad = abs((float(Image::RGBA::r(imageColorNeighbor)) - Image::RGBA::r(imageColor)) / 255.0) +
                            abs((float(Image::RGBA::g(imageColorNeighbor)) - Image::RGBA::g(imageColor)) / 255.0) +
                            abs((float(Image::RGBA::b(imageColorNeighbor)) - Image::RGBA::b(imageColor)) / 255.0);
          weightImage = exp(-abs(sad * sad * sigmaImage));
        }
      }

      // Now calculate the distance between source and target
      float distSpace = length(make_float2(dx, dy)) / maxDist;
      float weightSpace = exp(-abs(distSpace * distSpace * sigmaSpace));
      // Now i do really look at the neighbor on the other side to see  how think is going on there
      float weight = weightSpace * weightImage * neighborConfidence;
      sumWeight += weight;
      sumContribution += weight * neighborflowOffset;
    }
  }

  // If my confidence is high, i would tend to keep mine, don't care about the neighbor's confidence
  // Here is where to set the weight
  if (sumWeight == 0) {
    outputFlow[y * size.x + x] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
  } else {
    outputFlow[y * size.x + x] = sumContribution / sumWeight;
  }
}

__global__ void flowConfidenceKernel(const int windowSize, const float gradientWeight, const int2 size0,
                                     const uint32_t* input0, const float* gradient0, const float2* inputFlow,
                                     const int2 size1, const uint32_t* input1, const float* gradient1,
                                     float* confidence) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size0.x || y >= size0.y) return;
  float2 flow = inputFlow[y * size0.x + x];
  if (flow.x == INVALID_FLOW_VALUE) {
    confidence[y * size0.x + x] = 0;
  } else {
    int2 mapCoord = make_int2(x + flow.x, y + flow.y);
    int2 sampleCoord = make_int2(x, y);
    confidence[y * size0.x + x] =
        getCUR(windowSize, gradientWeight, size0, input0, gradient0, sampleCoord, size1, input1, gradient1, mapCoord);
  }
}

Status SimpleFlow::findForwardFlow(const int flowSize, const int windowSize, const float flowMagnitudeWeight,
                                   const float gradientWeight, const int2 size0, const int2 offset0,
                                   const GPU::Buffer<const uint32_t> inputBuffer0,
                                   const GPU::Buffer<const float> inputGradientBuffer0, const int2 size1,
                                   const int2 offset1, const GPU::Buffer<const uint32_t> inputBuffer1,
                                   const GPU::Buffer<const float> inputGradientBuffer1, GPU::Buffer<float2> flow,
                                   GPU::Buffer<float> confidence, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(SIMPLEFLOW_KERNEL_BLOCK_SIZE_X, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size0.x, SIMPLEFLOW_KERNEL_BLOCK_SIZE_X),
               (unsigned)Cuda::ceilDiv(size0.y, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y), 1);
  forwardFlowKernel<<<dimGrid, dimBlock, 0, stream>>>(flowSize, windowSize, flowMagnitudeWeight, gradientWeight, size0,
                                                      offset0, inputBuffer0.get(), inputGradientBuffer0.get(), size1,
                                                      offset1, inputBuffer1.get(), inputGradientBuffer1.get(), 0,
                                                      flow.get(), confidence.get());

  return CUDA_STATUS;
}

__global__ void offsetCostKernel(const int2 flowOffset, const int flowSize, const float flowMagnitudeWeight,
                                 const float gradientWeight, const int2 size0, const int2 offset0,
                                 const uint32_t* input0, const float* gradient0, const int2 size1, const int2 offset1,
                                 const uint32_t* input1, const float* gradient1, float2* cost) {
  // Check whether we need to calculate the flow
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size0.x || y >= size0.y) return;

  uint32_t v0 = input0[y * size0.x + x];
  if (Image::RGBA::a(v0) == 0) {  // If current alpha is 0, do nothing
    cost[y * size0.x + x] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
    return;
  }
  // Try to find the best forward flow here
  int2 sampleCoord = make_int2(x, y);
  int2 mapCoord = make_int2(flowOffset.x + x + offset0.x - offset1.x, flowOffset.y + y + offset0.y - offset1.y);
  cost[y * size0.x + x] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
  if (inRange(mapCoord, size1)) {
    float difCost =
        getCost(0, gradientWeight, size0, input0, gradient0, sampleCoord, size1, input1, gradient1, mapCoord) +
        flowMagnitudeWeight * sqrtf(flowOffset.x * flowOffset.x + flowOffset.y * flowOffset.y) /
            sqrtf(2 * flowSize * flowSize);
    cost[y * size0.x + x] = make_float2(difCost, difCost);
  }
}

Status SimpleFlow::findOffsetCost(const int2 flowOffset, const int flowSize, const float flowMagnitudeWeight,
                                  const float gradientWeight, const int2 size0, const int2 offset0,
                                  const GPU::Buffer<const uint32_t> inputBuffer0,
                                  const GPU::Buffer<const float> inputGradientBuffer0, const int2 size1,
                                  const int2 offset1, const GPU::Buffer<const uint32_t> inputBuffer1,
                                  const GPU::Buffer<const float> inputGradientBuffer1, GPU::Buffer<float2> cost,
                                  GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(SIMPLEFLOW_KERNEL_BLOCK_SIZE_X, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size0.x, SIMPLEFLOW_KERNEL_BLOCK_SIZE_X),
               (unsigned)Cuda::ceilDiv(size0.y, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y), 1);
  offsetCostKernel<<<dimGrid, dimBlock, 0, stream>>>(
      flowOffset, flowSize, flowMagnitudeWeight, gradientWeight, size0, offset0, inputBuffer0.get(),
      inputGradientBuffer0.get(), size1, offset1, inputBuffer1.get(), inputGradientBuffer1.get(), cost.get());

  return CUDA_STATUS;
}

__global__ void updateBestCostKernel(const int2 flowOffset, const int2 size0, const float2* cost, float* bestCost,
                                     float2* bestOffset) {
  // Check whether we need to calculate the flow
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size0.x || y >= size0.y) return;
  if (flowOffset.x == 123456) {
    bestCost[y * size0.x + x] = MAX_INVALID_COST;
    bestOffset[y * size0.x + x] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
  }
  if (cost[y * size0.x + x].x >= 0 && cost[y * size0.x + x].x < bestCost[y * size0.x + x]) {
    bestCost[y * size0.x + x] = cost[y * size0.x + x].x;
    bestOffset[y * size0.x + x] = make_float2(flowOffset.x, flowOffset.y);
  }
}

Status SimpleFlow::updateBestCost(const int2 flowOffset, const int2 size0, const GPU::Buffer<const float2> cost,
                                  GPU::Buffer<float> bestCost, GPU::Buffer<float2> bestOffset, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(SIMPLEFLOW_KERNEL_BLOCK_SIZE_X, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size0.x, SIMPLEFLOW_KERNEL_BLOCK_SIZE_X),
               (unsigned)Cuda::ceilDiv(size0.y, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y), 1);
  updateBestCostKernel<<<dimGrid, dimBlock, 0, stream>>>(flowOffset, size0, cost.get(), bestCost.get(),
                                                         bestOffset.get());
  return CUDA_STATUS;
}

Status SimpleFlow::findBackwardAndForwardFlowAgreementConfidence(
    const int flowSize, const int2 size0, const int2 offset0, const GPU::Buffer<const float2> flow0,
    const GPU::Buffer<const float> confidence0, const int2 size1, const int2 offset1,
    const GPU::Buffer<const float2> flow1, const GPU::Buffer<const float> confidence1,
    GPU::Buffer<float> flowAgreementConfidence0, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(SIMPLEFLOW_KERNEL_BLOCK_SIZE_X, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size0.x, SIMPLEFLOW_KERNEL_BLOCK_SIZE_X),
               (unsigned)Cuda::ceilDiv(size0.y, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y), 1);

  flowAgreementConfidenceKernel<<<dimGrid, dimBlock, 0, stream>>>(flowSize, size0, offset0, flow0.get(),
                                                                  confidence0.get(), size1, offset1, flow1.get(),
                                                                  confidence1.get(), flowAgreementConfidence0.get());

  return CUDA_STATUS;
}

Status SimpleFlow::performConfidenceTransform(const int width, const int height, const float threshold,
                                              const float gamma, const float clampedValue,
                                              const GPU::Buffer<const float> inputConfidence,
                                              GPU::Buffer<float> outputConfidence, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(SIMPLEFLOW_KERNEL_BLOCK_SIZE_X, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(width, SIMPLEFLOW_KERNEL_BLOCK_SIZE_X),
               (unsigned)Cuda::ceilDiv(height, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y), 1);
  confidenceTransformKernel<<<dimGrid, dimBlock, 0, stream>>>(width, height, threshold, gamma, clampedValue,
                                                              inputConfidence.get(), outputConfidence.get());

  return CUDA_STATUS;
}

Status SimpleFlow::findConfidence(const int windowSize, const float gradientWeight, const int2 size0,
                                  const GPU::Buffer<const uint32_t> input0, const GPU::Buffer<const float> gradient0,
                                  GPU::Buffer<const float2> forwardFlow0, const int2 size1,
                                  const GPU::Buffer<const uint32_t> input1, const GPU::Buffer<const float> gradient1,
                                  GPU::Buffer<float> confidence, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(SIMPLEFLOW_KERNEL_BLOCK_SIZE_X, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size0.x, SIMPLEFLOW_KERNEL_BLOCK_SIZE_X),
               (unsigned)Cuda::ceilDiv(size0.y, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y), 1);
  flowConfidenceKernel<<<dimGrid, dimBlock, 0, stream>>>(windowSize, gradientWeight, size0, input0.get(),
                                                         gradient0.get(), forwardFlow0.get(), size1, input1.get(),
                                                         gradient1.get(), confidence.get());

  return CUDA_STATUS;
}

Status SimpleFlow::performConfidenceAwareFlowInterpolation(const bool extrapolation, const int2 size,
                                                           const int kernelSize, const float sigmaSpace,
                                                           const float sigmaImage, const float sigmaConfidence,
                                                           const GPU::Buffer<const uint32_t> inputImage,
                                                           const GPU::Buffer<const float2> inputFlow,
                                                           const GPU::Buffer<const float> inputConfidence,
                                                           GPU::Buffer<float2> outputFlow, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(SIMPLEFLOW_KERNEL_BLOCK_SIZE_X, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, SIMPLEFLOW_KERNEL_BLOCK_SIZE_X),
               (unsigned)Cuda::ceilDiv(size.y, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y), 1);
  if (inputConfidence.wasAllocated()) {
    confidenceAwareFlowBlurKernel<<<dimGrid, dimBlock, 0, stream>>>(
        extrapolation, size, kernelSize, sigmaSpace, sigmaImage, sigmaConfidence, inputImage.get(), inputFlow.get(),
        inputConfidence.get(), outputFlow.get());
  } else {
    confidenceAwareFlowBlurKernel<<<dimGrid, dimBlock, 0, stream>>>(extrapolation, size, kernelSize, sigmaSpace,
                                                                    sigmaImage, sigmaConfidence, inputImage.get(),
                                                                    inputFlow.get(), nullptr, outputFlow.get());
  }
  return CUDA_STATUS;
}

__global__ void temporalAwareFlowBlurKernel(const bool extrapolation, const int frameId, const int frameCount,
                                            const int2 size, const int kernelSize, const float sigmaSpace,
                                            const float sigmaImage, const float sigmaTime, const float* const frames,
                                            const uint32_t* const inputImages, const float2* const inputFlows,
                                            const float* const inputConfidences, float2* const outputFlow) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size.x || y >= size.y) return;
  int baseOffset = frameId * size.x * size.y;
  if (extrapolation) {
    float2 inFlow = inputFlows[baseOffset + y * size.x + x];
    if (inFlow.x != INVALID_FLOW_VALUE) {
      outputFlow[y * size.x + x] = inFlow;
      return;
    }
  }
  // check if the current flow is not valid, then just do nothing
  float maxDist = kernelSize * 1.4142;
  uint32_t imageColor;
  if (sigmaImage) {
    imageColor = inputImages[baseOffset + y * size.x + x];
  }
  if (!extrapolation) {
    if (sigmaImage > 0) {
      if (!Image::RGBA::a(imageColor)) {
        outputFlow[y * size.x + x] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
        return;
      }
    }
  }
  float sumWeight = 0;
  float2 sumContribution = make_float2(0, 0);

  if (maxDist == 0) maxDist = 1;
  for (int t = 0; t < frameCount; t++)
    if (frames[t] >= 0) {
      for (int dx = -kernelSize; dx <= kernelSize; dx++)
        for (int dy = -kernelSize; dy <= kernelSize; dy++) {
          const int offset = t * size.x * size.y;
          // Here i came across a neighbor, what he is look like
          int2 neighborCoord = make_int2(x + dx, y + dy);
          if (!inRange(neighborCoord, size)) {
            continue;
          }
          float neighborConfidence = 1;
          if (inputConfidences) {
            neighborConfidence = inputConfidences[offset + neighborCoord.y * size.x + neighborCoord.x];
          }
          if (neighborConfidence == 0) {
            continue;
          }
          float2 neighborflowOffset = inputFlows[offset + neighborCoord.y * size.x + neighborCoord.x];
          if (neighborflowOffset.x == INVALID_FLOW_VALUE) {
            continue;
          }
          float weightImage = 1.0f;
          if (sigmaImage) {
            uint32_t imageColorNeighbor = inputImages[offset + neighborCoord.y * size.x + neighborCoord.x];
            if (Image::RGBA::a(imageColorNeighbor) > 0 && Image::RGBA::a(imageColor) > 0) {
              const float sad = abs((float(Image::RGBA::r(imageColorNeighbor)) - Image::RGBA::r(imageColor)) / 255.0) +
                                abs((float(Image::RGBA::g(imageColorNeighbor)) - Image::RGBA::g(imageColor)) / 255.0) +
                                abs((float(Image::RGBA::b(imageColorNeighbor)) - Image::RGBA::b(imageColor)) / 255.0);
              weightImage = exp(-abs(sad * sad * sigmaImage));
            }
          }
          // Now calculate the distance of time
          float distTime = float(frames[t] - frames[frameId]) / frameCount;
          float weightTime = exp(-abs(distTime * distTime * sigmaTime));
          // Now calculate the distance between source and target
          float distSpace = length(make_float2(dx, dy)) / maxDist;
          float weightSpace = exp(-abs(distSpace * distSpace * sigmaSpace));
          // Now i do really look at the neighbor on the other side to see  how think is going on there
          float weight = weightSpace * weightImage * weightTime * neighborConfidence;
          sumWeight += weight;
          sumContribution += weight * neighborflowOffset;
        }
    }
  // If my confidence is high, i would tend to keep mine, don't care about the neighbor's confidence
  // Here is where to set the weight
  if (sumWeight == 0) {
    outputFlow[y * size.x + x] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
  } else {
    outputFlow[y * size.x + x] = sumContribution / sumWeight;
  }
}

Status SimpleFlow::performTemporalAwareFlowInterpolation(
    const bool extrapolation, const frameid_t frameId, const int2 size, const int kernelSize, const float sigmaSpace,
    const float sigmaImage, const float sigmaTime, const GPU::Buffer<const float> frames,
    const GPU::Buffer<const uint32_t> inputImages, const GPU::Buffer<const float2> inputFlows,
    const GPU::Buffer<const float> inputConfidences, GPU::Buffer<float2> outputFlow, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(SIMPLEFLOW_KERNEL_BLOCK_SIZE_X, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, SIMPLEFLOW_KERNEL_BLOCK_SIZE_X),
               (unsigned)Cuda::ceilDiv(size.y, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y), 1);
  const int frameCount = (int)frames.numElements();
  temporalAwareFlowBlurKernel<<<dimGrid, dimBlock, 0, stream>>>(
      extrapolation, frameId, frameCount, size, kernelSize, sigmaSpace, sigmaImage, sigmaTime, frames.get(),
      inputImages.get(), inputFlows.get(), inputConfidences.get(), outputFlow.get());
  return CUDA_STATUS;
}

Status SimpleFlow::performFlowJittering(const int jitterSize, const int windowSize, const float flowMagnitudeWeight,
                                        const float gradientWeight, const int2 size0, const int2 offset0,
                                        const GPU::Buffer<const uint32_t> inputBuffer0,
                                        const GPU::Buffer<const float> inputGradientBuffer0, const int2 size1,
                                        const int2 offset1, const GPU::Buffer<const uint32_t> inputBuffer1,
                                        const GPU::Buffer<const float> inputGradientBuffer1,
                                        const GPU::Buffer<const float2> inputFlow, GPU::Buffer<float2> outputFlow,
                                        GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(SIMPLEFLOW_KERNEL_BLOCK_SIZE_X, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size0.x, SIMPLEFLOW_KERNEL_BLOCK_SIZE_X),
               (unsigned)Cuda::ceilDiv(size0.y, SIMPLEFLOW_KERNEL_BLOCK_SIZE_Y), 1);

  forwardFlowKernel<<<dimGrid, dimBlock, 0, stream>>>(jitterSize, windowSize, flowMagnitudeWeight, gradientWeight,
                                                      size0, offset0, inputBuffer0.get(), inputGradientBuffer0.get(),
                                                      size1, offset1, inputBuffer1.get(), inputGradientBuffer1.get(),
                                                      inputFlow.get(), outputFlow.get(), 0);
  return CUDA_STATUS;
}

}  // namespace Core
}  // namespace VideoStitch
