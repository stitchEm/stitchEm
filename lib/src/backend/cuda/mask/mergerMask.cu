// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "mask/mergerMask.hpp"

#include "backend/common/imageOps.hpp"

#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "backend/cuda/surface.hpp"
#include "cuda/util.hpp"
#include "gpu/core1/voronoi.hpp"
#include "gpu/memcpy.hpp"
#include "mask/mergerMaskConstant.hpp"

namespace VideoStitch {
namespace MergerMask {
#define MERGER_MASK_KERNEL_SIZE_X 16
#define MERGER_MASK_KERNEL_SIZE_Y 16

__global__ void updateInputIndexByDistortionMapKernel(
    const videoreaderid_t camId, const unsigned char distortionThreshold, const int2 camSize, const int2 camOffset,
    const unsigned char* __restrict__ camDistortionBuffer, const int2 inputSize,
    const uint32_t* __restrict__ inputNonOverlappingIndexBuffer,
    const unsigned char* __restrict__ inputDistortionBuffer, uint32_t* __restrict__ nextNonOverlappingIndexBuffer,
    unsigned char* __restrict__ nextDistortionBuffer) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < inputSize.x && y < inputSize.y) {
    const int inputIndex = y * inputSize.x + x;
    const unsigned char inputDistortion = inputDistortionBuffer[inputIndex];
    const uint32_t inputNonOverlappingIndex = inputNonOverlappingIndexBuffer[inputIndex];
    const int camX = (x - camOffset.x + inputSize.x) % inputSize.x;
    const int camY = y - camOffset.y;
    nextNonOverlappingIndexBuffer[inputIndex] = inputNonOverlappingIndex;
    nextDistortionBuffer[inputIndex] = inputDistortion;
    if (camX >= 0 && camX < camSize.x && camY >= 0 && camY < camSize.y) {
      const int camIndex = camY * camSize.x + camX;
      const unsigned char camDistortion = camDistortionBuffer[camIndex];
      if ((camDistortion < inputDistortion && inputDistortion > distortionThreshold) ||
          (inputNonOverlappingIndex == 0 && camDistortion < 255)) {
        nextNonOverlappingIndexBuffer[inputIndex] = 1 << camId;
        nextDistortionBuffer[inputIndex] = camDistortion;
      }
    }
  }
}

Status MergerMask::updateInputIndexByDistortionMap(const videoreaderid_t camId, const int2 inputSize,
                                                   const GPU::Buffer<const uint32_t> inputNonOverlappingIndexBuffer,
                                                   const GPU::Buffer<const unsigned char> inputDistortionBuffer,
                                                   GPU::Buffer<uint32_t> nextNonOverlappingIndexBuffer,
                                                   GPU::Buffer<unsigned char> nextDistortionBuffer, GPU::Stream stream,
                                                   const bool original) {
  const int2 camSize =
      original ? make_int2((int)cachedOriginalMappedRects[camId].getWidth(),
                           (int)cachedOriginalMappedRects[camId].getHeight())
               : make_int2((int)cachedMappedRects[camId].getWidth(), (int)cachedMappedRects[camId].getHeight());
  const int2 camOffset =
      original ? make_int2((int)cachedOriginalMappedRects[camId].left(), (int)cachedOriginalMappedRects[camId].top())
               : make_int2((int)cachedMappedRects[camId].left(), (int)cachedMappedRects[camId].top());
  const unsigned char distortionThreshold = mergerMaskConfig.getDistortionThreshold();
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(inputSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(inputSize.y, dimBlock.y), 1);
  updateInputIndexByDistortionMapKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      camId, distortionThreshold, camSize, camOffset,
      original ? originalDistortionMaps[camId].get() : distortionMaps[camId].get(), inputSize,
      inputNonOverlappingIndexBuffer.get(), inputDistortionBuffer.get(), nextNonOverlappingIndexBuffer.get(),
      nextDistortionBuffer.get());
  return CUDA_STATUS;
}

__global__ void updateDistortionFromMaskKernel(videoreaderid_t camId, const int2 camSize, const int2 camOffset,
                                               unsigned char* __restrict__ camDistortionBuffer, const int2 inputSize,
                                               const uint32_t* __restrict__ srcMapBuffer) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < camSize.x && y < camSize.y) {
    int inputX = (x + camOffset.x) % inputSize.x;
    int inputY = (y + camOffset.y);
    if (inputX >= 0 && inputX < inputSize.x && inputY >= 0 && inputY < inputSize.y) {
      for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
          const int neiX = inputX + i;
          const int neiY = inputY + j;
          if (neiX >= 0 && neiX < inputSize.x && neiY >= 0 && neiY < inputSize.y) {
            if ((srcMapBuffer[neiY * inputSize.x + neiX] & (1 << camId)) == 0) {
              camDistortionBuffer[y * camSize.x + x] = 255;
              return;
            }
          }
        }
      }
    }
  }
}

Status MergerMask::updateDistortionFromMask(const videoreaderid_t camId, const int2 distortionBufferSize,
                                            const int2 distortionBufferOffset,
                                            GPU::Buffer<unsigned char> distortionBuffer, const int2 inputSize,
                                            const GPU::Buffer<const uint32_t> srcMap, GPU::Stream stream) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(distortionBufferSize.x, dimBlock.x),
               (unsigned)Cuda::ceilDiv(distortionBufferSize.y, dimBlock.y), 1);
  updateDistortionFromMaskKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      camId, distortionBufferSize, distortionBufferOffset, distortionBuffer.get(), inputSize, srcMap.get());
  return CUDA_STATUS;
}

__global__ void initializeMasksKernel(videoreaderid_t camId, const int2 camSize, const int2 camOffset,
                                      const unsigned char* __restrict__ camDistortionBuffer, const int2 inputSize,
                                      uint32_t* __restrict__ inputNonOverlappingIndexBuffer,
                                      unsigned char* inputDistortionBuffer) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < inputSize.x && y < inputSize.y) {
    const int inputIndex = y * inputSize.x + x;
    const int camX = (x - camOffset.x + inputSize.x) % inputSize.x;
    const int camY = y - camOffset.y;
    inputNonOverlappingIndexBuffer[inputIndex] = 0;
    inputDistortionBuffer[inputIndex] = 255;
    if (camX >= 0 && camX < camSize.x && camY >= 0 && camY < camSize.y) {
      unsigned char camDistortion = camDistortionBuffer[camY * camSize.x + camX];
      if (camDistortion < 255) {
        inputDistortionBuffer[inputIndex] = camDistortion;
        inputNonOverlappingIndexBuffer[inputIndex] = 1 << camId;
      }
    }
  }
}

Status MergerMask::initializeMasks(const int2 inputSize, const videoreaderid_t camId,
                                   GPU::Buffer<uint32_t> inputNonOverlappingIndexBuffer,
                                   GPU::Buffer<unsigned char> inputDistortionBuffer, GPU::Stream stream,
                                   const bool original) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);
  const Core::Rect camRect = original ? cachedOriginalMappedRects[camId] : cachedMappedRects[camId];
  dim3 dimGrid((unsigned)Cuda::ceilDiv(inputSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(inputSize.y, dimBlock.y), 1);
  initializeMasksKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      camId, make_int2((int)camRect.getWidth(), (int)camRect.getHeight()),
      make_int2((int)camRect.left(), (int)camRect.top()),
      original ? originalDistortionMaps[camId].get() : distortionMaps[camId].get(), inputSize,
      inputNonOverlappingIndexBuffer.get(), inputDistortionBuffer.get());
  return CUDA_STATUS;
}

__global__ void transformDistortionKernel(const int2 inputSize, const float distortionParam,
                                          unsigned char* __restrict__ distortionBuffer) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < inputSize.x && y < inputSize.y) {
    const unsigned index = y * inputSize.x + x;
    const unsigned char inputDistortion = distortionBuffer[index];
    unsigned char remappedDistortion = (unsigned char)(pow(float(inputDistortion) / 255.0f, distortionParam) * 255.0f);
    distortionBuffer[index] = remappedDistortion;
  }
}

Status MergerMask::transformDistortion(const int2 inputSize, GPU::Buffer<unsigned char> distortionBuffer,
                                       GPU::Stream stream) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(inputSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(inputSize.y, dimBlock.y), 1);
  const float distortionParam = mergerMaskConfig.getDistortionParam();
  transformDistortionKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(inputSize, distortionParam, distortionBuffer.get());
  return CUDA_STATUS;
}

__global__ void updateIndexMaskKernel(const videoreaderid_t camId, const int maxOverlappingCount,
                                      const char* const __restrict__ cameraIndices, const int2 distortionBufferSize,
                                      const int2 distortionBufferOffset,
                                      const unsigned char* const __restrict__ distortionBuffer, const int2 size,
                                      uint32_t* __restrict__ inputIndexBuffer, unsigned char* __restrict__ mask,
                                      const uint32_t* const __restrict__ srcMap) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    const unsigned index = y * size.x + x;
    uint32_t inputIndex = inputIndexBuffer[index];
    if ((mask[index] < 255) && (srcMap[index] & (1 << camId))) {
      int coordX = (x - distortionBufferOffset.x + size.x) % size.x;
      int coordY = y - distortionBufferOffset.y;
      // Make sure to only put pixels those are not very distorted
      if (coordX >= 0 && coordX < distortionBufferSize.x && coordY >= 0 && coordY < distortionBufferSize.y) {
        unsigned char distortion = distortionBuffer[coordY * distortionBufferSize.x + coordX];
        // if this pixel is already occupied and the distortion is large, just ignore it
        if (inputIndex > 0 && distortion > 130) {
          mask[index] = 0;
          return;
        }
      }

      int countBitOne = 0;
      int count = 0;
      int minCount = -1;
      while (inputIndex > 0) {
        if (inputIndex & 1) {
          countBitOne++;
          if (minCount < 0) {
            minCount = count;
          } else if (cameraIndices[count] < cameraIndices[minCount]) {
            minCount = count;
          }
        }
        inputIndex = inputIndex >> 1;
        count++;
      }
      if (countBitOne < maxOverlappingCount) {
        inputIndexBuffer[index] |= (1 << camId);
      } else if (countBitOne == maxOverlappingCount) {
        inputIndexBuffer[index] = (inputIndexBuffer[index] - (1 << minCount)) | (1 << camId);
      }
      mask[index] = 255;
    } else {
      mask[index] = 0;
    }
  }
}

Status MergerMask::updateIndexMask(const videoreaderid_t camId, const int maxOverlappingCount,
                                   const GPU::Buffer<const char> cameraIndices, const int2 distortionBufferSize,
                                   const int2 distortionBufferOffset,
                                   const GPU::Buffer<const unsigned char> distortionBuffer, const int2 inputSize,
                                   GPU::Buffer<uint32_t> inputIndexBuffer, GPU::Buffer<unsigned char> mask,
                                   const GPU::Buffer<const uint32_t> srcMap, GPU::Stream stream) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);

  dim3 dimGrid((unsigned)Cuda::ceilDiv(inputSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(inputSize.y, dimBlock.y), 1);
  updateIndexMaskKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      camId, maxOverlappingCount, cameraIndices.get(), distortionBufferSize, distortionBufferOffset,
      distortionBuffer.get(), inputSize, inputIndexBuffer.get(), mask.get(), srcMap.get());
  return CUDA_STATUS;
}

// Get the first 1 bit (from right to left), set it to 0
// For example input number = 1000100 --> return 2 and set number = 1000
// @NOTE: Faster implementation can be found at : https://graphics.stanford.edu/~seander/bithacks.html
__device__ int getFirstOnBitPosition(uint32_t& number) {
  int x = -1;
  int count = 0;
  while ((number > 0) && ((number & 1) == 0)) {
    count++;
    number = number >> 1;
  }
  if ((number & 1) > 0) {
    x = count;
    number = number >> 1;
  }
  return x;
}

// Get index of the first two bit with value 1
__device__ int2 getFirstTwoOnBitPosition(const uint32_t input) {
  uint32_t number = input;
  int32_t x = getFirstOnBitPosition(number);
  int32_t y = -1;
  if (x >= 0) {
    int32_t offsetY = getFirstOnBitPosition(number);
    if (offsetY >= 0) {
      y = x + offsetY + 1;
    }
  }
  return make_int2(x, y);
}

__global__ void updateStitchingCostKernel(const size_t camCount, const int2 size, const int kernelSize,
                                          const uint32_t* __restrict__ inputIndexBuffer,
                                          const uint32_t* __restrict__ mappedOffset,
                                          const int2* __restrict__ mappedRectOffset,
                                          const int2* __restrict__ mappedRectSize,
                                          const uint32_t* __restrict__ mappedBuffer, float* __restrict__ cost,
                                          uint32_t* __restrict__ debugBuffer0, uint32_t* __restrict__ debugBuffer1) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    const unsigned index = y * size.x + x;
    debugBuffer0[index] = 0;
    debugBuffer1[index] = 0;
    const uint32_t inputIndex = inputIndexBuffer[index];
    const int2 firstTwo = getFirstTwoOnBitPosition(inputIndex);

    if (firstTwo.x >= 0 && firstTwo.y >= 0) {
      const int input0 = firstTwo.x;
      const int input1 = firstTwo.y;
      debugBuffer1[index] = 1 << input0 + 1 << input1;

      int x0 = (x - mappedRectOffset[input0].x + size.x) % size.x;
      int y0 = y - mappedRectOffset[input0].y;
      const unsigned index0 = y0 * mappedRectSize[input0].x + x0;
      const uint32_t color0 = mappedBuffer[mappedOffset[input0] + index0];
      debugBuffer0[index] = color0;

      int x1 = x - mappedRectOffset[input1].x;
      int y1 = y - mappedRectOffset[input1].y;
      const unsigned index1 = y1 * mappedRectSize[input1].x + x1;
      const uint32_t color1 = mappedBuffer[mappedOffset[input1] + index1];
      debugBuffer1[index] = color1;

      // Update stitching cost using min pooling metric
      const int left = max(x1 - kernelSize, 0);
      const int right = min(x1 + kernelSize, int(mappedRectSize[input1].x - 1));
      const int top = max(y1 - kernelSize, 0);
      const int bottom = min(y1 + kernelSize, int(mappedRectSize[input1].y - 1));
      float sadMin = -1;
      for (int i = left; i <= right; i++) {
        for (int j = top; j <= bottom; j++) {
          const unsigned warpI = (i + size.x) % size.x;
          const unsigned index1 = j * mappedRectSize[input1].x + warpI;
          const uint32_t color1 = mappedBuffer[mappedOffset[input1] + index1];
          if (color1 != INVALID_VALUE) {
            const float sadLab = abs((float(Image::RGBA::r(color0)) - Image::RGBA::r(color1)) / 255.0) +
                                 abs((float(Image::RGBA::g(color0)) - Image::RGBA::g(color1)) / 255.0) +
                                 abs((float(Image::RGBA::b(color0)) - Image::RGBA::b(color1)) / 255.0);
            const float sadGradient = abs((float(Image::RGBA::a(color0)) - Image::RGBA::a(color1)) / 255.0);

            const float sad = (sadLab + 2.0f * sadGradient) / (1.0f + 2.0f);
            if (sad < sadMin || sadMin < 0) {
              sadMin = sad;
            }
          }
        }
      }
      if (sadMin >= 0) {
        // Prefer to focus all effort in the middle of the output panorama, give these pixels more weights
        float yDistance = min(1.0f, (float)(abs((size.y / 2) - y)) / (size.y / 2));
        float yCost = max(0.0f, expf(yDistance * yDistance * (-0.5f)));
        cost[index] += max(sadMin * yCost, 0.001);
      }
    }
  }
}

Status MergerMask::updateStitchingCost(const int2 inputSize, const int kernelSize,
                                       const GPU::Buffer<const uint32_t> inputIndexBuffer,
                                       const GPU::Buffer<const uint32_t> mappedOffset,
                                       const GPU::Buffer<const int2> mappedRectOffset,
                                       const GPU::Buffer<const int2> mappedRectSize,
                                       const GPU::Buffer<const uint32_t> mappedBuffer, GPU::Buffer<float> cost,
                                       GPU::Buffer<uint32_t> debugBuffer0, GPU::Buffer<uint32_t> debugBuffer1,
                                       GPU::Stream stream) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);

  dim3 dimGrid((unsigned)Cuda::ceilDiv(inputSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(inputSize.y, dimBlock.y), 1);
  updateStitchingCostKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      pano.numInputs(), inputSize, kernelSize, inputIndexBuffer.get(), mappedOffset.get(), mappedRectOffset.get(),
      mappedRectSize.get(), mappedBuffer.get(), cost.get(), debugBuffer0.get(), debugBuffer1.get());
  return CUDA_STATUS;
}

__global__ void extractLayerFromIndexBufferKernel(const videoreaderid_t id, int2 bufferSize,
                                                  const uint32_t* const __restrict__ inputBuffer,
                                                  uint32_t* __restrict__ extractedBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < bufferSize.x && y < bufferSize.y) {
    const unsigned index = y * bufferSize.x + x;
    const uint32_t input = inputBuffer[index];
    if ((input & id) > 0) {
      extractedBuffer[index] = id;
    } else {
      extractedBuffer[index] = 0;
    }
  }
}

Status MergerMask::extractLayerFromIndexBuffer(const videoreaderid_t id, const int2 bufferSize,
                                               const GPU::Buffer<const uint32_t> inputIndexBuffer,
                                               GPU::Buffer<uint32_t> extractedBuffer, GPU::Stream stream) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(bufferSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(bufferSize.y, dimBlock.y), 1);
  extractLayerFromIndexBufferKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(id, bufferSize, inputIndexBuffer.get(),
                                                                            extractedBuffer.get());
  return CUDA_STATUS;
}

__global__ void updateIndexMaskAfterSeamKernel(const videoreaderid_t id0s, const videoreaderid_t id1, int2 bufferSize,
                                               const unsigned char* const __restrict__ seamBuffer,
                                               uint32_t* __restrict__ indexBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < bufferSize.x && y < bufferSize.y) {
    const unsigned index = y * bufferSize.x + x;
    uint32_t input = indexBuffer[index];
    const uint32_t seam = seamBuffer[index];
    if (seam == (1 << 0)) {
      if ((input & id1) == id1) {
        input -= id1;
      }
    } else if (seam == (1 << 1)) {
      if ((input & id0s) > 0) {
        input = (input & (~id0s));
      }
    }
    indexBuffer[index] = input;
  }
}

Status MergerMask::updateIndexMaskAfterSeam(const videoreaderid_t id0s, const videoreaderid_t id1,
                                            const int2 bufferSize, const GPU::Buffer<const unsigned char> seamBuffer,
                                            GPU::Buffer<uint32_t> indexBuffer, GPU::Stream stream) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(bufferSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(bufferSize.y, dimBlock.y), 1);
  updateIndexMaskAfterSeamKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(id0s, id1, bufferSize, seamBuffer.get(),
                                                                         indexBuffer.get());
  return CUDA_STATUS;
}

__global__ void lookupColorBufferFromInputIndexKernel(
    const int wrapWidth, const int camCount, const unsigned char* const __restrict__ cameraIndices,
    const int2* __restrict__ const mappedRectOffsets, const int2* __restrict__ const mappedRectSizes,
    const uint32_t* __restrict__ const mappedOffsets, const uint32_t* __restrict__ const mappedBuffer,
    const int2 bufferSize, const uint32_t* const __restrict__ inputIndexBuffer, uint32_t* __restrict__ outputBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < bufferSize.x && y < bufferSize.y) {
    const unsigned index = y * bufferSize.x + x;
    outputBuffer[index] = INVALID_VALUE;

    // If the signal is on
    const uint32_t inputIndex = inputIndexBuffer[index];
    for (int i = camCount - 1; i >= 0; i--)
      if ((inputIndex & (1 << cameraIndices[i])) > 0) {
        unsigned char camIndex = cameraIndices[i];
        uint32_t camOffset = mappedOffsets[camIndex];
        int2 camRectOffset = mappedRectOffsets[camIndex];
        int2 camRectSize = mappedRectSizes[camIndex];
        int32_t camX = (x - camRectOffset.x + wrapWidth) % wrapWidth;
        int32_t camY = y - camRectOffset.y;
        if (camX >= 0 && camX < camRectSize.x && camY >= 0 && camY < camRectSize.y) {
          int32_t camIndex = camY * camRectSize.x + camX;
          outputBuffer[index] = mappedBuffer[camOffset + camIndex];
        }
        break;
      }
  }
}

Status MergerMask::lookupColorBufferFromInputIndex(
    const int wrapWidth, const GPU::Buffer<const unsigned char> camBuffer,
    const GPU::Buffer<const int2> mappedRectOffsets, const GPU::Buffer<const int2> mappedRectSizes,
    const GPU::Buffer<const uint32_t> mappedOffsets, const GPU::Buffer<const uint32_t> mappedBuffers,
    const int2 bufferSize, const GPU::Buffer<const uint32_t> inputIndexBuffer, GPU::Buffer<uint32_t> outputBuffer,
    GPU::Stream stream) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(bufferSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(bufferSize.y, dimBlock.y), 1);
  lookupColorBufferFromInputIndexKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      wrapWidth, (int)camBuffer.numElements(), camBuffer.get(), mappedRectOffsets.get(), mappedRectSizes.get(),
      mappedOffsets.get(), mappedBuffers.get(), bufferSize, inputIndexBuffer.get(), outputBuffer.get());
  return CUDA_STATUS;
}

__global__ void updateSeamMaskKernel(const videoreaderid_t id, const int2 size,
                                     const uint32_t* __restrict__ const originalInputIndexBuffer,
                                     const unsigned char* const __restrict__ distanceBuffer,
                                     uint32_t* __restrict__ seamOuputIndexBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    const unsigned index = y * size.x + x;
    if ((originalInputIndexBuffer[index] & (1 << id)) > 0) {
      if (distanceBuffer[index] < 255) {
        seamOuputIndexBuffer[index] |= (1 << id);
      }
    }
  }
}

Status MergerMask::updateSeamMask(const videoreaderid_t id, const int2 size,
                                  const GPU::Buffer<const uint32_t> originalInputIndexBuffer,
                                  const GPU::Buffer<const unsigned char> distanceBuffer,
                                  GPU::Buffer<uint32_t> seamOuputIndexBuffer, GPU::Stream stream) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, dimBlock.x), (unsigned)Cuda::ceilDiv(size.y, dimBlock.y), 1);
  updateSeamMaskKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(id, size, originalInputIndexBuffer.get(),
                                                               distanceBuffer.get(), seamOuputIndexBuffer.get());
  return CUDA_STATUS;
}

__global__ void getInputMaskFromOutputIndicesKernel(const videoreaderid_t imId, const int scaleFactor,
                                                    const int2 outputSize,
                                                    const uint32_t* __restrict__ const maskBuffer, const int2 inputSize,
                                                    const float2* __restrict__ const inputCoordBuffer,
                                                    unsigned char* const __restrict__ inputMask) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < scaleFactor * inputSize.x && y < scaleFactor * inputSize.y) {
    const unsigned index = y * inputSize.x * scaleFactor + x;
    const float2 coord = inputCoordBuffer[index];
    if (coord.x < 0 || coord.y < 0) {
      return;
    }
    const int2 roundedCoord = make_int2(roundf(coord.x), roundf(coord.y));
    inputMask[index] = 0;
    if (roundedCoord.x >= 0 && roundedCoord.x < outputSize.x && roundedCoord.y >= 0 && roundedCoord.y < outputSize.y) {
      if ((maskBuffer[roundedCoord.y * outputSize.x + (roundedCoord.x % outputSize.x)] & (1 << imId)) > 0) {
        inputMask[index] = 255;
      }
    }
  }
}

Status MergerMask::getInputMaskFromOutputIndices(const videoreaderid_t imId, const int scaleFactor,
                                                 const int2 outputSize, const GPU::Buffer<const uint32_t> maskBuffer,
                                                 const int2 inputSize, const GPU::Buffer<const float2> inputCoordBuffer,
                                                 GPU::Buffer<unsigned char> inputMask, GPU::Stream stream) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(scaleFactor * inputSize.x, dimBlock.x),
               (unsigned)Cuda::ceilDiv(scaleFactor * inputSize.y, dimBlock.y), 1);
  FAIL_RETURN(
      GPU::memsetToZeroBlocking<unsigned char>(inputMask, inputSize.x * inputSize.y * scaleFactor * scaleFactor));
  getInputMaskFromOutputIndicesKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      imId, scaleFactor, outputSize, maskBuffer.get(), inputSize, inputCoordBuffer.get(), inputMask.get());
  return CUDA_STATUS;
}

__global__ void getOutputIndicesFromInputMaskKernel(const videoreaderid_t imId, const int scaleFactor,
                                                    const int2 inputSize,
                                                    const unsigned char* const __restrict__ inputMask,
                                                    const int2 outputSize, cudaTextureObject_t coordBuffer,
                                                    uint32_t* __restrict__ const maskBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < outputSize.x && y < outputSize.y) {
    const unsigned index = y * outputSize.x + x;
    const float2 coord = tex2D<float2>(coordBuffer, x, y);
    const int2 roundedCoord = make_int2(roundf(coord.x * scaleFactor), roundf(coord.y * scaleFactor));
    if (roundedCoord.x >= 0 && roundedCoord.x < scaleFactor * inputSize.x && roundedCoord.y >= 0 &&
        roundedCoord.y < scaleFactor * inputSize.y) {
      if (inputMask[roundedCoord.y * (scaleFactor * inputSize.x) + roundedCoord.x] > 0) {
        maskBuffer[index] |= (1 << imId);
      }
    }
  }
}

Status MergerMask::getOutputIndicesFromInputMask(const videoreaderid_t imId, const int scaleFactor,
                                                 const int2 inputSize, const GPU::Buffer<const unsigned char> inputMask,
                                                 const int2 outputSize, const GPU::Surface& coordBuffer,
                                                 GPU::Buffer<uint32_t> maskBuffer, GPU::Stream stream) {
  dim3 dimBlock(MERGER_MASK_KERNEL_SIZE_X, MERGER_MASK_KERNEL_SIZE_Y, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(outputSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(outputSize.y, dimBlock.y), 1);
  getOutputIndicesFromInputMaskKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(
      imId, scaleFactor, inputSize, inputMask.get(), outputSize, coordBuffer.get().texture(), maskBuffer.get());
  return CUDA_STATUS;
}

}  // namespace MergerMask
}  // namespace VideoStitch
