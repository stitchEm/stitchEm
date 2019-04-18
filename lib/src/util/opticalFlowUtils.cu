// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "opticalFlowUtils.hpp"

#include "backend/common/vectorOps.hpp"
#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"
#include "gpu/image/sampling.hpp"
#include "gpu/image/imageOps.hpp"
#include "gpu/image/blur.hpp"
#include "gpu/stream.hpp"
#include "cuda/error.hpp"
#include "cuda/util.hpp"
#include "util/imageProcessingGPUUtils.hpp"

#include "backend/cuda/core1/kernels/samplingKernel.cu"
#include "parallax/gpu/cuda/kernels/patchDifferenceFunction.cu"

#define REGULARIZATION_TILE_WIDTH 16
#define KERNEL_SIZE 25
#define AREA_SIZE (REGULARIZATION_TILE_WIDTH + 2 * KERNEL_SIZE)
#define TILE_WIDTH 16
#define CUDABLOCKSIZE 512

namespace VideoStitch {
namespace Util {

__global__ void backwardCoordLookupKernel(const int2 inputOffset, int2 inputSize, const float2* g_iCoord,
                                          const int2 outputOffset, int2 outputSize, float2* g_oCoord) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < inputSize.x && y < inputSize.y) {
    const float2 iCoord = g_iCoord[y * inputSize.x + x];
    if (iCoord.x != INVALID_FLOW_VALUE) {
      const float2 inputCoord = make_float2(inputOffset.x + x, inputOffset.y + y);
      const float2 outputCoord = inputCoord + iCoord;
      const float2 relativeOutputCoord = outputCoord - make_float2(outputOffset.x, outputOffset.y);
      if (inRange(relativeOutputCoord, outputSize)) {
        g_oCoord[int(round(relativeOutputCoord.y)) * outputSize.x + int(round(relativeOutputCoord.x))] =
            make_float2(0, 0) - iCoord;
      }
    }
  }
}

Status OpticalFlow::backwardCoordLookup(const int2 inputOffset, const int2 inputSize,
                                        const GPU::Buffer<const float2> inputCoordBuffer, const int2 outputOffset,
                                        const int2 outputSize, GPU::Buffer<float2> outputCoordBuffer,
                                        GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(inputSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(inputSize.y, dimBlock.y), 1);
  backwardCoordLookupKernel<<<dimGrid, dimBlock, 0, stream>>>(inputOffset, inputSize, inputCoordBuffer.get(),
                                                              outputOffset, outputSize, outputCoordBuffer.get());
  return CUDA_STATUS;
}

struct BilinearFlowInterpolation {
  typedef float2 Type;

  static inline __device__ float2 interpolate(float2 a, float2 b, float2 c, float2 d) {
    if (a.x == INVALID_FLOW_VALUE || b.x == INVALID_FLOW_VALUE || c.x == INVALID_FLOW_VALUE ||
        d.x == INVALID_FLOW_VALUE) {
      return make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
    } else {
      return (9.0f / 16.0f * a + 3.0f / 16.0f * (b + c) + 1.0f / 16.0f * d);
    }
  }
};

Status OpticalFlow::upsampleFlow22(GPU::Buffer<float2> dst, GPU::Buffer<const float2> src, std::size_t dstWidth,
                                   std::size_t dstHeight, bool wrap, unsigned blockSize, GPU::Stream stream) {
  const unsigned srcWidth = ((unsigned)dstWidth + 1) / 2;
  const unsigned srcHeight = ((unsigned)dstHeight + 1) / 2;
  const dim3 dimBlock(blockSize, blockSize, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(srcWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(srcHeight, dimBlock.y), 1);
  if (wrap) {
    Image::upsample22Kernel<Image::HWrapBoundary<float2>, BilinearFlowInterpolation>
        <<<dimGrid, dimBlock, (blockSize + 2) * (blockSize + 2) * sizeof(float2), stream.get()>>>(
            dst.get(), src.get(), (unsigned)dstWidth, (unsigned)dstHeight, srcWidth, srcHeight);
  } else {
    Image::upsample22Kernel<Image::ExtendBoundary<float2>, BilinearFlowInterpolation>
        <<<dimGrid, dimBlock, (blockSize + 2) * (blockSize + 2) * sizeof(float2), stream.get()>>>(
            dst.get(), src.get(), (unsigned)dstWidth, (unsigned)dstHeight, srcWidth, srcHeight);
  }
  return CUDA_STATUS;
}

__global__ void outwardCoordLookupKernel(const int2 offset1, int2 size1, const float2* g_iCoord, const int2 offset0,
                                         const int2 size0, const uint32_t* g_iBuffer, uint32_t* g_oBuffer) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size1.x && y < size1.y) {
    g_oBuffer[y * size1.x + x] = 0;
    const float2 iCoord = g_iCoord[y * size1.x + x];
    if (iCoord.x != INVALID_FLOW_VALUE) {
      // const float2 outputCoord = make_float2(offset1.x - offset0.x + x + iCoord.x, offset1.y - offset0.y + y +
      // iCoord.y); if (inRange(outputCoord, size0))
      { g_oBuffer[y * size1.x + x] = g_iBuffer[y * size0.x + x]; }
    }
  }
}

Status OpticalFlow::outwardCoordLookup(const int2 offset1, const int2 size1,
                                       const GPU::Buffer<const float2> coordBuffer, const int2 offset0,
                                       const int2 size0, const GPU::Buffer<const uint32_t> inputBuffer,
                                       GPU::Buffer<uint32_t> outputBuffer, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size1.x, dimBlock.x), (unsigned)Cuda::ceilDiv(size1.y, dimBlock.y), 1);
  outwardCoordLookupKernel<<<dimGrid, dimBlock, 0, stream>>>(offset1, size1, coordBuffer.get(), offset0, size0,
                                                             inputBuffer.get(), outputBuffer.get());
  return CUDA_STATUS;
}

__global__ void forwardCoordLookupKernel(const int2 inputOffset, int2 inputSize, const float2* g_iCoord,
                                         const int2 originalOffset, const int2 originalSize,
                                         const float2* g_originalCoord, const int2 outputOffset, int2 outputSize,
                                         float2* g_oCoord) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < inputSize.x && y < inputSize.y) {
    const float2 iCoord = g_iCoord[y * inputSize.x + x];
    if (iCoord.x != INVALID_FLOW_VALUE) {
      const float2 inputCoord = make_float2(inputOffset.x + x, inputOffset.y + y);
      const float2 outputCoord = inputCoord + iCoord;

      const float2 relativeOutputCoord = outputCoord - make_float2(outputOffset.x, outputOffset.y);
      const float2 originalOutputcoord = outputCoord - make_float2(originalOffset.x, originalOffset.y);

      // Check the original flow value, in exist at all
      float2 originalFlow = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
      if (inRange(originalOutputcoord, originalSize)) {
        originalFlow = g_originalCoord[int(originalOutputcoord.y) * originalSize.x + int(originalOutputcoord.x)];
      }
      if (inRange(relativeOutputCoord, outputSize)) {
        if (originalFlow.x == INVALID_FLOW_VALUE) {
          g_oCoord[int(round(relativeOutputCoord.y)) * outputSize.x + int(round(relativeOutputCoord.x))] =
              make_float2(0, 0) - iCoord;
        } else {
          g_oCoord[int(round(relativeOutputCoord.y)) * outputSize.x + int(round(relativeOutputCoord.x))] = originalFlow;
        }
      }
    }
  }
}

Status OpticalFlow::forwardCoordLookup(const int2 inputOffset, const int2 inputSize,
                                       const GPU::Buffer<const float2> inputCoordBuffer, const int2 originalOffset,
                                       const int2 originalSize, const GPU::Buffer<const float2> originalCoordBuffer,
                                       const int2 outputOffset, const int2 outputSize,
                                       GPU::Buffer<float2> outputCoordBuffer, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(inputSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(inputSize.y, dimBlock.y), 1);
  forwardCoordLookupKernel<<<dimGrid, dimBlock, 0, stream>>>(inputOffset, inputSize, inputCoordBuffer.get(),
                                                             originalOffset, originalSize, originalCoordBuffer.get(),
                                                             outputOffset, outputSize, outputCoordBuffer.get());
  return CUDA_STATUS;
}

__global__ void putOverOriginalFlowKernel(const int2 inputOffset, const int2 inputSize, const float2* const inputFlow,
                                          const int2 outputOffset, const int2 outputSize, float2* outputFlow) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= inputSize.x || y >= inputSize.y) return;
  float2 input = inputFlow[y * inputSize.x + x];
  if (input.x != INVALID_FLOW_VALUE) {
    int2 outputCoord = make_int2(x, y) + inputOffset - outputOffset;
    if (inRange(outputCoord, outputSize)) {
      outputFlow[outputCoord.y * outputSize.x + outputCoord.x] = input;
    }
  }
}

Status OpticalFlow::putOverOriginalFlow(const int2 inputOffset, const int2 inputSize,
                                        const GPU::Buffer<const float2> inputFlow, const int2 outputOffset,
                                        const int2 outputSize, GPU::Buffer<float2> outputFlow, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(inputSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(inputSize.y, dimBlock.y), 1);
  putOverOriginalFlowKernel<<<dimGrid, dimBlock, 0, stream>>>(inputOffset, inputSize, inputFlow.get(), outputOffset,
                                                              outputSize, outputFlow.get());
  return CUDA_STATUS;
}

__global__ void identityFlowKernel(const bool normalizedFlow, const int2 size, float2* coordBuffer) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size.x || y >= size.y) return;
  if (normalizedFlow) {
    coordBuffer[y * size.x + x] = make_float2(float(x) / size.x, float(y) / size.y);
  } else {
    coordBuffer[y * size.x + x] = make_float2(x, y);
  }
}

Status OpticalFlow::generateIdentityFlow(const int2 size, GPU::Buffer<float2> coordBuffer, GPU::Stream gpuStream,
                                         const bool normalizedFlow) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, TILE_WIDTH), (unsigned)Cuda::ceilDiv(size.y, TILE_WIDTH), 1);
  identityFlowKernel<<<dimGrid, dimBlock, 0, stream>>>(normalizedFlow, size, coordBuffer.get());
  return CUDA_STATUS;
}

__global__ void transformOffsetToFlowKernel(const int2 size0, const int2 offset0, const int2 offset1,
                                            const float2* const inputBuffer, float2* const outputBuffer) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size0.x || y >= size0.y) return;
  float2 offset = inputBuffer[y * size0.x + x];
  if (offset.x != INVALID_FLOW_VALUE) {
    outputBuffer[y * size0.x + x] = offset + make_float2(x + offset0.x - offset1.x, y + offset0.y - offset1.y);
  } else {
    outputBuffer[y * size0.x + x] = offset;
  }
}

Status OpticalFlow::transformOffsetToFlow(const int2 size0, const int2 offset0, const int2 offset1,
                                          GPU::Buffer<float2> buffer, GPU::Stream gpuStream) {
  return transformOffsetToFlow(size0, offset0, offset1, buffer, buffer, gpuStream);
}

Status OpticalFlow::transformOffsetToFlow(const int2 size0, const int2 offset0, const int2 offset1,
                                          const GPU::Buffer<const float2> inputBuffer, GPU::Buffer<float2> outputBuffer,
                                          GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size0.x, TILE_WIDTH), (unsigned)Cuda::ceilDiv(size0.y, TILE_WIDTH), 1);
  transformOffsetToFlowKernel<<<dimGrid, dimBlock, 0, stream>>>(size0, offset0, offset1, inputBuffer.get(),
                                                                outputBuffer.get());
  return CUDA_STATUS;
}

__global__ void transformFlowToOffsetKernel(const int2 size0, const int2 offset0, const int2 offset1,
                                            const float2* const inputBuffer, float2* const outputBuffer) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size0.x || y >= size0.y) return;
  float2 flow = inputBuffer[y * size0.x + x];
  if (flow.x != INVALID_FLOW_VALUE)
    outputBuffer[y * size0.x + x] = flow - make_float2(x + offset0.x - offset1.x, y + offset0.y - offset1.y);
  else
    outputBuffer[y * size0.x + x] = flow;
}

Status OpticalFlow::transformFlowToOffset(const int2 size0, const int2 offset0, const int2 offset1,
                                          const GPU::Buffer<const float2> inputBuffer, GPU::Buffer<float2> outputBuffer,
                                          GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size0.x, TILE_WIDTH), (unsigned)Cuda::ceilDiv(size0.y, TILE_WIDTH), 1);
  transformFlowToOffsetKernel<<<dimGrid, dimBlock, 0, stream>>>(size0, offset0, offset1, inputBuffer.get(),
                                                                outputBuffer.get());
  return CUDA_STATUS;
}

__global__ void coordLookupKernel(int outputWidth, int outputHeight, const float2* g_iCoord, int inputWidth,
                                  int inputHeight, const uint32_t* g_idata, uint32_t* g_odata) {
  // calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < outputWidth && y < outputHeight) {
    float2 uv = g_iCoord[y * outputWidth + x];
    int2 size = make_int2(inputWidth, inputHeight);
    g_odata[y * outputWidth + x] = Image::bilinearLookup<Image::BilinearLookupRGBAtoRGBA>(uv, size, g_idata);
  }
}

Status OpticalFlow::coordLookup(const int outputWidth, const int outputHeight,
                                const GPU::Buffer<const float2> coordBuffer, const int inputWidth,
                                const int inputHeight, const GPU::Buffer<const uint32_t> inputBuffer,
                                GPU::Buffer<uint32_t> outputBuffer, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(outputWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(outputHeight, dimBlock.y), 1);
  coordLookupKernel<<<dimGrid, dimBlock, 0, stream>>>(outputWidth, outputHeight, coordBuffer.get(), inputWidth,
                                                      inputHeight, inputBuffer.get(), outputBuffer.get());
  return CUDA_STATUS;
}

__global__ void mulFlowOperatorKernel(float2* dst, const float2 toMul, std::size_t size) {
  std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    if (dst[i].x != INVALID_FLOW_VALUE) {
      dst[i] *= toMul;
    }
  }
}

Status OpticalFlow::mulFlowOperator(GPU::Buffer<float2> dst, const float2 toMul, std::size_t size, GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  mulFlowOperatorKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), toMul, size);
  return CUDA_STATUS;
}

__global__ void mulFlowOperatorKernel(float2* dst, const float2* src, const float2 toMul, std::size_t size) {
  std::size_t i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    if (dst[i].x != INVALID_FLOW_VALUE) {
      dst[i] = make_float2(src[i].x * toMul.x, src[i].y * toMul.y);
    }
  }
}

Status OpticalFlow::mulFlowOperator(GPU::Buffer<float2> dst, const GPU::Buffer<const float2> src, const float2 toMul,
                                    std::size_t size, GPU::Stream stream) {
  dim3 dimBlock(CUDABLOCKSIZE);
  dim3 dimGrid(Cuda::compute2DGridForFlatBuffer(size, CUDABLOCKSIZE));
  mulFlowOperatorKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(dst.get(), src.get(), toMul, size);
  return CUDA_STATUS;
}

__global__ void generateWeightKernel(const int kernelSize, const float sigmaDistance, float* kernelWeight) {
  int y = blockIdx.y * blockDim.x + threadIdx.y;
  int x = blockIdx.x * blockDim.y + threadIdx.x;
  if (x <= 2 * kernelSize && y <= 2 * kernelSize) {
    float maxDist = kernelSize * 1.4142;
    float distSpace = length(make_float2(x - kernelSize, y - kernelSize)) / maxDist;
    float weightDist = exp(-sigmaDistance * distSpace * distSpace);
    kernelWeight[y * (2 * kernelSize + 1) + x] = weightDist;
  }
}

__global__ void interCoordLookupKernel(const int warpWidth, const int interOffsetX, const int interOffsetY,
                                       const int interWidth, const int interHeight, const uint32_t* inputBuffer,
                                       const int coordWidth, const int coordHeight, const float2* coordBuffer,
                                       uint32_t* outputBuffer) {
  // calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < coordWidth && y < coordHeight) {
    float2 coord = coordBuffer[y * coordWidth + x] - make_float2(interOffsetX, interOffsetY);
    if (coord.x >= 0 && coord.x < interWidth && coord.y >= 0 && coord.y < interHeight) {
      outputBuffer[y * coordWidth + x] = inputBuffer[int(coord.y) * interWidth + int(coord.x)];
    } else {
      outputBuffer[y * coordWidth + x] = 0;
    }
  }
}

Status OpticalFlow::interCoordLookup(const int warpWidth, const int interOffsetX, const int interOffsetY,
                                     const int interWidth, const int interHeight,
                                     const GPU::Buffer<const uint32_t> inputBuffer, const int coordWidth,
                                     const int coordHeight, const GPU::Buffer<const float2> coordBuffer,
                                     GPU::Buffer<uint32_t> output, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(coordWidth, dimBlock.x), (unsigned)Cuda::ceilDiv(coordHeight, dimBlock.y), 1);
  interCoordLookupKernel<<<dimGrid, dimBlock, 0, stream>>>(warpWidth, interOffsetX, interOffsetY, interWidth,
                                                           interHeight, inputBuffer.get(), coordWidth, coordHeight,
                                                           coordBuffer.get(), output.get());
  return CUDA_STATUS;
}

__global__ void flowToRGBAKernel(const int2 size, const float2* inputBuffer, const int2 maxFlowValue,
                                 uint32_t* outputBuffer) {
  // calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    float2 input = inputBuffer[y * size.x + x];
    if (abs(input.x - INVALID_FLOW_VALUE) < 1e-5) {
      outputBuffer[y * size.x + x] = 0;
    } else {
      outputBuffer[y * size.x + x] =
          Image::RGBA::pack((float(input.x) / maxFlowValue.x) * 255, (float(input.y) / maxFlowValue.y) * 255, 0, 255);
    }
  }
}

Status OpticalFlow::convertFlowToRGBA(const int2 size, const GPU::Buffer<const float2> src, const int2 maxFlowValue,
                                      GPU::Buffer<uint32_t> dst, GPU::Stream stream) {
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  const dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, dimBlock.x), (unsigned)Cuda::ceilDiv(size.y, dimBlock.y), 1);
  flowToRGBAKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(size, src.get(), maxFlowValue, dst.get());
  return CUDA_STATUS;
}

__global__ void setAlphaToFlowBufferKernel(const int2 size, const uint32_t* colorBuffer, float2* flowBuffer) {
  // calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    const unsigned index = y * size.x + x;
    if (Image::RGBA::a(colorBuffer[index]) == 0) {
      flowBuffer[index] = make_float2(INVALID_FLOW_VALUE, INVALID_FLOW_VALUE);
    }
  }
}

Status OpticalFlow::setAlphaToFlowBuffer(const int2 size, const GPU::Buffer<const uint32_t> colorBuffer,
                                         GPU::Buffer<float2> flowBuffer, GPU::Stream gpuStream) {
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, dimBlock.x), (unsigned)Cuda::ceilDiv(size.y, dimBlock.y), 1);

  cudaStream_t stream = gpuStream.get();

  setAlphaToFlowBufferKernel<<<dimGrid, dimBlock, 0, stream>>>(size, colorBuffer.get(), flowBuffer.get());

  return CUDA_STATUS;
}

}  // namespace Util
}  // namespace VideoStitch
