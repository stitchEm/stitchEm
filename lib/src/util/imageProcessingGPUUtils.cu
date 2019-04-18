// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "imageProcessingGPUUtils.hpp"

#include "backend/common/imageOps.hpp"
#include "gpu/image/sampling.hpp"
#include "gpu/image/imageOps.hpp"
#include "gpu/image/blur.hpp"
#include "cuda/util.hpp"
#include "backend/cuda/deviceBuffer.hpp"
#include "backend/cuda/deviceStream.hpp"

#define INVALID_VALUE Image::RGBA::pack(1, 2, 3, 0)
#define REDUCE_THREADS_PER_BLOCK 512
#define REGULARIZATION_TILE_WIDTH 16
#define KERNEL_SIZE 25

#define TILE_WIDTH 16
#define CUDABLOCKSIZE 512

namespace VideoStitch {
namespace Util {

// Output lab in [0..255] range instead of the original [0..100, -128..127, -128..127]
inline __host__ __device__ uint32_t RGBandGradientToNormalizeLab(const uint32_t rgbAndGradient) {
  // Check if the pixel is without alpha
  if (rgbAndGradient == INVALID_VALUE) {
    return rgbAndGradient;
  }
  const float r = min(1.0f, float(Image::RGBA::r(rgbAndGradient)) / 255.0f);
  const float g = min(1.0f, float(Image::RGBA::g(rgbAndGradient)) / 255.0f);
  const float b = min(1.0f, float(Image::RGBA::b(rgbAndGradient)) / 255.0f);

  const float3 rgb = make_float3(r, g, b);
  const float3 lab = Image::rgbToLab(rgb);

  const uint32_t l_ui = uint32_t(lab.x * 2.55f);
  const uint32_t a_ui = uint32_t(lab.y + 128);
  const uint32_t b_ui = uint32_t(lab.z + 128);
  const uint32_t i_ui = Image::RGBA::a(rgbAndGradient);
  return Image::RGBA::pack(l_ui, a_ui, b_ui, i_ui);
}

inline __host__ __device__ uint32_t normalizeLabAndGradientToRGBA(const uint32_t normalizedLabAndGradient) {
  // Check if the pixel is without alpha
  if (normalizedLabAndGradient == INVALID_VALUE) {
    return 0;
  }
  const float l = float(Image::RGBA::r(normalizedLabAndGradient)) / 2.55f;
  const float a = float(Image::RGBA::g(normalizedLabAndGradient)) - 128.0f;
  const float b = float(Image::RGBA::b(normalizedLabAndGradient)) - 128.0f;

  const float3 lab = make_float3(l, a, b);
  const float3 rgb = Image::labToRGB(lab);

  const uint32_t red = uint32_t(min(1.0f, rgb.x) * 255.0f);
  const uint32_t green = uint32_t(min(1.0f, rgb.y) * 255.0f);
  const uint32_t blue = uint32_t(min(1.0f, rgb.z) * 255.0f);
  return Image::RGBA::pack(red, green, blue, 255);
}

__global__ void convertRGBAndGradientToNormalizedLABKernel(const int2 bufferSize, uint32_t* __restrict__ colorBuffer) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < bufferSize.x && y < bufferSize.y) {
    const unsigned index = y * bufferSize.x + x;
    colorBuffer[index] = RGBandGradientToNormalizeLab(colorBuffer[index]);
  }
}

Status ImageProcessingGPU::convertRGBandGradientToNormalizedLABandGradient(const int2 bufferSize,
                                                                           GPU::Buffer<uint32_t> colorBuffer,
                                                                           GPU::Stream stream) {
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(bufferSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(bufferSize.y, dimBlock.y), 1);
  convertRGBAndGradientToNormalizedLABKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(bufferSize, colorBuffer.get());
  return CUDA_STATUS;
}

__global__ void convertNormalizedLABandGradientToRGBAKernel(const int2 bufferSize, uint32_t* __restrict__ colorBuffer) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < bufferSize.x && y < bufferSize.y) {
    const unsigned index = y * bufferSize.x + x;
    colorBuffer[index] = normalizeLabAndGradientToRGBA(colorBuffer[index]);
  }
}

Status ImageProcessingGPU::convertNormalizedLABandGradientToRGBA(const int2 bufferSize,
                                                                 GPU::Buffer<uint32_t> colorBuffer,
                                                                 GPU::Stream stream) {
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(bufferSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(bufferSize.y, dimBlock.y), 1);
  convertNormalizedLABandGradientToRGBAKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(bufferSize, colorBuffer.get());
  return CUDA_STATUS;
}

// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/Data-Parallel_Algorithms.html#reduction
/*
This version adds multiple elements per thread sequentially.  This reduces the overall
cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
(Brent's Theorem optimization)
*/

/*
This version adds multiple elements per thread sequentially.  This reduces the overall
cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
(Brent's Theorem optimization)
*/

__global__ void reduce6(const float* g_idata, float* g_odata, float* g_omask, unsigned int n) {
  extern __shared__ float sharedData[];
  float* sdata = &sharedData[0];
  float* smask = &sharedData[blockDim.x];

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  unsigned int gridSize = blockDim.x * 2 * gridDim.x;
  sdata[tid] = 0;
  smask[tid] = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    sdata[tid] += g_idata[i];
    smask[tid] += (g_idata[i] > 0 ? 1 : 0);
    if (i + blockDim.x < n) {
      sdata[tid] += g_idata[i + blockDim.x];
      smask[tid] += (g_idata[i + blockDim.x] > 0 ? 1 : 0);
    }
    i += gridSize;
  }
  __syncthreads();

  // do reduction in shared mem
  if (blockDim.x >= 512) {
    if (tid < 256) {
      sdata[tid] += sdata[tid + 256];
      smask[tid] += smask[tid + 256];
    }
    __syncthreads();
  }
  if (blockDim.x >= 256) {
    if (tid < 128) {
      sdata[tid] += sdata[tid + 128];
      smask[tid] += smask[tid + 128];
    }
    __syncthreads();
  }
  if (blockDim.x >= 128) {
    if (tid < 64) {
      sdata[tid] += sdata[tid + 64];
      smask[tid] += smask[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    if (blockDim.x >= 64) {
      sdata[tid] += sdata[tid + 32];
      smask[tid] += smask[tid + 32];
    }
    if (blockDim.x >= 32) {
      sdata[tid] += sdata[tid + 16];
      smask[tid] += smask[tid + 16];
    }
    if (blockDim.x >= 16) {
      sdata[tid] += sdata[tid + 8];
      smask[tid] += smask[tid + 8];
    }
    if (blockDim.x >= 8) {
      sdata[tid] += sdata[tid + 4];
      smask[tid] += smask[tid + 4];
    }
    if (blockDim.x >= 4) {
      sdata[tid] += sdata[tid + 2];
      smask[tid] += smask[tid + 2];
    }
    if (blockDim.x >= 2) {
      sdata[tid] += sdata[tid + 1];
      smask[tid] += smask[tid + 1];
    }
  }

  // write result for this block to global mem
  if (tid == 0) {
    g_odata[blockIdx.x] = sdata[0];
    g_omask[blockIdx.x] = smask[0];
  }
}

Status ImageProcessingGPU::calculateSum(const int numElement, const GPU::Buffer<const float> buffer,
                                        const unsigned blockSize, GPU::Stream stream, float& output, float& mask) {
  int gridSize = (unsigned)Cuda::ceilDiv(numElement, blockSize);
  auto potOutBuffer = GPU::Buffer<float>::allocate(gridSize, "Merger Mask");
  PROPAGATE_FAILURE_STATUS(potOutBuffer.status());
  auto outBuffer = potOutBuffer.value();

  auto potMaskBuffer = GPU::Buffer<float>::allocate(gridSize, "Merger Mask");
  PROPAGATE_FAILURE_STATUS(potMaskBuffer.status());
  auto maskBuffer = potMaskBuffer.value();

  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid(gridSize, 1, 1);
  reduce6<<<dimGrid, dimBlock, blockSize * 2 * sizeof(float), stream.get()>>>(buffer.get(), outBuffer.get(),
                                                                              maskBuffer.get(), numElement);
  PROPAGATE_FAILURE_STATUS(stream.synchronize());

  std::vector<float> h_odata(gridSize);
  std::vector<float> h_omask(gridSize);
  if (gridSize > 0) {
    cudaMemcpy(h_odata.data(), outBuffer.get(), gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_omask.data(), maskBuffer.get(), gridSize * sizeof(float), cudaMemcpyDeviceToHost);
  }
  output = 0;
  mask = 0;
  for (int i = 0; i < gridSize; i++) {
    output += h_odata[i];
    mask += h_omask[i];
  }

  PROPAGATE_FAILURE_STATUS(outBuffer.release());
  PROPAGATE_FAILURE_STATUS(maskBuffer.release());
  return CUDA_STATUS;
}

__constant__ float kernel[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

__global__ void convertRGB210ToRGBandGradientKernel(int2 size, const uint32_t* __restrict__ inputBuffer,
                                                    uint32_t* __restrict__ colorBuffer) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    uint32_t oriInput = inputBuffer[y * size.x + x];
    if (!Image::RGB210::a(oriInput)) {
      colorBuffer[y * size.x + x] = INVALID_VALUE;
      return;
    }
    float Gx = 0;
    float Gy = 0;
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++) {
        int2 localPos = make_int2(dx + x, dy + y);
        if (localPos.x >= 0 && localPos.x < size.x && localPos.y >= 0 && localPos.y < size.y) {
          uint32_t input = inputBuffer[localPos.y * size.x + localPos.x];
          if (!Image::RGB210::a(input)) {
            colorBuffer[y * size.x + x] =
                Image::RGBA::pack(min(255, Image::RGB210::r(oriInput)), min(255, Image::RGB210::g(oriInput)),
                                  min(255, Image::RGB210::b(oriInput)), 0);
            return;
          }
          float c = (0.299f * float(Image::RGB210::r(input)) + 0.587f * float(Image::RGB210::g(input)) +
                     0.114f * float(Image::RGB210::b(input))) /
                    255.0f;
          float coefX = kernel[dx + 1][dy + 1];
          float coefY = kernel[dy + 1][dx + 1];
          Gx += coefX * c;
          Gy += coefY * c;
        }
      }
    uint32_t gradient = (unsigned char)(min(255.0f, sqrt(Gx * Gx + Gy * Gy) * 255.0f));
    colorBuffer[y * size.x + x] =
        Image::RGBA::pack(min(255, Image::RGB210::r(oriInput)), min(255, Image::RGB210::g(oriInput)),
                          min(255, Image::RGB210::b(oriInput)), gradient);
  }
}

Status ImageProcessingGPU::convertRGB210ToRGBandGradient(const int2 bufferSize,
                                                         const GPU::Buffer<const uint32_t> inputBuffer,
                                                         GPU::Buffer<uint32_t> colorBuffer, GPU::Stream stream) {
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(bufferSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(bufferSize.y, dimBlock.y), 1);
  convertRGB210ToRGBandGradientKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(bufferSize, inputBuffer.get(),
                                                                              colorBuffer.get());
  return CUDA_STATUS;
}

__global__ void convertRGB210ToRGBAKernel(int2 size, uint32_t* __restrict__ colorBuffer) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    uint32_t oriInput = colorBuffer[y * size.x + x];
    colorBuffer[y * size.x + x] = Image::RGBA::pack(Image::RGB210::r(oriInput), Image::RGB210::g(oriInput),
                                                    Image::RGB210::b(oriInput), Image::RGB210::a(oriInput));
  }
}

Status ImageProcessingGPU::convertRGB210ToRGBA(const int2 bufferSize, GPU::Buffer<uint32_t> buffer,
                                               GPU::Stream stream) {
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(bufferSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(bufferSize.y, dimBlock.y), 1);
  convertRGB210ToRGBAKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(bufferSize, buffer.get());
  return CUDA_STATUS;
}

__global__ void extractChannelKernel(int2 bufferSize, const int channelIndex,
                                     const uint32_t* const __restrict__ colorBuffer,
                                     unsigned char* __restrict__ outputBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < bufferSize.x && y < bufferSize.y) {
    const unsigned index = y * bufferSize.x + x;
    const uint32_t color = colorBuffer[index];
    if (channelIndex == 0)
      outputBuffer[index] = Image::RGBA::r(color);
    else if (channelIndex == 1)
      outputBuffer[index] = Image::RGBA::g(color);
    else if (channelIndex == 2)
      outputBuffer[index] = Image::RGBA::b(color);
    else if (channelIndex == 3)
      outputBuffer[index] = Image::RGBA::a(color);
  }
}

Status ImageProcessingGPU::extractChannel(const int2 bufferSize, const GPU::Buffer<const uint32_t> inputBuffer,
                                          const int channelIndex, GPU::Buffer<unsigned char> outputBuffer,
                                          GPU::Stream stream) {
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(bufferSize.x, dimBlock.x), (unsigned)Cuda::ceilDiv(bufferSize.y, dimBlock.y), 1);
  extractChannelKernel<<<dimGrid, dimBlock, 0, stream.get()>>>(bufferSize, channelIndex, inputBuffer.get(),
                                                               outputBuffer.get());
  return CUDA_STATUS;
}

__global__ void gradientKernel(int2 size, const uint32_t* inputBuffer, float* outputGradientBuffer) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    float Gx = 0;
    float Gy = 0;
    float wX = 0;
    float wY = 0;
    for (int dy = -1; dy <= 1; dy++)
      for (int dx = -1; dx <= 1; dx++) {
        int2 localPos = make_int2(dx + x, dy + y);
        if (inRange(localPos, size)) {
          uint32_t input = inputBuffer[localPos.y * size.x + localPos.x];
          if (Image::RGBA::a(input) == 0) {
            outputGradientBuffer[y * size.x + x] = 0;
            return;
          }
          float c = (0.299f * float(Image::RGBA::r(input)) + 0.587f * float(Image::RGBA::g(input)) +
                     0.114f * float(Image::RGBA::b(input))) /
                    255.0f;
          float coefX = kernel[dx + 1][dy + 1];
          float coefY = kernel[dy + 1][dx + 1];
          Gx += coefX * c;
          Gy += coefY * c;
          wX += coefX;
          wY += coefY;
        }
      }
    outputGradientBuffer[y * size.x + x] = sqrt(Gx * Gx + Gy * Gy);
  }
}

Status ImageProcessingGPU::findGradient(const int2 size, const GPU::Buffer<const uint32_t> inputBuffer,
                                        GPU::Buffer<float> outputGradientBuffer, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, dimBlock.x), (unsigned)Cuda::ceilDiv(size.y, dimBlock.y), 1);
  gradientKernel<<<dimGrid, dimBlock, 0, stream>>>(size, inputBuffer.get(), outputGradientBuffer.get());
  return CUDA_STATUS;
}

__global__ void luminanceKernel(const int2 size, const uint32_t* const inputBuffer,
                                float* const outputLuminanceBuffer) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size.x || y >= size.y) return;
  uint32_t input = inputBuffer[y * size.x + x];
  outputLuminanceBuffer[y * size.x + x] =
      (0.299f * float(Image::RGBA::r(input)) + 0.587f * float(Image::RGBA::g(input)) +
       0.114f * float(Image::RGBA::b(input))) /
      255.0f;
}

Status ImageProcessingGPU::findLuminance(const int2 size, const GPU::Buffer<const uint32_t> inputBuffer,
                                         GPU::Buffer<float> outputLuminanceBuffer, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, dimBlock.x), (unsigned)Cuda::ceilDiv(size.y, dimBlock.y), 1);
  luminanceKernel<<<dimGrid, dimBlock, 0, stream>>>(size, inputBuffer.get(), outputLuminanceBuffer.get());
  return CUDA_STATUS;
}

// Output lab in [0..1] range instead of the original [0..100, -128..127, -128..127]
inline __host__ __device__ uint32_t rgbToNormalizeLab(const uint32_t rgba) {
  const float r = float(Image::RGBA::r(rgba)) / 255;
  const float g = float(Image::RGBA::g(rgba)) / 255;
  const float b = float(Image::RGBA::b(rgba)) / 255;

  const float3 rgb = make_float3(r, g, b);
  const float3 lab = Image::rgbToLab(rgb);

  const uint32_t l_ui = uint32_t(lab.x * 2.55f);
  const uint32_t a_ui = uint32_t(lab.y + 128);
  const uint32_t b_ui = uint32_t(lab.z + 128);
  const uint32_t a = Image::RGBA::a(rgba);
  return Image::RGBA::pack(l_ui, a_ui, b_ui, a);
}

__global__ void rgbToNormalizeLabKernel(const int2 size, const uint32_t* const inputRGBBuffer,
                                        uint32_t* const outputNormalizedLABBuffer) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size.x || y >= size.y) return;
  uint32_t input = inputRGBBuffer[y * size.x + x];
  outputNormalizedLABBuffer[y * size.x + x] = rgbToNormalizeLab(input);
}

Status ImageProcessingGPU::convertRGBToNormalizedLAB(const int2 size, const GPU::Buffer<const uint32_t> inputRGBBuffer,
                                                     GPU::Buffer<uint32_t> outputNormalizedLABBuffer,
                                                     GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, dimBlock.x), (unsigned)Cuda::ceilDiv(size.y, dimBlock.y), 1);
  rgbToNormalizeLabKernel<<<dimGrid, dimBlock, 0, stream>>>(size, inputRGBBuffer.get(),
                                                            outputNormalizedLABBuffer.get());
  return CUDA_STATUS;
}

__global__ void buffer2DRGBACompactBlendOffsetOperatorKernel(int offsetX, int offsetY, int width, int height,
                                                             uint32_t* __restrict__ dst, float w0, int offsetX0,
                                                             int offsetY0, int width0, int height0,
                                                             const uint32_t* __restrict__ src0, float w1, int offsetX1,
                                                             int offsetY1, int width1, int height1,
                                                             const uint32_t* __restrict__ src1) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int localIndex = width * y + x;
    float w = 0;
    float r = 0;
    float g = 0;
    float b = 0;
    float a = 0;
    dst[localIndex] = 0;
    if (offsetY + y - offsetY0 >= 0 && offsetY + y - offsetY0 < height0 && offsetX + x - offsetX0 >= 0 &&
        offsetX + x - offsetX0 < width0) {
      int localIndex0 = (offsetY + y - offsetY0) * width0 + (offsetX + x - offsetX0);
      if (Image::RGBA::a(src0[localIndex0]) > 0) {
        w += w0;
        r += w0 * Image::RGBA::r(src0[localIndex0]);
        g += w0 * Image::RGBA::g(src0[localIndex0]);
        b += w0 * Image::RGBA::b(src0[localIndex0]);
        a += w0 * Image::RGBA::a(src0[localIndex0]);
      }
    }
    if (offsetY + y - offsetY1 >= 0 && offsetY + y - offsetY1 < height1 && offsetX + x - offsetX1 >= 0 &&
        offsetX + x - offsetX1 < width1) {
      int localIndex1 = (offsetY + y - offsetY1) * width1 + (offsetX + x - offsetX1);
      if (Image::RGBA::a(src1[localIndex1]) > 0) {
        w += w1;
        r += w1 * Image::RGBA::r(src1[localIndex1]);
        g += w1 * Image::RGBA::g(src1[localIndex1]);
        b += w1 * Image::RGBA::b(src1[localIndex1]);
        a += w1 * Image::RGBA::a(src1[localIndex1]);
      }
    }
    if (w > 0) {
      dst[localIndex] = Image::RGBA::pack(r / w, g / w, b / w, 255);
    } else {
      dst[localIndex] = 0;
    }
  }
}

Status ImageProcessingGPU::buffer2DRGBACompactBlendOffsetOperator(const Core::Rect& dstRect, GPU::Buffer<uint32_t> dst,
                                                                  const float weight0, const Core::Rect& src0Rect,
                                                                  const GPU::Buffer<const uint32_t> src0,
                                                                  const float weight1, const Core::Rect& src1Rect,
                                                                  const GPU::Buffer<const uint32_t> src1,
                                                                  GPU::Stream gpuStream) {
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(dstRect.getWidth(), dimBlock.x),
               (unsigned)Cuda::ceilDiv(dstRect.getHeight(), dimBlock.y), 1);
  cudaStream_t stream = gpuStream.get();
  buffer2DRGBACompactBlendOffsetOperatorKernel<<<dimGrid, dimBlock, 0, stream>>>(
      (int)dstRect.left(), (int)dstRect.top(), (int)dstRect.getWidth(), (int)dstRect.getHeight(), dst.get(), weight0,
      (int)src0Rect.left(), (int)src0Rect.top(), (int)src0Rect.getWidth(), (int)src0Rect.getHeight(), src0.get(),
      weight1, (int)src1Rect.left(), (int)src1Rect.top(), (int)src1Rect.getWidth(), (int)src1Rect.getHeight(),
      src1.get());
  return CUDA_STATUS;
}

__global__ void binarizeMaskKernel(const int2 size, const uint32_t* inputBuffer, uint32_t* binarizedBuffer) {
  // calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    binarizedBuffer[y * size.x + x] = inputBuffer[y * size.x + x] > 0 ? 1 : 0;
  }
}

Status ImageProcessingGPU::binarizeMask(const int2 size, const GPU::Buffer<const uint32_t> inputMask,
                                        GPU::Buffer<uint32_t> binarizedMask, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, dimBlock.x), (unsigned)Cuda::ceilDiv(size.y, dimBlock.y), 1);
  binarizeMaskKernel<<<dimGrid, dimBlock, 0, stream>>>(size, inputMask.get(), binarizedMask.get());
  return CUDA_STATUS;
}

__global__ void onBothBufferOperatorKernel(const int warpWidth, const int input0OffsetX, const int input0OffsetY,
                                           const int input0Width, const int input0Height, const uint32_t* input0Buffer,
                                           const int input1OffsetX, const int input1OffsetY, const int input1Width,
                                           const int input1Height, const uint32_t* input1Buffer,
                                           const int outputOffsetX, const int outputOffsetY, const int outputWidth,
                                           const int outputHeight, uint32_t* outputMask) {
  // calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < outputWidth && y < outputHeight) {
    uint32_t v = 0;
    const int outputX = x + outputOffsetX;
    const int outputY = y + outputOffsetY;
    const int input0X = (outputX + warpWidth - input0OffsetX) % warpWidth;
    const int input0Y = (outputY - input0OffsetY);
    const int input1X = (outputX + warpWidth - input1OffsetX) % warpWidth;
    const int input1Y = (outputY - input1OffsetY);
    if (input1X >= 0 && input1X < input1Width && input1Y >= 0 && input1Y < input1Height && input0X >= 0 &&
        input0X < input0Width && input0Y >= 0 && input0Y < input0Height) {
      if (input0Buffer[input0Y * input0Width + input0X] > 0 && input1Buffer[input1Y * input1Width + input1X] > 0) {
        v = 1;
      } else {
        v = 0;
      }
    }
    outputMask[y * outputWidth + x] = v;
  }
}

Status ImageProcessingGPU::onBothBufferOperator(const int warpWidth, const Core::Rect boundingRect0,
                                                const GPU::Buffer<const uint32_t> buffer0,
                                                const Core::Rect boundingRect1,
                                                const GPU::Buffer<const uint32_t> buffer1,
                                                const Core::Rect boundingRectBuffer, GPU::Buffer<uint32_t> buffer,
                                                GPU::Stream gpuStream) {
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(boundingRectBuffer.getWidth(), dimBlock.x),
               (unsigned)Cuda::ceilDiv(boundingRectBuffer.getHeight(), dimBlock.y), 1);
  cudaStream_t stream = gpuStream.get();

  onBothBufferOperatorKernel<<<dimGrid, dimBlock, 0, stream>>>(
      (int)warpWidth, (int)boundingRect0.left(), (int)boundingRect0.top(), (int)boundingRect0.getWidth(),
      (int)boundingRect0.getHeight(), buffer0.get(), (int)boundingRect1.left(), (int)boundingRect1.top(),
      (int)boundingRect1.getWidth(), (int)boundingRect1.getHeight(), buffer1.get(), (int)boundingRectBuffer.left(),
      (int)boundingRectBuffer.top(), (int)boundingRectBuffer.getWidth(), (int)boundingRectBuffer.getHeight(),
      buffer.get());

  return CUDA_STATUS;
}

template <typename T>
__global__ void constantBufferKernel(const int2 size, T* buffer, T value) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= size.x || y >= size.y) return;
  buffer[y * size.x + x] = value;
}

template <typename T>
Status ImageProcessingGPU::setConstantBuffer(const int2 size, GPU::Buffer<T> buffer, const T value,
                                             GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, TILE_WIDTH), (unsigned)Cuda::ceilDiv(size.y, TILE_WIDTH), 1);
  constantBufferKernel<T><<<dimGrid, dimBlock, 0, stream>>>(size, buffer.get(), value);
  return CUDA_STATUS;
}

template Status ImageProcessingGPU::setConstantBuffer(const int2 size, GPU::Buffer<float2> buffer, const float2 value,
                                                      GPU::Stream gpuStream);
template Status ImageProcessingGPU::setConstantBuffer(const int2 size, GPU::Buffer<uint32_t> buffer,
                                                      const uint32_t value, GPU::Stream gpuStream);

template <typename T>
__global__ void packBufferKernel(const int wrapWidth, const T invalidValue, const int inputOffsetX,
                                 const int inputOffsetY, const int inputWidth, const int inputHeight, const T* input,
                                 const int packedOffsetX, const int packedOffsetY, const int packedWidth,
                                 const int packedHeight, T* output) {
  // calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < packedWidth && y < packedHeight) {
    const int inputX = (x + packedOffsetX - inputOffsetX + wrapWidth) % wrapWidth;
    const int inputY = (y + packedOffsetY - inputOffsetY);
    if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight) {
      output[y * packedWidth + x] = input[inputY * inputWidth + inputX];
    } else {
      output[y * packedWidth + x] = invalidValue;
    }
  }
}

template <typename T>
Status ImageProcessingGPU::packBuffer(const int warpWidth, const T invalidValue, const Core::Rect inputRect,
                                      const GPU::Buffer<const T> inputBuffer, const Core::Rect outputRect,
                                      GPU::Buffer<T> outputBuffer, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(outputRect.getWidth(), dimBlock.x),
               (unsigned)Cuda::ceilDiv(outputRect.getHeight(), dimBlock.y), 1);
  packBufferKernel<T><<<dimGrid, dimBlock, 0, stream>>>(
      warpWidth, invalidValue, (int)inputRect.left(), (int)inputRect.top(), (int)inputRect.getWidth(),
      (int)inputRect.getHeight(), inputBuffer.get(), (int)outputRect.left(), (int)outputRect.top(),
      (int)outputRect.getWidth(), (int)outputRect.getHeight(), outputBuffer.get());
  return CUDA_STATUS;
}

template Status ImageProcessingGPU::packBuffer(const int warpWidth, const uint32_t invalidValue,
                                               const Core::Rect inputRect,
                                               const GPU::Buffer<const uint32_t> inputBuffer,
                                               const Core::Rect outputRect, GPU::Buffer<uint32_t> outputBuffer,
                                               GPU::Stream gpuStream);
template Status ImageProcessingGPU::packBuffer(const int warpWidth, const float2 invalidValue,
                                               const Core::Rect inputRect, const GPU::Buffer<const float2> inputBuffer,
                                               const Core::Rect outputRect, GPU::Buffer<float2> outputBuffer,
                                               GPU::Stream gpuStream);

template <typename T>
__global__ void thresholdingBufferKernel(const int2 size, const T thresholdValue, const T minBoundValue,
                                         const T maxBoundValue, T* buffer) {
  // calculate normalized texture coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < size.x && y < size.y) {
    const int index = y * size.x + x;
    const T value = buffer[index];
    if (value <= thresholdValue) {
      buffer[index] = minBoundValue;
    } else {
      buffer[index] = maxBoundValue;
    }
  }
}

template <typename T>
Status ImageProcessingGPU::thresholdingBuffer(const int2 size, const T thresholdValue, const T minBoundValue,
                                              const T maxBoundValue, GPU::Buffer<T> inputBuffer,
                                              GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((unsigned)Cuda::ceilDiv(size.x, dimBlock.x), (unsigned)Cuda::ceilDiv(size.y, dimBlock.y), 1);
  thresholdingBufferKernel<T>
      <<<dimGrid, dimBlock, 0, stream>>>(size, thresholdValue, minBoundValue, maxBoundValue, inputBuffer.get());
  return CUDA_STATUS;
}

template Status ImageProcessingGPU::thresholdingBuffer(const int2 size, const unsigned char thresholdValue,
                                                       const unsigned char minBoundValue,
                                                       const unsigned char maxBoundValue,
                                                       GPU::Buffer<unsigned char> inputBuffer, GPU::Stream gpuStream);

}  // namespace Util
}  // namespace VideoStitch
