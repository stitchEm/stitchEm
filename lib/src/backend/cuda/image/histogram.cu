// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "image/histogram.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceBuffer2D.hpp"
#include "../deviceStream.hpp"

#include "backend/common/imageOps.hpp"

namespace VideoStitch {
namespace Image {

/**
 * This kernel computes the RGB histograms of a RGBA8888 video frame.
 * Expects 16x16 thread blocks.
 */
__global__ void rgbHistogramKernel(const uint32_t* __restrict__ frame, int64_t size, uint32_t* rHist, uint32_t* gHist,
                                   uint32_t* bHist) {
  uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t stride = blockDim.x * gridDim.x;

  __shared__ unsigned r[256];
  __shared__ unsigned g[256];
  __shared__ unsigned b[256];
  r[threadIdx.x] = 0;
  g[threadIdx.x] = 0;
  b[threadIdx.x] = 0;
  __syncthreads();
  while (i < size) {
    atomicAdd(&r[RGBA::r(frame[i])], 1);
    atomicAdd(&g[RGBA::g(frame[i])], 1);
    atomicAdd(&b[RGBA::b(frame[i])], 1);
    i += stride;
  }
  __syncthreads();
  atomicAdd(&rHist[threadIdx.x], r[threadIdx.x]);
  atomicAdd(&gHist[threadIdx.x], g[threadIdx.x]);
  atomicAdd(&bHist[threadIdx.x], b[threadIdx.x]);
}

/**
 * This kernel computes the RGB histograms of a RGBA8888 video frame.
 * Expects 16x16 thread blocks.
 */
Status rgbHistogram(GPU::Buffer<const uint32_t> frame, int64_t width, int64_t height, GPU::Buffer<uint32_t> rHist,
                    GPU::Buffer<uint32_t> gHist, GPU::Buffer<uint32_t> bHist, GPU::Stream stream) {
  cudaMemset(rHist.get().raw(), 0, 256 * sizeof(uint32_t));
  cudaMemset(gHist.get().raw(), 0, 256 * sizeof(uint32_t));
  cudaMemset(bHist.get().raw(), 0, 256 * sizeof(uint32_t));

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int blocks = prop.multiProcessorCount;

  dim3 threadsPerBlock(256, 1, 1);
  dim3 blocksPerGrid(blocks * 2, 1, 1);
  rgbHistogramKernel<<<blocksPerGrid, threadsPerBlock, 0, stream.get()>>>(frame.get(), width * height, rHist.get(),
                                                                          gHist.get(), bHist.get());
  return CUDA_STATUS;
}

/**
 * This kernel computes the relative distribution of all luma values in a grayscale video frame.
 */
__global__ void lumaHistogramKernel(const unsigned char* __restrict__ frame, size_t width, size_t height, size_t pitch,
                                    uint32_t* __restrict__ hist) {
  uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t stride = blockDim.x * gridDim.x;

  __shared__ uint32_t l[256];
  l[threadIdx.x] = 0;
  __syncthreads();
  while (i < width * height) {
    uint32_t y = i / width;
    uint32_t x = i - y * width;
    atomicAdd(&l[frame[y * pitch + x]], 1);
    i += stride;
  }
  __syncthreads();
  atomicAdd(&hist[threadIdx.x], l[threadIdx.x]);
}

/**
 * This kernel computes the relative distribution of all luma values in a grayscale video frame.
 */
Status lumaHistogram(GPU::Buffer2D frame, frameid_t frameid, GPU::Buffer<uint32_t> hist, GPU::Stream stream) {
  cudaMemset(hist.get().raw() + 256 * frameid, 0, 256 * sizeof(uint32_t));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int blocks = prop.multiProcessorCount;
  dim3 threadsPerBlock(256, 1, 1);
  dim3 blocksPerGrid(blocks * 96, 1, 1);
  lumaHistogramKernel<<<blocksPerGrid, threadsPerBlock, 0, stream.get()>>>(
      frame.get(), frame.getWidth(), frame.getHeight(), frame.getPitch(), hist.get().raw() + 256 * frameid);
  return CUDA_STATUS;
}

}  // namespace Image
}  // namespace VideoStitch
