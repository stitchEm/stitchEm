// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "../../common/core1/voronoiKernel.hpp"

#include "../deviceBuffer.hpp"
#include "../deviceStream.hpp"
#include "kernels/voronoiKernel.cu"

#include "cuda/util.hpp"

#include <algorithm>
#include <cassert>

#define JFA_ADDITIONAL_PASSES 4
#define JFA_ADDITIONAL_PASSES_BASE 5

namespace {
static const int CudaBlockSize = 16;
}

namespace VideoStitch {
namespace Core {

Status voronoiInit(GPU::Buffer<uint32_t> buffer, std::size_t width, std::size_t height, uint32_t blackMask,
                   uint32_t whiteMask, unsigned blockSize, GPU::Stream gpuStream) {
  cudaStream_t stream = gpuStream.get();
  dim3 dimBlock2D(blockSize, blockSize, 1);
  assert((width % dimBlock2D.x) == 0);
  assert((height % dimBlock2D.x) == 0);
  dim3 dimGrid2D((unsigned)width / dimBlock2D.x, (unsigned)height / dimBlock2D.y, 1);
  voronoiInitKernel<<<dimGrid2D, dimBlock2D, 0, stream>>>(buffer.get(), (unsigned)width, (unsigned)height, blackMask,
                                                          whiteMask);
  return CUDA_STATUS;
}

Status voronoiComputeEuclidean(GPU::Buffer<uint32_t> dst, GPU::Buffer<uint32_t> src, std::size_t width,
                               std::size_t height, uint32_t step, bool hWrap, unsigned blockSize, GPU::Stream stream) {
  dim3 dimBlock2D(blockSize, blockSize, 1);
  dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y), 1);

  PanoRegion region;
  region.panoDim = {-1};
  region.viewLeft = 0;
  region.viewTop = 0;
  region.viewWidth = (int32_t)width;
  region.viewHeight = (int32_t)height;

  if (hWrap) {
    voronoiCompute_Wrap_distSqr<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get(), src.get(), region, step);
  } else {
    voronoiCompute_NoWrap_distSqr<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get(), src.get(), region, step);
  }
  return CUDA_STATUS;
}

Status voronoiComputeErect(GPU::Buffer<uint32_t> dst, GPU::Buffer<uint32_t> src, const PanoRegion& region,
                           uint32_t step, bool hWrap, unsigned blockSize, GPU::Stream stream) {
  dim3 dimBlock2D(blockSize, blockSize, 1);
  dim3 dimGrid2D((unsigned)Cuda::ceilDiv(region.viewWidth, dimBlock2D.x),
                 (unsigned)Cuda::ceilDiv(region.viewHeight, dimBlock2D.y), 1);

  if (hWrap) {
    voronoiCompute_Wrap_distSphere<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get(), src.get(), region, step);
  } else {
    voronoiCompute_NoWrap_distSphere<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get(), src.get(), region, step);
  }
  return CUDA_STATUS;
}

Status voronoiMakeMask(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, std::size_t width, std::size_t height,
                       unsigned blockSize, GPU::Stream stream) {
  dim3 dimBlock2D(blockSize, blockSize, 1);
  assert((width % dimBlock2D.x) == 0);
  assert((height % dimBlock2D.x) == 0);
  dim3 dimGrid2D((unsigned)width / dimBlock2D.x, (unsigned)height / dimBlock2D.y, 1);
  voronoiMakeMaskKernel<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)width,
                                                                    (unsigned)height);
  return CUDA_STATUS;
}

Status initForMaskComputation(GPU::Buffer<uint32_t> dst, GPU::Buffer<const uint32_t> buf, std::size_t width,
                              std::size_t height, uint32_t mask, uint32_t otherMask, GPU::Stream stream) {
  dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y), 1);
  edtInit<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get(), buf.get(), (unsigned)width, (unsigned)height, mask,
                                                      otherMask);
  return CUDA_STATUS;
}

Status makeMaskErect(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> blackResult,
                     GPU::Buffer<uint32_t> whiteResult, const PanoRegion& region, bool hWrap,
                     float maxTransitionDistance, float power, GPU::Stream stream) {
  dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  assert((region.viewWidth % dimBlock2D.x) == 0);
  assert((region.viewHeight % dimBlock2D.x) == 0);
  dim3 dimGrid2D((unsigned)region.viewWidth / dimBlock2D.x, (unsigned)region.viewHeight / dimBlock2D.y, 1);
  if (hWrap) {
    buildTransitionMask_Wrap_distSphere<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(
        dst.get(), blackResult.get(), whiteResult.get(), region, maxTransitionDistance, power);
  } else {
    buildTransitionMask_NoWrap_distSphere<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(
        dst.get(), blackResult.get(), whiteResult.get(), region, maxTransitionDistance, power);
  }
  return CUDA_STATUS;
}

Status makeMaskEuclidean(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> blackResult,
                         GPU::Buffer<uint32_t> whiteResult, const PanoRegion& region, bool hWrap,
                         float maxTransitionDistance, float power, GPU::Stream stream) {
  dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  assert((region.viewWidth % dimBlock2D.x) == 0);
  assert((region.viewHeight % dimBlock2D.x) == 0);
  dim3 dimGrid2D((unsigned)region.viewWidth / dimBlock2D.x, (unsigned)region.viewHeight / dimBlock2D.y, 1);
  if (hWrap) {
    buildTransitionMask_Wrap_distSqr<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(
        dst.get(), blackResult.get(), whiteResult.get(), region, maxTransitionDistance, power);
  } else {
    buildTransitionMask_NoWrap_distSqr<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(
        dst.get(), blackResult.get(), whiteResult.get(), region, maxTransitionDistance, power);
  }
  return CUDA_STATUS;
}

Status setInitialImageMask(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> src, std::size_t width,
                           std::size_t height, uint32_t fromIdMask, GPU::Stream stream) {
  dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  // FIXME: make sure this holds ?
  assert((width % dimBlock2D.x) == 0);
  assert((height % dimBlock2D.x) == 0);
  dim3 dimGrid2D((unsigned)width / dimBlock2D.x, (unsigned)height / dimBlock2D.y, 1);
  edtReflexiveKernel<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(dst.get(), src.get(), (unsigned)width,
                                                                 (unsigned)height, fromIdMask);
  return CUDA_STATUS;
}

Status extractEuclideanDist(GPU::Buffer<unsigned char> dst, GPU::Buffer<uint32_t> whiteResult, std::size_t width,
                            std::size_t height, bool hWrap, float maxTransitionDistance, float power,
                            GPU::Stream stream) {
  dim3 dimBlock2D(CudaBlockSize, CudaBlockSize, 1);
  dim3 dimGrid2D((unsigned)Cuda::ceilDiv(width, dimBlock2D.x), (unsigned)Cuda::ceilDiv(height, dimBlock2D.y), 1);

  if (hWrap) {
    extractDistKernel_Wrap_distSqr<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(
        dst.get(), whiteResult.get(), (unsigned)width, (unsigned)height, maxTransitionDistance, power);
  } else {
    extractDistKernel_NoWrap_distSqr<<<dimGrid2D, dimBlock2D, 0, stream.get()>>>(
        dst.get(), whiteResult.get(), (unsigned)width, (unsigned)height, maxTransitionDistance, power);
  }
  return CUDA_STATUS;
}

}  // namespace Core
}  // namespace VideoStitch
